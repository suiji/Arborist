// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file index.cc

   @brief Methods for maintaining the index-tree representation of splitable nodes.

   @author Mark Seligman
 */

#include "index.h"
#include "bv.h"
#include "pretree.h"
#include "sample.h"
#include "samplepred.h"
#include "splitnux.h"
#include "splitnode.h"
#include "bottom.h"
#include "path.h"
#include "ompthread.h"

#include <numeric>


unsigned int IndexLevel::totLevels = 0;
unsigned int IndexLevel::minNode = 0;

void IndexLevel::immutables(unsigned int _minNode, unsigned int _totLevels) {
  minNode = _minNode;
  totLevels = _totLevels;
}


void IndexLevel::deImmutables() {
  totLevels = 0;
  minNode = 0;
}


void IndexSet::decr(vector<SumCount> &_ctgSum, const vector<SumCount> &_ctgSub) {
  unsigned i = 0;
  for (auto & sc : _ctgSum) {
    sc.decr(_ctgSub[i++]);
  }
}


IndexLevel::~IndexLevel() {
}


IndexSet::IndexSet() :
  splitIdx(0),
  ptId(0),
  lhStart(0),
  extent(0),
  sCount(0),
  sum(0.0),
  minInfo(0.0),
  path(0),
  unsplitable(false),
  sumExpl(0.0){
}


shared_ptr<PreTree> IndexLevel::oneTree(const SummaryFrame* frame,
                                        const Sample *sample) {
  auto index = make_unique<IndexLevel>(frame, sample);
  return index->levels(frame, sample);
}



IndexLevel::IndexLevel(const SummaryFrame* frame,
                       const Sample* sample) :
  samplePred(sample->predictors()),
  indexSet(vector<IndexSet>(1)),
  bagCount(sample->getBagCount()),
  bottom(make_unique<Bottom>(frame, bagCount)),
  nodeRel(false),
  idxLive(bagCount),
  relBase(vector<IndexType>(1)),
  rel2ST(vector<unsigned int>(bagCount)),
  st2Split(vector<unsigned int>(bagCount)),
  st2PT(vector<unsigned int>(bagCount)),
  replayExpl(make_unique<BV>(bagCount)) {
  indexSet[0].initRoot(sample);
  relBase[0] = 0;
  iota(rel2ST.begin(), rel2ST.end(), 0);
  fill(st2Split.begin(), st2Split.end(), 0);
  fill(st2PT.begin(), st2PT.end(), 0);
}

void IndexSet::initRoot(const Sample* sample) {
  splitIdx = 0;
  sCount = sample->getNSamp();
  lhStart = 0;
  extent = sample->getBagCount();
  minInfo = 0.0;
  ptId = 0;
  sum = sample->getBagSum();
  path = 0;
  relBase = 0;
  ctgSum = sample->getCtgRoot();
  ctgExpl = vector<SumCount>(ctgSum.size());

  initInattainable(sample->getBagCount());
}


shared_ptr<PreTree> IndexLevel::levels(const SummaryFrame* frame,
                                       const Sample* sample) {
  bottom->rootDef(sample->stage(samplePred.get()), bagCount);
  shared_ptr<PreTree> preTree = make_shared<PreTree>(frame, bagCount);
  unique_ptr<SplitNode> splitNode = sample->splitNodeFactory(frame);

  unsigned int level = 0;
  while (!indexSet.empty()) {
    bottom->scheduleSplits(samplePred.get(), splitNode.get(), this);
    indexSet = move(splitDispatch(splitNode.get(), preTree.get(), level++));
  }

  relFlush();
  preTree->subtreeFrontier(st2PT);

  return preTree;
}


vector<IndexSet> IndexLevel::splitDispatch(SplitNode* splitNode,
                                           PreTree* preTree,
                                           unsigned int level) {
  levelTerminal = (level + 1 == totLevels);
  unsigned int idxMax, splitNext, leafThis, idxExtent;
  idxExtent = idxLive; // Previous level's index space.
  leafThis = splitNext = idxLive = idxMax = 0;

  vector<SplitNux> argMax(splitNode->split(samplePred.get()));
  for (auto & iSet : indexSet) {
    iSet.applySplit(argMax);
    iSet.splitCensus(this, leafThis, splitNext, idxLive, idxMax);
  }

  // Restaging is implemented as a patient stable partition.
  //
  unsigned int leafNext = 2 * (indexSet.size() - leafThis) - splitNext;
  succBase = vector<unsigned int>(splitNext + leafNext + leafThis);
  fill(succBase.begin(), succBase.end(), idxExtent); // Inattainable base.

  consume(splitNode, preTree, argMax, splitNext, leafNext, idxMax);
  splitNode->levelClear();

  return produce(preTree, splitNext);
}


void IndexSet::splitCensus(IndexLevel *indexLevel,
                           unsigned int &leafThis,
                           unsigned int &splitNext,
                           unsigned int &idxLive,
                           unsigned int &idxMax) {
  if (doesSplit) {
    splitNext += splitAccum(indexLevel, lhExtent, idxLive, idxMax);
    splitNext += splitAccum(indexLevel, extent - lhExtent, idxLive, idxMax);
  }
  else {
    leafThis++;
  }
}


void IndexSet::applySplit(const vector<SplitNux> &argMaxVec) {
  doesSplit = argMaxVec[splitIdx].isInformative(minInfo, lhSCount, lhExtent);
}


unsigned IndexSet::splitAccum(IndexLevel *indexLevel,
                              unsigned int succExtent,
                              unsigned int &idxLive,
                              unsigned int &idxMax) {
  if (indexLevel->isSplitable(succExtent)) {
    idxLive += succExtent;
    idxMax = succExtent > idxMax ? succExtent : idxMax;
    return 1;
  }
  else {
    return 0;
  }
}

  
void IndexLevel::consume(const SplitNode* splitNode,
                         PreTree *preTree,
                         const vector<SplitNux> &argMax,
                         IndexType splitNext,
                         unsigned int leafNext,
                         IndexType idxMax) {
  preTree->levelStorage(splitNext, leafNext); // Overlap:  two levels co-exist.
  replayExpl->Clear();
  succLive = 0;
  succExtinct = splitNext; // Pseudo-indexing for extinct sets.
  liveBase = 0;
  extinctBase = idxLive;
  for (auto & iSet : indexSet) {
    iSet.consume(this, splitNode, preTree, argMax);
  }

  reindex(idxMax, splitNext);
  relBase = move(succBase);
}


void IndexLevel::reindex(IndexType idxMax, IndexType splitNext) {
  if (nodeRel) {
    nodeReindex();
  }
  else {
    nodeRel = IdxPath::localizes(bagCount, idxMax);
    if (nodeRel) {
      transitionReindex(splitNext);
    }
    else {
      subtreeReindex(splitNext);
    }
  }
}



void IndexSet::consume(IndexLevel *indexLevel, const SplitNode* splitNode, PreTree *preTree, const vector<SplitNux> &argMax) {
  if (doesSplit) {
    nonTerminal(indexLevel, splitNode, preTree, argMax[splitIdx]);
  }
  else {
    terminal(indexLevel);
  }
}


void IndexSet::terminal(IndexLevel *indexLevel) {
  succOnly = indexLevel->idxSucc(extent, ptId, offOnly, true);
}


void IndexSet::nonTerminal(IndexLevel *indexLevel, const SplitNode* splitNode, PreTree *preTree, const SplitNux &argMax) {
  leftExpl = preTree->nonterminal(splitNode, argMax, indexLevel, this);

  ptExpl = getPTIdSucc(preTree, leftExpl);
  ptImpl = getPTIdSucc(preTree, !leftExpl);
  succExpl = indexLevel->idxSucc(getExtentSucc(leftExpl), ptExpl, offExpl);
  succImpl = indexLevel->idxSucc(getExtentSucc(!leftExpl), ptImpl, offImpl);

  pathExpl = IdxPath::pathNext(path, leftExpl);
  pathImpl = IdxPath::pathNext(path, !leftExpl);
}


bool IndexSet::branchCut(const SplitNux& argMax,
                         IndexLevel* indexLevel) {
  sumExpl += indexLevel->blockReplay(argMax, ctgExpl);
  
  return argMax.leftIsExplicit();
}


double IndexLevel::blockReplay(const SplitNux& argMax, vector<SumCount>& ctgExpl) {
  return samplePred->blockReplay(argMax, replayExpl.get(), ctgExpl);
}


void IndexSet::blockReplay(const SplitNux& argMax,
                           const IndexRange& range,
                           IndexLevel* indexLevel) {
  sumExpl += indexLevel->blockReplay(argMax, range, ctgExpl);
}


double IndexLevel::blockReplay(const SplitNux& argMax,
                               const IndexRange& range,
                               vector<SumCount>& ctgExpl) const {
  return samplePred->blockReplay(argMax, range, replayExpl.get(), ctgExpl);
}


unsigned int IndexLevel::idxSucc(unsigned int extent,
                                 unsigned int ptId,
                                 unsigned int &offOut,
                                 bool predTerminal) {
  unsigned int idxSucc_;
  if (predTerminal || !isSplitable(extent)) { // Pseudo split caches settings.
    idxSucc_ = succExtinct++;
    offOut = extinctBase;
    extinctBase += extent;
  }
  else {
    idxSucc_ = succLive++;
    offOut = liveBase;
    liveBase += extent;
  }
  succBase[idxSucc_] = offOut;

  return idxSucc_;
}


void IndexLevel::nodeReindex() {
  vector<unsigned int> succST(idxLive);
  rel2PT = vector<unsigned int>(idxLive);

  OMPBound splitIdx;
#pragma omp parallel default(shared) private(splitIdx) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1) 
    for (splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
      indexSet[splitIdx].reindex(replayExpl.get(), this, idxLive, succST);
    }
  }
  rel2ST = move(succST);
}


void IndexSet::reindex(const BV* replayExpl,
                       IndexLevel* index,
                       unsigned int idxLive,
                       vector<unsigned int>& succST) {
  if (!doesSplit) {
    index->relExtinct(relBase, extent, ptId);
  }
  else {
    nontermReindex(replayExpl, index, idxLive, succST);
  }
}


void IndexSet::nontermReindex(const BV *replayExpl,
                              IndexLevel *index,
                              unsigned int idxLive,
                              vector<unsigned int> &succST) {
  unsigned int baseExpl = offExpl;
  unsigned int baseImpl = offImpl;
  for (IndexType relIdx = relBase; relIdx < relBase + extent; relIdx++) {
    bool expl = replayExpl->testBit(relIdx);
    unsigned int targIdx = expl ? offExpl++ : offImpl++;

    if (targIdx < idxLive) {
      succST[targIdx] = index->relLive(relIdx, targIdx, expl ? pathExpl : pathImpl, expl? baseExpl : baseImpl, expl ? ptExpl : ptImpl);
    }
    else {
      index->relExtinct(relIdx, expl ? ptExpl : ptImpl);
    }
  }
}


unsigned int IndexLevel::relLive(unsigned int relIdx,
                                 unsigned int targIdx,
                                 unsigned int path,
                                 unsigned int base,
                                 unsigned int ptIdx) {
  unsigned int stIdx = rel2ST[relIdx];
  rel2PT[targIdx] = ptIdx;
  bottom->setLive(relIdx, targIdx, stIdx, path, base);

  return stIdx;
}


void IndexLevel::relExtinct(unsigned int relIdx, unsigned int ptId) {
  unsigned int stIdx = rel2ST[relIdx];
  st2PT[stIdx] = ptId;
  bottom->setExtinct(relIdx, stIdx);
}


void IndexLevel::subtreeReindex(unsigned int splitNext) {
  unsigned int chunkSize = 1024;
  unsigned int nChunk = (bagCount + chunkSize - 1) / chunkSize;

  OMPBound chunk;
#pragma omp parallel default(shared) private(chunk) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (chunk = 0; chunk < nChunk; chunk++) {
    chunkReindex(bottom->getSubtreePath(), splitNext, chunk * chunkSize, (chunk + 1) * chunkSize);
  }
  }
}


void IndexLevel::chunkReindex(IdxPath *stPath,
                              unsigned int splitNext,
                              unsigned int chunkStart,
                              unsigned int chunkNext) {
  unsigned int chunkEnd = chunkNext > bagCount ? bagCount : chunkNext;
  for (unsigned int stIdx = chunkStart; stIdx < chunkEnd; stIdx++) {
    if (stPath->isLive(stIdx)) {
      unsigned int pathSucc, ptSucc;
      unsigned int splitIdx = st2Split[stIdx];
      IndexType splitSucc = indexSet[splitIdx].offspring(replayExpl->testBit(stIdx), pathSucc, ptSucc);
      st2Split[stIdx] = splitSucc;
      stPath->setSuccessor(stIdx, pathSucc, splitSucc < splitNext);
      st2PT[stIdx] = ptSucc;
    }
  }
}


void IndexLevel::transitionReindex(unsigned int splitNext) {
  IdxPath *stPath = bottom->getSubtreePath();
  for (unsigned int stIdx = 0; stIdx < bagCount; stIdx++) {
    if (stPath->isLive(stIdx)) {
      unsigned int pathSucc, idxSucc, ptSucc;
      unsigned int splitIdx = st2Split[stIdx];
      IndexType splitSucc = indexSet[splitIdx].offspring(replayExpl->testBit(stIdx), pathSucc, idxSucc, ptSucc);
      if (splitSucc < splitNext) {
        stPath->setLive(stIdx, pathSucc, idxSucc);
        rel2ST[idxSucc] = stIdx;
      }
      else {
        stPath->setExtinct(stIdx);
      }
      st2PT[stIdx] = ptSucc;
    }
  }
}


vector<IndexSet> IndexLevel::produce(const PreTree *preTree, IndexType splitNext) {
  bottom->overlap(splitNext, bagCount, idxLive, nodeRel);
  vector<IndexSet> indexNext(splitNext);
  for (auto & iSet : indexSet) {
    iSet.succHand(indexNext, bottom.get(), this, preTree, true);
    iSet.succHand(indexNext, bottom.get(), this, preTree, false);
  }
  return indexNext;
}


void IndexSet::succHand(vector<IndexSet>& indexNext, Bottom* bottom, IndexLevel* indexLevel, const PreTree* preTree, bool isLeft) const {
  if (doesSplit) {
    IndexType succIdx = getIdxSucc(isLeft);
    if (succIdx < indexNext.size()) {
      indexNext[succIdx].succInit(indexLevel, bottom, preTree, this, isLeft);
    }
  }
}


void IndexSet::succInit(IndexLevel *indexLevel,
                        Bottom *bottom,
                        const PreTree* preTree,
                        const IndexSet* par,
                        bool isLeft) {
  splitIdx = par->getIdxSucc(isLeft);
  sCount = par->getSCountSucc(isLeft);
  lhStart = par->getLHStartSucc(isLeft);
  extent = par->getExtentSucc(isLeft);
  minInfo = par->getMinInfo();
  ptId = par->getPTIdSucc(preTree, isLeft);
  sum = par->getSumSucc(isLeft);
  path = par->getPathSucc(isLeft);
  relBase = indexLevel->getRelBase(splitIdx);
  bottom->reachingPath(splitIdx, par->getSplitIdx(), lhStart, extent, relBase, path);

  if (par->isExplHand(isLeft)) {
    ctgSum = par->getCtgExpl();
  }
  else {
    ctgSum = par->getCtgSum();
    decr(ctgSum, par->getCtgExpl());
  }
  ctgExpl = vector<SumCount>(ctgSum.size());

  // Inattainable value.  Reset only when non-terminal:
  initInattainable(indexLevel->getBagCount());
}


IndexType IndexSet::getPTIdSucc(const PreTree* preTree, bool isLeft) const {
  return preTree->getSuccId(ptId, isLeft);
}


vector<double> IndexLevel::sumsAndSquares(vector<vector<double> >&ctgSum) {
  vector<double> sumSquares(indexSet.size());

  OMPBound splitIdx;
#pragma omp parallel default(shared) private(splitIdx) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
      ctgSum[splitIdx] = indexSet[splitIdx].sumsAndSquares(sumSquares[splitIdx]);
    }
  }
  return sumSquares;
}


vector<double> IndexSet::sumsAndSquares(double& sumSquares) {
  vector<double> sumOut(ctgSum.size());
  sumSquares =  0.0;
  for (unsigned int ctg = 0; ctg < ctgSum.size(); ctg++) {
    unsplitable |= !ctgSum[ctg].splitable(sCount, sumOut[ctg]);
    sumSquares += sumOut[ctg] * sumOut[ctg];
  }

  return sumOut;
}
