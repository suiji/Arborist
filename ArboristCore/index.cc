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
#include "splitcand.h"
#include "splitnode.h"
#include "bottom.h"
#include "path.h"
#include "runset.h"
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


shared_ptr<PreTree> IndexLevel::oneTree(const FrameTrain *frameTrain,
                                        const RowRank* rowRank,
                                        const Sample *sample) {
  auto index = make_unique<IndexLevel>(frameTrain, rowRank, sample);
  return index->levels(frameTrain, sample);
}



IndexLevel::IndexLevel(const FrameTrain* frameTrain,
                       const RowRank* rowRank,
                       const Sample* sample) :
  samplePred(sample->predictors()),
  indexSet(vector<IndexSet>(1)),
  bagCount(sample->getBagCount()),
  bottom(make_unique<Bottom>(frameTrain, rowRank, bagCount)),
  nodeRel(false),
  idxLive(bagCount),
  relBase(vector<unsigned int>(1)),
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


shared_ptr<PreTree> IndexLevel::levels(const FrameTrain *frameTrain,
                                       const Sample* sample) {
  auto stageCount = sample->stage(samplePred.get());
  bottom->rootDef(stageCount);
  shared_ptr<PreTree> preTree = make_shared<PreTree>(frameTrain, bagCount);
  auto splitNode = sample->splitNodeFactory(frameTrain);

  for (unsigned int level = 0; !indexSet.empty(); level++) {
    bottom->scheduleSplits(samplePred.get(), splitNode.get(), this);
    auto argMax = splitNode->split(samplePred.get());
    splitDispatch(splitNode.get(), argMax, preTree.get(), level + 1 == totLevels);
    splitNode->levelClear();
  }

  relFlush();
  preTree->subtreeFrontier(st2PT);

  return preTree;
}


void IndexLevel::splitDispatch(const SplitNode* splitNode,
                               const vector<SplitCand> &argMax,
                               PreTree* preTree,
                               bool levelTerminal) {
  this->levelTerminal = levelTerminal;
  unsigned int leafNext, idxMax, splitNext, leafThis, idxExtent;
  idxExtent = idxLive; // Previous level's index space.
  leafThis = splitNext = idxLive = idxMax = 0;
  for (auto & iSet : indexSet) {
    iSet.applySplit(argMax);
    iSet.splitCensus(this, leafThis, splitNext, idxLive, idxMax);
  }

  // Restaging is implemented as a patient stable partition.
  //
  leafNext = 2 * (indexSet.size() - leafThis) - splitNext;
  succBase = vector<unsigned int>(splitNext + leafNext + leafThis);
  fill(succBase.begin(), succBase.end(), idxExtent); // Inattainable base.

  consume(splitNode, preTree, argMax, splitNext, leafNext, idxMax);
  produce(preTree, splitNext);
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


void IndexSet::applySplit(const vector<SplitCand> &argMaxVec) {
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
                         const vector<SplitCand> &argMax,
                         unsigned int splitNext,
                         unsigned int leafNext,
                         unsigned int idxMax) {
  preTree->levelStorage(splitNext, leafNext); // Overlap:  two levels co-exist.
  replayExpl->Clear();
  succLive = 0;
  succExtinct = splitNext; // Pseudo-indexing for extinct sets.
  liveBase = 0;
  extinctBase = idxLive;
  for (auto & iSet : indexSet) {
    iSet.consume(this, splitNode->getRuns(), preTree, argMax);
  }

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

  relBase = move(succBase);
}


void IndexSet::consume(IndexLevel *indexLevel, const Run* run, PreTree *preTree, const vector<SplitCand> &argMax) {
  if (doesSplit) {
    nonTerminal(indexLevel, run, preTree, argMax[splitIdx]);
  }
  else {
    terminal(indexLevel);
  }
}


void IndexSet::terminal(IndexLevel *indexLevel) {
  succOnly = indexLevel->idxSucc(extent, ptId, offOnly, true);
}


void IndexSet::nonTerminal(IndexLevel *indexLevel, const Run* run, PreTree *preTree, const SplitCand &argMax) {
  leftExpl =   run->isRun(argMax) ? run->branchFac(argMax, this, preTree, indexLevel) : branchNum(argMax, preTree, indexLevel);

  ptExpl = getPTIdSucc(preTree, leftExpl);
  ptImpl = getPTIdSucc(preTree, !leftExpl);
  succExpl = indexLevel->idxSucc(getExtentSucc(leftExpl), ptExpl, offExpl);
  succImpl = indexLevel->idxSucc(getExtentSucc(!leftExpl), ptImpl, offImpl);

  pathExpl = IdxPath::pathNext(path, leftExpl);
  pathImpl = IdxPath::pathNext(path, !leftExpl);
}


bool IndexSet::branchNum(const SplitCand& argMax,
                         PreTree *preTree,
                         IndexLevel* indexLevel) {
  preTree->branchNum(argMax, ptId);
  sumExpl += indexLevel->blockReplay(argMax, ctgExpl);
  
  return argMax.leftIsExplicit();
}


double IndexLevel::blockReplay(const SplitCand& argMax, vector<SumCount>& ctgExpl) {
  return samplePred->blockReplay(argMax, replayExpl.get(), ctgExpl);
}


void IndexSet::blockReplay(const SplitCand& argMax,
                           unsigned int blockStart,
                           unsigned int blockExtent,
                           IndexLevel* indexLevel) {
  sumExpl += indexLevel->blockReplay(argMax, blockStart, blockExtent, ctgExpl);
}


double IndexLevel::blockReplay(const SplitCand& argMax,
                             unsigned int blockStart,
                             unsigned int blockExtent,
                             vector<SumCount>& ctgExpl) const {
  return samplePred->blockReplay(argMax, blockStart, blockExtent, replayExpl.get(), ctgExpl);
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
  for (unsigned int relIdx = relBase; relIdx < relBase + extent; relIdx++) {
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
    chunkReindex(bottom->subtreePath(), splitNext, chunk * chunkSize, (chunk + 1) * chunkSize);
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
      unsigned int splitSucc = indexSet[splitIdx].offspring(replayExpl->testBit(stIdx), pathSucc, ptSucc);
      st2Split[stIdx] = splitSucc;
      stPath->setSuccessor(stIdx, pathSucc, splitSucc < splitNext);
      st2PT[stIdx] = ptSucc;
    }
  }
}


void IndexLevel::transitionReindex(unsigned int splitNext) {
  IdxPath *stPath = bottom->subtreePath();
  for (unsigned int stIdx = 0; stIdx < bagCount; stIdx++) {
    if (stPath->isLive(stIdx)) {
      unsigned int pathSucc, idxSucc, ptSucc;
      unsigned int splitIdx = st2Split[stIdx];
      unsigned int splitSucc = indexSet[splitIdx].offspring(replayExpl->testBit(stIdx), pathSucc, idxSucc, ptSucc);
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


void IndexLevel::produce(const PreTree *preTree, unsigned int splitNext) {
  bottom->overlap(splitNext, idxLive, nodeRel);
  vector<IndexSet> indexNext(splitNext);
  for (auto & iSet : indexSet) {
    iSet.succHand(indexNext, bottom.get(), this, preTree, true);
    iSet.succHand(indexNext, bottom.get(), this, preTree, false);
  }
  indexSet = move(indexNext);
}


void IndexSet::succHand(vector<IndexSet>& indexNext, Bottom* bottom, IndexLevel* indexLevel, const PreTree* preTree, bool isLeft) const {
  unsigned int succIdx = getIdxSucc(isLeft);
  if (doesSplit && succIdx < indexNext.size()){
    indexNext[succIdx].succInit(indexLevel, bottom, preTree, this, isLeft);
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


unsigned int IndexSet::getPTIdSucc(const PreTree* preTree, bool isLeft) const {
  return isLeft ? preTree->getLHId(ptId) : preTree->getRHId(ptId); 
}


void IndexLevel::sumsAndSquares(unsigned int ctgWidth,
                                vector<double> &sumSquares,
                                vector<double> &ctgSum) {
  OMPBound splitIdx;
#pragma omp parallel default(shared) private(splitIdx) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
      indexSet[splitIdx].sumsAndSquares(sumSquares[splitIdx], &ctgSum[splitIdx * ctgWidth]);
    }
  }
}


void IndexSet::sumsAndSquares(double  &sumSquares, double *sumOut) {
  for (unsigned int ctg = 0; ctg < ctgSum.size(); ctg++) {
    unsigned int scSCount;
    ctgSum[ctg].ref(sumOut[ctg], scSCount);
    sumSquares += sumOut[ctg] * sumOut[ctg];
    if (scSCount == sCount)
      unsplitable = true;
  }
}


unsigned int IndexLevel::setCand(SplitCand* cand) const {
  return indexSet[cand->getSplitIdx()].setCand(cand);
}


unsigned int IndexSet::setCand(SplitCand* cand) const {
  cand->setIdxStart(lhStart);
  cand->setSCount(sCount);
  cand->setSum(sum);

  return extent;
}
