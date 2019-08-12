// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file frontier.cc

   @brief Maintains the sample-index representation of the frontier, typically by level.

   @author Mark Seligman
 */

#include "frontier.h"
#include "bv.h"
#include "pretree.h"
#include "sample.h"
#include "obspart.h"
#include "splitfrontier.h"
#include "bottom.h"
#include "path.h"
#include "ompthread.h"

#include <numeric>


unsigned int Frontier::totLevels = 0;
unsigned int Frontier::minNode = 0;

void Frontier::immutables(unsigned int minNode, unsigned int totLevels) {
  Frontier::minNode = minNode;
  Frontier::totLevels = totLevels;
}


void Frontier::deImmutables() {
  totLevels = 0;
  minNode = 0;
}


void IndexSet::decr(vector<SumCount> &_ctgSum, const vector<SumCount> &_ctgSub) {
  unsigned i = 0;
  for (auto & sc : _ctgSum) {
    sc.decr(_ctgSub[i++]);
  }
}


Frontier::~Frontier() {
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
  doesSplit(false),
  unsplitable(false),
  sumExpl(0.0){
}


unique_ptr<PreTree> Frontier::oneTree(const SummaryFrame* frame,
                                      const Sample *sample) {
  unique_ptr<Frontier> frontier(make_unique<Frontier>(frame, sample));
  unique_ptr<PreTree> pretree(make_unique<PreTree>(frame, frontier.get()));
  frontier->levels(pretree.get());
  return pretree;
}



Frontier::Frontier(const SummaryFrame* frame,
                   const Sample* sample) :
  obsPart(sample->predictors()),
  indexSet(vector<IndexSet>(1)),
  bagCount(sample->getBagCount()),
  splitFrontier(sample->frontierFactory(frame)),
  bottom(make_unique<Bottom>(frame, bagCount)),
  nodeRel(false),
  idxLive(bagCount),
  relBase(vector<IndexType>(1)),
  rel2ST(vector<IndexType>(bagCount)),
  st2Split(vector<IndexType>(bagCount)),
  st2PT(vector<IndexType>(bagCount)),
  replayExpl(make_unique<BV>(bagCount)) {
  indexSet[0].initRoot(sample);
  relBase[0] = 0;
  iota(rel2ST.begin(), rel2ST.end(), 0);
  fill(st2Split.begin(), st2Split.end(), 0);
  fill(st2PT.begin(), st2PT.end(), 0);
  bottom->rootDef(sample->stage(obsPart.get()), bagCount);
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


void Frontier::levels(PreTree* preTree) {
  unsigned int level = 0;
  while (!indexSet.empty()) {
    bottom->scheduleSplits(obsPart.get(), splitFrontier.get(), this);
    indexSet = move(splitDispatch(preTree, level++));
  }

  relFlush();
  preTree->subtreeFrontier(st2PT);
}


vector<IndexSet> Frontier::splitDispatch(PreTree* preTree,
                                         unsigned int level) {
  levelTerminal = (level + 1 == totLevels);
  splitFrontier->split(obsPart.get());

  IndexType idxMax;
  IndexType splitNext = nextLevel(idxMax);
  consume(preTree, splitNext);

  reindex(idxMax, splitNext);
  relBase = move(succBase);

  return produce(preTree, splitNext);
}


IndexType Frontier::nextLevel(IndexType& idxMax) {
  IndexType idxExtent = idxLive; // Previous level's index extent.
  IndexType leafThis, splitNext;
  leafThis = idxLive = splitNext = idxMax = 0;
  for (auto & iSet : indexSet) {
    if (!splitFrontier->isInformative(&iSet)) {
      leafThis++;
    }
    else {
      splitNext += splitCensus(iSet, idxMax);
    }
  }
  IndexType leafNext = 2 * (indexSet.size() - leafThis) - splitNext;

  succBase = vector<IndexType>(splitNext + leafNext + leafThis);
  fill(succBase.begin(), succBase.end(), idxExtent); // Inattainable base.
  return splitNext;
}


unsigned int Frontier::splitCensus(const IndexSet& iSet,
                                   IndexType& idxMax) {
  unsigned int nSucc = 0;
  IndexType succExtent(splitFrontier->getLHExtent(iSet));
  nSucc += splitAccum(succExtent, idxMax);
  nSucc += splitAccum(iSet.getExtent() - succExtent, idxMax);

  return nSucc;
}


unsigned int Frontier::splitAccum(IndexType succExtent,
                                  IndexType& idxMax) {
  if (isSplitable(succExtent)) {
    idxLive += succExtent;
    idxMax = max(idxMax, succExtent);
    return 1;
  }
  else {
    return 0;
  }
}

  
void Frontier::consume(PreTree* preTree,
                       IndexType splitNext) {
  replayExpl->Clear();
  succLive = 0;
  succExtinct = splitNext; // Pseudo-indexing for extinct sets.
  liveBase = 0;
  extinctBase = idxLive;
  for (auto & iSet : indexSet) {
    iSet.consume(this, splitFrontier.get(), preTree);
  }
  splitFrontier->clear();
}


void IndexSet::consume(Frontier *frontier, const SplitFrontier* splitFrontier, PreTree *preTree) {
  if (splitFrontier->isInformative(this)) {
    nonterminal(frontier, splitFrontier, preTree);
  }
  else {
    terminal(frontier);
  }
}


void IndexSet::terminal(Frontier *frontier) {
  succOnly = frontier->idxSucc(extent, ptId, offOnly, true);
}


void IndexSet::nonterminal(Frontier* frontier, const SplitFrontier* splitFrontier, PreTree* preTree) {
  doesSplit = true;
  leftExpl = splitFrontier->branch(frontier, preTree, this);

  ptExpl = getPTIdSucc(preTree, leftExpl);
  ptImpl = getPTIdSucc(preTree, !leftExpl);
  succExpl = frontier->idxSucc(getExtentSucc(leftExpl), ptExpl, offExpl);
  succImpl = frontier->idxSucc(getExtentSucc(!leftExpl), ptImpl, offImpl);

  pathExpl = IdxPath::pathNext(path, leftExpl);
  pathImpl = IdxPath::pathNext(path, !leftExpl);
}


IndexType Frontier::idxSucc(IndexType extent,
                            IndexType ptId,
                            IndexType& offOut,
                            bool predTerminal) {
  IndexType idxSucc_;
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


void Frontier::reindex(IndexType idxMax, IndexType splitNext) {
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


void IndexSet::blockReplay(Frontier* frontier,
                           const IndexRange& range) {
  sumExpl += frontier->blockReplay(this, range, ctgExpl);
}


double Frontier::blockReplay(const IndexSet* iSet,
                             const IndexRange& range,
                             vector<SumCount>& ctgExpl) const {
  return obsPart->blockReplay(splitFrontier.get(), iSet, range, replayExpl.get(), ctgExpl);
}


void Frontier::nodeReindex() {
  vector<IndexType> succST(idxLive);
  rel2PT = vector<IndexType>(idxLive);

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
                       Frontier* index,
                       IndexType idxLive,
                       vector<IndexType>& succST) {
  if (!doesSplit) {
    index->relExtinct(relBase, extent, ptId);
  }
  else {
    nontermReindex(replayExpl, index, idxLive, succST);
  }
}


void IndexSet::nontermReindex(const BV* replayExpl,
                              Frontier* index,
                              IndexType idxLive,
                              vector<IndexType>&succST) {
  IndexType baseExpl = offExpl;
  IndexType baseImpl = offImpl;
  for (IndexType relIdx = relBase; relIdx < relBase + extent; relIdx++) {
    bool expl = replayExpl->testBit(relIdx);
    IndexType targIdx = expl ? offExpl++ : offImpl++;

    if (targIdx < idxLive) {
      succST[targIdx] = index->relLive(relIdx, targIdx, expl ? pathExpl : pathImpl, expl? baseExpl : baseImpl, expl ? ptExpl : ptImpl);
    }
    else {
      index->relExtinct(relIdx, expl ? ptExpl : ptImpl);
    }
  }
}


IndexType Frontier::relLive(IndexType relIdx,
                            IndexType targIdx,
                            IndexType path,
                            IndexType base,
                            IndexType ptIdx) {
  IndexType stIdx = rel2ST[relIdx];
  rel2PT[targIdx] = ptIdx;
  bottom->setLive(relIdx, targIdx, stIdx, path, base);

  return stIdx;
}


void Frontier::relExtinct(IndexType relIdx, IndexType ptId) {
  IndexType stIdx = rel2ST[relIdx];
  st2PT[stIdx] = ptId;
  bottom->setExtinct(relIdx, stIdx);
}


void Frontier::subtreeReindex(unsigned int splitNext) {
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


void Frontier::chunkReindex(IdxPath *stPath,
                            IndexType splitNext,
                            IndexType chunkStart,
                            IndexType chunkNext) {
  IndexType chunkEnd = min(chunkNext, bagCount);
  for (IndexType stIdx = chunkStart; stIdx < chunkEnd; stIdx++) {
    if (stPath->isLive(stIdx)) {
      IndexType pathSucc, ptSucc;
      IndexType splitIdx = st2Split[stIdx];
      IndexType splitSucc = indexSet[splitIdx].offspring(replayExpl->testBit(stIdx), pathSucc, ptSucc);
      st2Split[stIdx] = splitSucc;
      stPath->setSuccessor(stIdx, pathSucc, splitSucc < splitNext);
      st2PT[stIdx] = ptSucc;
    }
  }
}


void Frontier::transitionReindex(IndexType splitNext) {
  IdxPath *stPath = bottom->getSubtreePath();
  for (IndexType stIdx = 0; stIdx < bagCount; stIdx++) {
    if (stPath->isLive(stIdx)) {
      IndexType pathSucc, idxSucc, ptSucc;
      IndexType splitIdx = st2Split[stIdx];
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


vector<IndexSet> Frontier::produce(const PreTree *preTree, IndexType splitNext) {
  bottom->overlap(splitNext, bagCount, idxLive, nodeRel);
  vector<IndexSet> indexNext(splitNext);
  for (auto & iSet : indexSet) {
    iSet.succHand(indexNext, bottom.get(), this, preTree, true);
    iSet.succHand(indexNext, bottom.get(), this, preTree, false);
  }
  return indexNext;
}


void IndexSet::succHand(vector<IndexSet>& indexNext, Bottom* bottom, Frontier* frontier, const PreTree* preTree, bool isLeft) const {
  if (doesSplit) {
    IndexType succIdx = getIdxSucc(isLeft);
    if (succIdx < indexNext.size()) {
      indexNext[succIdx].succInit(frontier, bottom, preTree, this, isLeft);
    }
  }
}


void IndexSet::succInit(Frontier *frontier,
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
  relBase = frontier->getRelBase(splitIdx);
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
  initInattainable(frontier->getBagCount());
}


IndexType IndexSet::getPTIdSucc(const PreTree* preTree, bool isLeft) const {
  return preTree->getSuccId(ptId, isLeft);
}


vector<double> Frontier::sumsAndSquares(vector<vector<double> >&ctgSum) {
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
