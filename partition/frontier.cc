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


Frontier::~Frontier() {
}


IndexSet::IndexSet() :
  splitIdx(0),
  ptId(0),
  sCount(0),
  sum(0.0),
  minInfo(0.0),
  path(0),
  doesSplit(false),
  unsplitable(false),
  lhExtent(0),
  lhSCount(0),
  sumL(0.0) {
}


unique_ptr<PreTree> Frontier::oneTree(const SummaryFrame* frame,
                                      const Sample *sample) {
  unique_ptr<Frontier> frontier(make_unique<Frontier>(frame, sample));
  unique_ptr<SplitFrontier> splitFrontier(sample->frontierFactory(frame, frontier.get()));
  return frontier->levels(sample, splitFrontier.get());
}



Frontier::Frontier(const SummaryFrame* frame,
                   const Sample* sample) :
  indexSet(vector<IndexSet>(1)),
  bagCount(sample->getBagCount()),
  bottom(make_unique<Bottom>(frame, bagCount)),
  nodeRel(false),
  idxLive(bagCount),
  relBase(vector<IndexT>(1)),
  rel2ST(vector<IndexT>(bagCount)),
  st2Split(vector<IndexT>(bagCount)),
  st2PT(vector<IndexT>(bagCount)),
  replayExpl(make_unique<BV>(bagCount)),
  replayLeft(make_unique<BV>(bagCount)),
  pretree(make_unique<PreTree>(frame, this)) {
  indexSet[0].initRoot(sample);
  relBase[0] = 0;
  iota(rel2ST.begin(), rel2ST.end(), 0);
  fill(st2Split.begin(), st2Split.end(), 0);
  fill(st2PT.begin(), st2PT.end(), 0);
}


void IndexSet::initRoot(const Sample* sample) {
  splitIdx = 0;
  sCount = sample->getNSamp();
  bufRange.set(0, sample->getBagCount());
  minInfo = 0.0;
  ptId = 0;
  sum = sample->getBagSum();
  path = 0;
  relBase = 0;
  ctgSum = sample->getCtgRoot();
  ctgLeft = vector<SumCount>(ctgSum.size());

  initInattainable(sample->getBagCount());
}


unique_ptr<PreTree> Frontier::levels(const Sample* sample,
                                     SplitFrontier* splitFrontier) {
  bottom->rootDef(splitFrontier->stage(sample), bagCount);
  
  unsigned int level = 0;
  while (!indexSet.empty()) {
    bottom->scheduleSplits(splitFrontier, this);
    indexSet = splitDispatch(splitFrontier, level++);
  }

  relFlush();
  pretree->subtreeFrontier(st2PT);
  return move(pretree);
}


vector<IndexSet> Frontier::splitDispatch(SplitFrontier* splitFrontier,
                                         unsigned int level) {
  levelTerminal = (level + 1 == totLevels);

  SplitSurvey survey = splitFrontier->consume(pretree.get(), indexSet, replayExpl.get(), replayLeft.get());

  nextLevel(survey);
  for (auto & iSet : indexSet) {
    iSet.dispatch(this);
  }

  reindex(survey);
  relBase = move(succBase);

  return produce(survey.splitNext);
}


void IndexSet::dispatch(Frontier* frontier) {
  if (doesSplit) {
    nonterminal(frontier);
  }
  else {
    terminal(frontier);
  }
}


void Frontier::nextLevel(const SplitSurvey& survey) {
  succBase = vector<IndexT>(survey.succCount(indexSet.size()));
  fill(succBase.begin(), succBase.end(), idxLive); // Previous level's extent

  succLive = 0;
  succExtinct = survey.splitNext; // Pseudo-indexing for extinct sets.
  liveBase = 0;
  extinctBase = survey.idxLive;
  idxLive = survey.idxLive;
}


unsigned int Frontier::splitCensus(const IndexSet& iSet,
                                   SplitSurvey& survey) {
  return splitAccum(iSet.getExtentSucc(true), survey) + splitAccum(iSet.getExtentSucc(false), survey);
}


unsigned int Frontier::splitAccum(IndexT succExtent,
                                  SplitSurvey& survey) {
  if (isSplitable(succExtent)) {
    survey.idxLive += succExtent;
    survey.idxMax = max(survey.idxMax, succExtent);
    return 1;
  }
  else {
    return 0;
  }
}

  
void IndexSet::terminal(Frontier *frontier) {
  succOnly = frontier->idxSucc(bufRange.getExtent(), ptId, offOnly, true);
}


void IndexSet::nonterminal(Frontier* frontier) {
  ptLeft = getPTIdSucc(frontier, true);
  ptRight = getPTIdSucc(frontier, false);
  succLeft = frontier->idxSucc(getExtentSucc(true), ptLeft, offLeft);
  succRight = frontier->idxSucc(getExtentSucc(false), ptRight, offRight);

  pathLeft = IdxPath::pathNext(path, true);
  pathRight = IdxPath::pathNext(path, false);
}


IndexT Frontier::idxSucc(IndexT extent,
                         IndexT ptId,
                         IndexT& offOut,
                         bool predTerminal) {
  IndexT idxSucc_;
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


void Frontier::reindex(const SplitSurvey& survey) {
  if (nodeRel) {
    nodeReindex();
  }
  else {
    nodeRel = IdxPath::localizes(bagCount, survey.idxMax);
    if (nodeRel) {
      transitionReindex(survey.splitNext);
    }
    else {
      stReindex(survey.splitNext);
    }
  }
}


void Frontier::nodeReindex() {
  vector<IndexT> succST(idxLive);
  rel2PT = vector<IndexT>(idxLive);

  OMPBound splitIdx;
#pragma omp parallel default(shared) private(splitIdx) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1) 
    for (splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
      indexSet[splitIdx].reindex(replayExpl.get(), replayLeft.get(), this, idxLive, succST);
    }
  }
  rel2ST = move(succST);
}


void IndexSet::reindex(const BV* replayExpl,
                       const BV* replayLeft,
                       Frontier* index,
                       IndexT idxLive,
                       vector<IndexT>& succST) {
  if (!doesSplit) {
    index->relExtinct(relBase, bufRange.getExtent(), ptId);
  }
  else {
    nontermReindex(replayExpl, replayLeft, index, idxLive, succST);
  }
}


void IndexSet::nontermReindex(const BV* replayExpl,
                              const BV* replayLeft,
                              Frontier* index,
                              IndexT idxLive,
                              vector<IndexT>&succST) {
  IndexT baseLeft = offLeft;
  IndexT baseRight = offRight;
  for (IndexT relIdx = relBase; relIdx < relBase + bufRange.getExtent(); relIdx++) {
    bool isLeft = senseLeft(replayExpl, replayLeft, relIdx);
    IndexT targIdx = getOffSucc(isLeft);
    if (targIdx < idxLive) {
      succST[targIdx] = index->relLive(relIdx, targIdx, getPathSucc(isLeft), isLeft ? baseLeft : baseRight, getPTSucc(isLeft));
    }
    else {
      index->relExtinct(relIdx, getPTSucc(isLeft));
    }
  }
}


IndexT Frontier::relLive(IndexT relIdx,
                            IndexT targIdx,
                            IndexT path,
                            IndexT base,
                            IndexT ptIdx) {
  IndexT stIdx = rel2ST[relIdx];
  rel2PT[targIdx] = ptIdx;
  bottom->setLive(relIdx, targIdx, stIdx, path, base);

  return stIdx;
}


void Frontier::relExtinct(IndexT relIdx, IndexT ptId) {
  IndexT stIdx = rel2ST[relIdx];
  st2PT[stIdx] = ptId;
  bottom->setExtinct(relIdx, stIdx);
}


void Frontier::stReindex(unsigned int splitNext) {
  unsigned int chunkSize = 1024;
  IndexT nChunk = (bagCount + chunkSize - 1) / chunkSize;

  OMPBound chunk;
#pragma omp parallel default(shared) private(chunk) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (chunk = 0; chunk < nChunk; chunk++) {
    stReindex(bottom->getSubtreePath(), splitNext, chunk * chunkSize, (chunk + 1) * chunkSize);
  }
  }
}


void Frontier::stReindex(IdxPath* stPath,
                         IndexT splitNext,
                         IndexT chunkStart,
                         IndexT chunkNext) {
  IndexT chunkEnd = min(chunkNext, bagCount);
  for (IndexT stIdx = chunkStart; stIdx < chunkEnd; stIdx++) {
    if (stPath->isLive(stIdx)) {
      IndexT pathSucc, ptSucc;
      IndexT splitIdx = st2Split[stIdx];
      IndexT splitSucc = indexSet[splitIdx].offspring(replayExpl.get(), replayLeft.get(), stIdx, pathSucc, ptSucc);
      st2Split[stIdx] = splitSucc;
      stPath->setSuccessor(stIdx, pathSucc, splitSucc < splitNext);
      st2PT[stIdx] = ptSucc;
    }
  }
}


void Frontier::transitionReindex(IndexT splitNext) {
  IdxPath *stPath = bottom->getSubtreePath();
  for (IndexT stIdx = 0; stIdx < bagCount; stIdx++) {
    if (stPath->isLive(stIdx)) {
      IndexT pathSucc, idxSucc, ptSucc;
      IndexT splitIdx = st2Split[stIdx];
      IndexT splitSucc = indexSet[splitIdx].offspring(replayExpl.get(), replayLeft.get(), stIdx, pathSucc, idxSucc, ptSucc);
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


vector<IndexSet> Frontier::produce(IndexT splitNext) {
  bottom->overlap(splitNext, bagCount, idxLive, nodeRel);
  vector<IndexSet> indexNext(splitNext);
  for (auto & iSet : indexSet) {
    iSet.succHands(this, indexNext);
  }
  return indexNext;
}


void IndexSet::succHands(Frontier* frontier, vector<IndexSet>& indexNext) const {
  if (doesSplit) {
    succHand(frontier, indexNext, true);
    succHand(frontier, indexNext, false);
  }
}


void IndexSet::succHand(Frontier* frontier, vector<IndexSet>& indexNext, bool isLeft) const {
  IndexT succIdx = getIdxSucc(isLeft);
  if (succIdx < indexNext.size()) { // Otherwise terminal in next level.
    indexNext[succIdx].succInit(frontier, this, isLeft);
  }
}


void IndexSet::succInit(Frontier *frontier,
                        const IndexSet* par,
                        bool isLeft) {
  splitIdx = par->getIdxSucc(isLeft);
  sCount = par->getSCountSucc(isLeft);
  bufRange.set(par->getStartSucc(isLeft), par->getExtentSucc(isLeft));
  minInfo = par->getMinInfo();
  ptId = par->getPTIdSucc(frontier, isLeft);
  sum = par->getSumSucc(isLeft);
  path = par->getPathSucc(isLeft);
  relBase = frontier->getRelBase(splitIdx);
  frontier->reachingPath(splitIdx, par->getSplitIdx(), bufRange, relBase, path);

  ctgSum = isLeft ? move(par->ctgLeft) : SumCount::minus(par->getCtgSum(), par->getCtgLeft());
  ctgLeft = vector<SumCount>(ctgSum.size());

  // Inattainable value.  Reset only when non-terminal:
  initInattainable(frontier->getBagCount());
}


IndexT IndexSet::getPTIdSucc(const Frontier* frontier, bool isLeft) const {
  return frontier->getPTIdSucc(ptId, isLeft);
}


IndexT Frontier::getPTIdSucc(IndexT ptId, bool isLeft) const {
  return pretree->getSuccId(ptId, isLeft);
}


void Frontier::reachingPath(IndexT splitIdx,
                            IndexT parIdx,
                            const IndexRange& bufRange,
                            IndexT relBase,
                            unsigned int path) const {
  bottom->reachingPath(splitIdx, parIdx, bufRange, relBase, path);
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
