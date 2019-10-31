// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file indexset.cc

   @brief Maintains frontier tree nodes as blocks within ObsPart.

   @author Mark Seligman
 */

#include "indexset.h"
#include "sample.h"
#include "frontier.h"
#include "path.h"

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
  sumL(0.0),
  leftImpl(false) {
}

void IndexSet::initRoot(const Sample* sample) {
  splitIdx = 0;
  sCount = sample->getNSamp();
  bufRange = IndexRange(0, sample->getBagCount());
  minInfo = 0.0;
  ptId = 0;
  sum = sample->getBagSum();
  path = 0;
  relBase = 0;
  ctgSum = sample->getCtgRoot();
  ctgLeft = vector<SumCount>(ctgSum.size());
  
  initInattainable(sample->getBagCount());
}


void IndexSet::dispatch(Frontier* frontier) {
  if (doesSplit) {
    nonterminal(frontier);
  }
  else {
    terminal(frontier);
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


void IndexSet::reindex(const Replay* replay,
                       Frontier* index,
                       IndexT idxLive,
                       vector<IndexT>& succST) {
  if (!doesSplit) {
    index->relExtinct(relBase, bufRange.getExtent(), ptId);
  }
  else {
    nontermReindex(replay, index, idxLive, succST);
  }
}


void IndexSet::nontermReindex(const Replay* replay,
                              Frontier* index,
                              IndexT idxLive,
                              vector<IndexT>&succST) {
  IndexT baseLeft = offLeft;
  IndexT baseRight = offRight;
  for (IndexT relIdx = relBase; relIdx < relBase + bufRange.getExtent(); relIdx++) {
    bool isLeft = replay->senseLeft(relIdx, leftImpl);
    IndexT targIdx = getOffSucc(isLeft);
    if (targIdx < idxLive) {
      succST[targIdx] = index->relLive(relIdx, targIdx, getPathSucc(isLeft), isLeft ? baseLeft : baseRight, getPTSucc(isLeft));
    }
    else {
      index->relExtinct(relIdx, getPTSucc(isLeft));
    }
  }
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
  bufRange = IndexRange(par->getStartSucc(isLeft), par->getExtentSucc(isLeft));
  minInfo = par->getMinInfo();
  ptId = par->getPTIdSucc(frontier, isLeft);
  sum = par->getSumSucc(isLeft);
  path = par->getPathSucc(isLeft);
  relBase = frontier->getRelBase(splitIdx);
  frontier->reachingPath(splitIdx, par->getSplitIdx(), bufRange, relBase, path);

  ctgSum = isLeft ? par->ctgLeft : SumCount::minus(par->ctgSum, par->ctgLeft);
  ctgLeft = vector<SumCount>(ctgSum.size());

  // Inattainable value.  Reset only when non-terminal:
  initInattainable(frontier->getBagCount());
}


IndexT IndexSet::getPTIdSucc(const Frontier* frontier, bool isLeft) const {
  return frontier->getPTIdSucc(ptId, isLeft);
}


vector<double> IndexSet::sumsAndSquares(double& sumSquares) {
  vector<double> sumOut(ctgSum.size());
  sumSquares =  0.0;
  for (PredictorT ctg = 0; ctg < ctgSum.size(); ctg++) {
    unsplitable |= !ctgSum[ctg].splitable(sCount, sumOut[ctg]);
    sumSquares += sumOut[ctg] * sumOut[ctg];
  }

  return sumOut;
}

