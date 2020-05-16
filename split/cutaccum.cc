// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file cutaccum.cc

   @brief Base class for cut-based splitting workspace.

   @author Mark Seligman
 */

#include "splitnux.h"
#include "cutaccum.h"
#include "splitfrontier.h"
#include "obspart.h"
#include "residual.h"


CutAccum::CutAccum(const SplitNux* cand,
		   const SplitFrontier* splitFrontier_) :
  Accum(splitFrontier_, cand),
  splitFrontier(splitFrontier_),
  sCount(sCountCand),
  sum(sumCand),
  cutDense(cand->getIdxEnd() + 1) { // Unrealizeable index.
}


IndexT CutAccum::lhImplicit(const SplitNux* cand) const {
  return rankDense <= rankLH ? cand->getImplicitCount() : 0;
}


unique_ptr<Residual> CutAccum::makeResidual(const SplitNux* cand,
					 const SampleRank spn[]) {
  if (cand->getImplicitCount() == 0) {
    return make_unique<Residual>();
  }

  double sumExpl = 0.0;
  IndexT sCountExpl = 0;
  for (int idx = static_cast<int>(cand->getIdxEnd()); idx >= static_cast<int>(cand->getIdxStart()); idx--) {
    IndexT rkThis = spn[idx].regFields(ySumThis, sCountThis);
    if (rkThis > rankDense) {
      cutDense = idx;
    }
    sCountExpl += sCountThis;
    sumExpl += ySumThis;
  }
  
  return make_unique<Residual>(sumCand - sumExpl, sCountCand - sCountExpl);
}


double CutAccum::interpolateRank(double splitQuant) const {
  return IndexRange(rankLH, rankRH - rankLH).interpolate(splitQuant);
}


CutSet::CutSet() {
}


CutSig CutSet::getCut(const SplitNux& nux) const {
  return cutSig[nux.getAccumIdx()];
}


IndexT CutSet::addCut(const SplitNux* cand) {
  cutSig.emplace_back<CutSig>(cand->getRange());
  return cutSig.size() - 1;
}


void CutAccum::trialSplit(double infoTrial,
			  IndexT idxLeft,
			  IndexT idxRight) {
  IndexT rkLeft = sampleRank[idxLeft].getRank();
  IndexT rkRight = sampleRank[idxRight].getRank();
  if (rkLeft != rkRight && infoTrial > info) {
    info = infoTrial;
    lhSCount = sCount;
    lhSum = sum;
    this->idxRight = idxRight;
    this->idxLeft = idxLeft;
    rankRH = rkRight;
    rankLH = rkLeft;
  }
}
  

void CutSet::write(const SplitNux* nux, const CutAccum* accum) {
  IndexT accumIdx = nux->getAccumIdx();
  cutSig[accumIdx].idxLeft = accum->idxLeft;
  cutSig[accumIdx].idxRight = accum->idxRight;
  cutSig[accumIdx].implicitTrue = accum->lhImplicit(nux);
  cutSig[accumIdx].quantRank = accum->interpolateRank(nux->splitQuant[nux->getPredIdx()]);
}


bool CutSet::leftCut(const SplitNux* nux) const {
  return cutSig[nux->getAccumIdx()].cutLeft;
}


void CutSet::setCutSense(IndexT cutIdx, bool sense) {
  cutSig[cutIdx].cutLeft = sense;
}


double CutSet::getQuantRank(const SplitNux* nux) const {
  return cutSig[nux->getAccumIdx()].quantRank;
}


IndexT CutSet::getIdxRight(const SplitNux* nux) const {
  return cutSig[nux->getAccumIdx()].idxRight;
}


IndexT CutSet::getIdxLeft(const SplitNux* nux) const {
  return cutSig[nux->getAccumIdx()].idxLeft;
}


IndexT CutSet::getImplicitTrue(const SplitNux* nux) const {
  return cutSig[nux->getAccumIdx()].implicitTrue;
}
