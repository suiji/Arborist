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
#include "cutset.h"
#include "splitfrontier.h"
#include "partition.h"


CutAccum::CutAccum(const SplitNux* cand,
		   const SplitFrontier* splitFrontier) :
  Accum(splitFrontier, cand),
  cutDense(cand->getIdxEnd() + 1) { // Unrealizeable index.
}


IndexT CutAccum::lhImplicit(const SplitNux* cand) const {
  return rankDense <= rankLH ? cand->getImplicitCount() : 0;
}


unique_ptr<Residual> CutAccumReg::makeResidual(const SplitNux* cand,
					       const ObsCell spn[]) {
  if (cand->getImplicitCount() == 0) {
    return make_unique<Residual>();
  }

  double sumExpl = 0.0;
  IndexT sCountExpl = 0;
  for (IndexT idx = idxEnd + 1; idx-- != idxStart; ) {
    IndexT rkThis;
    ySumThis = spn[idx].regFields(sCountThis, rkThis);
    if (rkThis > rankDense) {
      cutDense = idx;
    }
    sCountExpl += sCountThis;
    sumExpl += ySumThis;
  }
  
  return make_unique<Residual>(sumCand - sumExpl, sCountCand - sCountExpl);
}


double CutAccum::interpolateRank(const SplitNux* cand) const {
  return IndexRange(rankLH, rankRH - rankLH).interpolate(cand->getSplitQuant());
}


void CutAccum::trialSplit(double infoTrial,
			  IndexT idxLeft,
			  IndexT idxRight) {
  IndexT rkLeft = obsCell[idxLeft].getRank();
  IndexT rkRight = obsCell[idxRight].getRank();
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


CutAccumReg::CutAccumReg(const SplitNux* cand,
			 const SFReg* sfReg) :
  CutAccum(cand, sfReg),
  monoMode(sfReg->getMonoMode(cand)),
  resid(CutAccumReg::makeResidual(cand, obsCell)) {
}


CutAccumCtg::CutAccumCtg(const SplitNux* cand,
			 SFCtg* sfCtg) :
  CutAccum(cand, sfCtg),
  nCtg(sfCtg->getNCtg()),
  resid(makeResidual(cand, sfCtg)),
  nodeSum(sfCtg->getSumSlice(cand)),
  ctgAccum(sfCtg->getAccumSlice(cand)),
  ssL(sfCtg->getSumSquares(cand)),
  ssR(0.0) {
}



unique_ptr<ResidualCtg> CutAccumCtg::makeResidual(const SplitNux* cand,
						  const SFCtg* spCtg) {
  if (cand->getImplicitCount() != 0) {
    return make_unique<ResidualCtg>();
  }

  vector<double> ctgImpl(spCtg->getSumSlice(cand));
  double sumExpl = 0.0;
  IndexT sCountExpl = 0;
  for (IndexT idx = cand->getIdxEnd() + 1; idx-- != cand->getIdxStart(); ) {
    PredictorT yCtg;
    IndexT rkThis = obsCell[idx].ctgFields(ySumThis, sCountThis, yCtg);
    if (rkThis > rankDense) {
      cutDense = idx;
    }
    ctgImpl[yCtg] -= ySumThis;
    sumExpl += ySumThis;
    sCountExpl += sCountThis;
  }

  return make_unique<ResidualCtg>(sumCand - sumExpl, sCountCand - sCountExpl, ctgImpl);
}


ResidualCtg::ResidualCtg(double sum_,
                         IndexT sCount_,
                         const vector<double>& ctgImpl_) :
  Residual(sum_, sCount_),
  ctgImpl(ctgImpl_) {
}
