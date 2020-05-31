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
#include "obspart.h"
#include "residual.h"


CutAccum::CutAccum(const SplitNux* cand,
		   const SplitFrontier* splitFrontier) :
  Accum(splitFrontier, cand),
  cutDense(cand->getIdxEnd() + 1) { // Unrealizeable index.
}


IndexT CutAccum::lhImplicit(const SplitNux* cand) const {
  return rankDense <= rankLH ? cand->getImplicitCount() : 0;
}


unique_ptr<Residual> CutAccumReg::makeResidual(const SplitNux* cand,
					       const SampleRank spn[]) {
  if (cand->getImplicitCount() == 0) {
    return make_unique<Residual>();
  }

  double sumExpl = 0.0;
  IndexT sCountExpl = 0;
  for (int idx = static_cast<int>(idxEnd); idx >= static_cast<int>(idxStart); idx--) {
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


CutAccumReg::CutAccumReg(const SplitNux* cand,
			 const SFReg* sfReg) :
  CutAccum(cand, sfReg),
  monoMode(sfReg->getMonoMode(cand)),
  resid(CutAccumReg::makeResidual(cand, sampleRank)) {
}

CutAccumReg::~CutAccumReg() {
}


CutAccumCtg::CutAccumCtg(const SplitNux* cand,
			 SFCtg* sfCtg) :
  CutAccum(cand, sfCtg),
  nCtg(sfCtg->getNCtg()),
  resid(makeResidual(cand, sfCtg)),
  ctgSum(sfCtg->getSumSlice(cand)),
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
  for (int idx = static_cast<int>(cand->getIdxEnd()); idx >= static_cast<int>(cand->getIdxStart()); idx--) {
    PredictorT yCtg;
    IndexT rkThis = sampleRank[idx].ctgFields(ySumThis, sCountThis, yCtg);
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


void ResidualCtg::apply(FltVal& sum,
                        IndexT& sCount,
                        double& ssR,
                        double& ssL,
                        CutAccumCtg* np) {
  sum = this->sum;
  sCount = this->sCount;
  for (PredictorT ctg = 0; ctg < ctgImpl.size(); ctg++) {
    np->accumCtgSS(ctgImpl[ctg], ctg, ssL, ssR);
  }
}

