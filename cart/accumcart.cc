// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitaccum.cc

   @brief Methods to implement CART-style splitting.

   @author Mark Seligman
 */

#include "accumcart.h"
#include "splitnux.h"
#include "sfcart.h"
#include "obspart.h"
#include "residual.h"

AccumCartReg::AccumCartReg(const SplitNux* cand,
                             const SampleRank spn[],
                             const SFCartReg* spReg) :
  Accum(cand, spReg->getDenseRank(cand)),
  monoMode(spReg->getMonoMode(cand)),
  resid(Accum::makeResidual(cand, spn)) {
}


AccumCartReg::~AccumCartReg() {
}


void AccumCartReg::split(const SFCartReg* spReg,
                          const SampleRank spn[],
                          SplitNux* cand) {
  if (!resid->isEmpty()) {
    splitImpl(spn, cand);
  }
  else {
    IndexT idxEnd = cand->getIdxEnd();
    IndexT idxStart = cand->getIdxStart();
    IndexT rkThis = spn[idxEnd].regFields(ySum, sCountThis);
    splitExpl(spn, rkThis, idxEnd-1, idxStart);
  }
  cand->writeNum(spReg, this);
}


void AccumCartReg::splitImpl(const SampleRank spn[],
                              const SplitNux* cand) {
  IndexT idxEnd = cand->getIdxEnd();
  IndexT idxStart = cand->getIdxStart();
  if (cutDense > idxEnd) {
    // Checks resid/idxEnd, ..., idxStart+1/idxStart.
    resid->apply(ySum, sCountThis);
    splitExpl(spn, rankDense, idxEnd, idxStart);
  }
  else {
    // Checks idxEnd/idxEnd-1, ..., denseCut+1/denseCut.
    IndexT rkThis = spn[idxEnd].regFields(ySum, sCountThis);
    splitExpl(spn, rkThis, idxEnd-1, cutDense);
    splitResidual(spn[cutDense].getRank()); // Checks denseCut/resid.

    // Checks resid/denseCut-1, ..., idxStart+1/idxStart, if applicable.
    if (cutDense > idxStart) {
      resid->apply(ySum, sCountThis);
      splitExpl(spn, rankDense, cutDense - 1, idxStart);
    }
  }
}


void AccumCartReg::splitResidual(IndexT rkThis) {
  // Rank exposed from previous invocation of splitExpl():
  sumL -= ySum;
  sCountL -= sCountThis;
  resid->apply(ySum, sCountThis);

  IndexT sCountR = sCount - sCountL;
  double sumR = sum - sumL;
  double infoTrial = infoSplit(sumL, sumR, sCountL, sCountR);
  if (infoTrial > info) {
    bool up = (sumL * sCountR <= sumR * sCountL);
    if (monoMode == 0 || (monoMode >0 && up) || (monoMode < 0 && !up)) {
      lhSCount = sCountL;
      rankRH = rkThis;
      rankLH = rankDense;
      rhMin = cutDense;
      info = infoTrial;
    }
  }
}


void AccumCartReg::splitExpl(const SampleRank spn[],
                              IndexT rkThis,
                              IndexT idxInit,
                              IndexT idxFinal) {
  // Per-sample monotonicity constraint confined to specialized method:
  if (monoMode != 0) {
    splitMono(spn, rkThis, idxInit, idxFinal);
    return;
  }

  for (int idx = static_cast<int>(idxInit); idx >= static_cast<int>(idxFinal); idx--) {
    IndexT rkRight = rkThis;
    sumL -= ySum;
    sCountL -= sCountThis;
    rkThis = spn[idx].regFields(ySum, sCountThis);

    trialSplit(idx, rkThis, rkRight);
  }
}

/**
   @brief As above, but checks monotonicity at every index.
 */
void AccumCartReg::splitMono(const SampleRank spn[],
                              IndexT rkThis,
                              IndexT idxInit,
                              IndexT idxFinal) {
  bool nonDecreasing = monoMode > 0;
  for (int idx = static_cast<int>(idxInit); idx >= static_cast<int>(idxFinal); idx--) {
    IndexT rkRight = rkThis;
    sumL -= ySum;
    sCountL -= sCountThis;
    rkThis = spn[idx].regFields(ySum, sCountThis);

    //    localMax(nonDecreasing);
    IndexT sCountR = sCount - sCountL;
    double sumR = sum - sumL;
    double infoTrial = infoSplit(sumL, sumR, sCountL, sCountR);
    if (infoTrial > info && rkThis != rkRight) {
      bool up = (sumL * sCountR <= sumR * sCountL);
      if (nonDecreasing ? up : !up) {
        info = infoTrial;
        lhSCount = sCountL;
        rankRH = rkRight;
        rankLH = rkThis;
        rhMin = rkRight == rankDense ? cutDense : idx + 1;
      }
    }
  }
}


AccumCartCtg::AccumCartCtg(const SplitNux* cand,
                             const SampleRank spn[],
                             SFCartCtg* spCtg) :
  Accum(cand, spCtg->getDenseRank(cand)),
  nCtg(spCtg->getNCtg()),
  resid(makeResidual(cand, spn, spCtg)),
  ctgSum(spCtg->getSumSlice(cand)),
  ctgAccum(spCtg->getAccumSlice(cand)),
  ssL(spCtg->getSumSquares(cand)),
  ssR(0.0) {
}


AccumCartCtg::~AccumCartCtg() {
}


// Initializes from final index and loops over remaining indices.
void AccumCartCtg::split(const SFCartCtg* spCtg,
                          const SampleRank spn[],
                          SplitNux* cand) {
  if (!resid->isEmpty()) {
    splitImpl(spn, cand);
  }
  else {
    IndexT idxEnd = cand->getIdxEnd();
    IndexT idxStart = cand->getIdxStart();
    stateNext(spn, idxEnd);
    splitExpl(spn, spn[idxEnd].getRank(), idxEnd-1, idxStart);
  }
  cand->writeNum(spCtg, this);
}


inline void AccumCartCtg::stateNext(const SampleRank spn[],
				     IndexT idx) {
  PredictorT yCtg;
  (void) spn[idx].ctgFields(ySum, sCountThis, yCtg);

  sumL -= ySum;
  sCountL -= sCountThis;
  accumCtgSS(ySum, yCtg, ssL, ssR);
}


void AccumCartCtg::splitExpl(const SampleRank spn[],
                              IndexT rkThis,
                              IndexT idxInit,
                              IndexT idxFinal) {
  for (int idx = static_cast<int>(idxInit); idx >= static_cast<int>(idxFinal); idx--) {
    IndexT rkRight = rkThis;
    rkThis = spn[idx].getRank();
    trialSplit(idx, rkThis, rkRight);
    stateNext(spn, idx);
  }
}


void AccumCartCtg::splitImpl(const SampleRank spn[],
                              const SplitNux* cand) {
  IndexT idxEnd = cand->getIdxEnd();
  IndexT idxStart = cand->getIdxStart();
  if (cutDense > idxEnd) { // Far right residual:  apply and split to left.
    residualAndLeft(spn, idxEnd, idxStart);
  }
  else { // Split far right, then residual, then possibly left.
    splitExpl(spn, spn[idxEnd].getRank(), idxEnd, cutDense);
    splitResidual(infoSplit(ssL, ssR, sumL, sum - sumL), spn[cutDense].getRank());
    if (cutDense > idxStart) { // Internal residual:  apply and split to left.
      residualAndLeft(spn, cutDense - 1, idxStart);
    }
  }
}


void AccumCartCtg::residualAndLeft(const SampleRank spn[],
				    IndexT idxLeft,
				    IndexT idxStart) {
  resid->apply(ySum, sCountThis, ssR, ssL, this);
  sumL -= ySum;
  sCountL -= sCountThis;
  splitExpl(spn, rankDense, idxLeft, idxStart);
}


unique_ptr<ResidualCtg>
AccumCartCtg::makeResidual(const SplitNux* cand,
                            const SampleRank spn[],
                            const SFCartCtg* spCtg) {
  if (cand->getImplicitCount() == 0) {
    return make_unique<ResidualCtg>();
  }

  vector<double> ctgImpl(spCtg->getSumSlice(cand));
  double sumExpl = 0.0;
  IndexT sCountExpl = 0;
  for (int idx = static_cast<int>(cand->getIdxEnd()); idx >= static_cast<int>(cand->getIdxStart()); idx--) {
    PredictorT yCtg;
    IndexT rkThis = spn[idx].ctgFields(ySum, sCountThis, yCtg);
    if (rkThis > rankDense) {
      cutDense = idx;
    }
    sCountExpl += sCountThis;
    ctgImpl[yCtg] -= ySum;
    sumExpl += ySum;
  }

  return make_unique<ResidualCtg>(sum - sumExpl, sCount - sCountExpl, ctgImpl);
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
                        AccumCartCtg* np) {
  sum = this->sum;
  sCount = this->sCount;
  for (PredictorT ctg = 0; ctg < ctgImpl.size(); ctg++) {
    np->accumCtgSS(ctgImpl[ctg], ctg, ssL, ssR);
  }
}
