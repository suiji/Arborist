// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitaccum.cc

   @brief Methods to implement splitting.

   @author Mark Seligman
 */

#include "splitaccum.h"
#include "splitcand.h"
#include "splitfrontier.h"
#include "obspart.h"


SplitAccum::SplitAccum(const SplitCand* cand,
                       IndexT rankDense_) :
  sCount(cand->getSCount()),
  sum(cand->getSum()),
  rankDense(rankDense_),
  sCountL(sCount),
  sumL(sum),
  cutDense(cand->getIdxEnd() + 1),
  info(cand->getInfo()) {
}

SplitAccumReg::SplitAccumReg(const SplitCand* cand,
                             const SampleRank spn[],
                             const SFReg* spReg) :
  SplitAccum(cand, spReg->getDenseRank(cand)),
  monoMode(spReg->getMonoMode(cand)),
  resid(makeResidual(cand, spn)) {
}

void SplitAccumReg::split(const SFReg* spReg,
                          const SampleRank spn[],
                          SplitCand* cand) {
  if (resid != nullptr) {
    splitImpl(spn, cand);
  }
  else {
    IndexT idxEnd = cand->getIdxEnd();
    IndexT idxStart = cand->getIdxStart();
    IndexT rkThis = spn[idxEnd].regFields(ySum, sCountThis);
    splitExpl(spn, rkThis, idxEnd-1, idxStart);
  }
}

void SplitAccumReg::splitImpl(const SampleRank spn[],
                              const SplitCand* cand) {
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
    leftResidual(spn[cutDense].getRank()); // Checks denseCut/resid.

    // Checks resid/denseCut-1, ..., idxStart+1/idxStart, if applicable.
    if (cutDense > 0) {
      splitExpl(spn, rankDense, cutDense - 1, idxStart);
    }
  }
}


void SplitAccumReg::leftResidual(IndexT rkThis) {
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


void SplitAccumReg::splitExpl(const SampleRank spn[],
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

    double infoTrial = infoSplit(sumL, sum - sumL, sCountL, sCount - sCountL);
    if (infoTrial > info && rkThis != rkRight) {
      info = infoTrial;
      lhSCount = sCountL;
      rankRH = rkRight;
      rankLH = rkThis;
      rhMin = rkRight == rankDense ? cutDense : idx + 1;
    }
  }
}

/**
   @brief As above, but checks monotonicity at every index.
 */
void SplitAccumReg::splitMono(const SampleRank spn[],
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


SplitAccumCtg::SplitAccumCtg(const SplitCand* cand,
                             const SampleRank spn[],
                             SFCtg* spCtg) :
  SplitAccum(cand, spCtg->getDenseRank(cand)),
  nCtg(spCtg->getNCtg()),
  resid(makeResidual(cand, spn, spCtg)),
  ctgSum(spCtg->getSumSlice(cand)),
  ctgAccum(spCtg->getAccumSlice(cand)),
  ssL(spCtg->getSumSquares(cand)),
  ssR(0.0) {
}


// Initializes from final index and loops over remaining indices.
void SplitAccumCtg::split(const SFCtg* spCtg,
                          const SampleRank spn[],
                          SplitCand* cand) {
  if (resid != nullptr) {
    splitImpl(spn, cand);
  }
  else {
    IndexT idxEnd = cand->getIdxEnd();
    IndexT idxStart = cand->getIdxStart();
    IndexT rkThis = stateNext(spn, idxEnd);
    (void) splitExpl(spn, rkThis, idxEnd-1, idxStart);
  }
}


inline IndexT SplitAccumCtg::stateNext(const SampleRank spn[],
                                       IndexT idx) {
  PredictorT yCtg;
  IndexT rkThis = spn[idx].ctgFields(ySum, sCountThis, yCtg);

  sumL -= ySum;
  sCountL -= sCountThis;
  accumCtgSS(ySum, yCtg, ssL, ssR);

  return rkThis;
}


IndexT SplitAccumCtg::splitExpl(const SampleRank spn[],
                              IndexT rkThis,
                              IndexT idxInit,
                              IndexT idxFinal) {
  IndexT rkRight = rkThis;
  for (int idx = static_cast<int>(idxInit); idx >= static_cast<int>(idxFinal); idx--) {
    trialSplit(idx, spn[idx].getRank(), rkRight);
    rkRight = stateNext(spn, idx);
  }
  return rkRight;
}


void SplitAccumCtg::splitImpl(const SampleRank spn[],
                              const SplitCand* cand) {
  IndexT idxEnd = cand->getIdxEnd();
  IndexT idxStart = cand->getIdxStart();
  if (cutDense > idxEnd) { // Far right residual.
    resid->apply(ySum, sCountThis, ssR, ssL, this);
    sumL -= ySum;
    sCountL -= sCountThis;
    (void) splitExpl(spn, rankDense, idxEnd, idxStart);
  }
  else {
    if (cutDense == idxStart) { // Far left residual
      IndexT rkThis = splitExpl(spn, spn[idxEnd].getRank(), idxEnd, cutDense);
      trialSplitLower(rkThis);
    }
    else { // Internal residual.
      resid->apply(ySum, sCountThis, ssR, ssL, this);
      (void) splitExpl(spn, rankDense, cutDense - 1, idxStart);
    }
  }
}


void SplitAccumCtg::trialSplitLower(IndexT rkRight) {
  double infoTrial = infoSplit(ssL, ssR, sumL, sum - sumL);
  if (infoTrial > info) {
    info = infoTrial;
    lhSCount = sCountL;
    rankRH = rkRight;
    rankLH = rankDense;
    rhMin = cutDense;
  }
}


shared_ptr<Residual> SplitAccumReg::makeResidual(const SplitCand* cand,
                                                 const SampleRank spn[]) {
  if (cand->getImplicitCount() == 0) {
    return nullptr;
  }
  double sumExpl = 0.0;
  IndexT sCountExpl = 0;
  for (int idx = static_cast<int>(cand->getIdxEnd()); idx >= static_cast<int>(cand->getIdxStart()); idx--) {
    IndexT rkThis = spn[idx].regFields(ySum, sCountThis);
    if (rkThis > rankDense) {
      cutDense = idx;
    }
    sCountExpl += sCountThis;
    sumExpl += ySum;
  }
  
  return make_shared<Residual>(sum - sumExpl, sCount - sCountExpl);
}


shared_ptr<ResidualCtg>
SplitAccumCtg::makeResidual(const SplitCand* cand,
                            const SampleRank spn[],
                            SFCtg* spCtg) {
  if (cand->getImplicitCount() == 0) {
    return nullptr;
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

  return make_shared<ResidualCtg>(sum - sumExpl, sCount - sCountExpl, ctgImpl);
}


Residual::Residual(double sum_,
                   IndexT sCount_) :
  sum(sum_),
  sCount(sCount_) {
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
                        SplitAccumCtg* np) {
  sum = this->sum;
  sCount = this->sCount;
  for (PredictorT ctg = 0; ctg < ctgImpl.size(); ctg++) {
    np->accumCtgSS(ctgImpl[ctg], ctg, ssL, ssR);
  }
}
