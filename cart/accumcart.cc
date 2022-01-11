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
#include "partition.h"
#include "residual.h"

CutAccumRegCart::CutAccumRegCart(const SplitNux* cand,
				 const SFRegCart* spReg) :
  CutAccumReg(cand, spReg) {
}


void CutAccumRegCart::split(const SFRegCart* spReg,
			    SplitNux* cand) {
  CutAccumRegCart cutAccum(cand, spReg);
  cutAccum.splitReg(spReg, cand);
}



void CutAccumRegCart::splitReg(const SFRegCart* spReg,
			       SplitNux* cand) {
  if (!resid->isEmpty()) {
    splitImpl(cand);
  }
  else {
    IndexT rkThis = sampleRank[idxEnd].regFields(ySumThis, sCountThis);
    splitExpl(rkThis, idxEnd-1, idxStart);
  }
  spReg->writeCut(cand, this);
  cand->infoGain(this);
}


void CutAccumRegCart::splitImpl(const SplitNux* cand) {
  if (cutDense > idxEnd) {
    // Checks resid/idxEnd, ..., idxStart+1/idxStart.
    resid->apply(ySumThis, sCountThis);
    splitExpl(rankDense, idxEnd, idxStart);
  }
  else {
    // Checks idxEnd/idxEnd-1, ..., denseCut+1/denseCut.
    IndexT rkThis = sampleRank[idxEnd].regFields(ySumThis, sCountThis);
    splitExpl(rkThis, idxEnd-1, cutDense);
    splitResidual(sampleRank[cutDense].getRank()); // Checks denseCut/resid.

    // Checks resid/denseCut-1, ..., idxStart+1/idxStart, if applicable.
    if (cutDense > idxStart) {
      resid->apply(ySumThis, sCountThis);
      splitExpl(rankDense, cutDense - 1, idxStart);
    }
  }
}


void CutAccumRegCart::splitResidual(IndexT rkThis) {
  // Rank exposed from previous invocation of splitExpl():
  sum -= ySumThis;
  sCount -= sCountThis;
  resid->apply(ySumThis, sCountThis);

  IndexT sCountR = sCountCand - sCount;
  double sumR = sumCand - sum;
  double infoTrial = infoVar(sum, sumR, sCount, sCountR);
  if (infoTrial > info) {
    bool up = (sum * sCountR <= sumR * sCount);
    if (monoMode == 0 || (monoMode >0 && up) || (monoMode < 0 && !up)) {
      lhSCount = sCount;
      rankRH = rkThis;
      rankLH = rankDense;
      idxRight = cutDense;
      info = infoTrial;
    }
  }
}


void CutAccumRegCart::splitExpl(IndexT rkThis,
                              IndexT idxInit,
                              IndexT idxFinal) {
  // Per-sample monotonicity constraint confined to specialized method:
  if (monoMode != 0) {
    splitMono(rkThis, idxInit, idxFinal);
    return;
  }

  for (int idx = static_cast<int>(idxInit); idx >= static_cast<int>(idxFinal); idx--) {
    IndexT rkRight = rkThis;
    sum -= ySumThis;
    sCount -= sCountThis;
    rkThis = sampleRank[idx].regFields(ySumThis, sCountThis);

    if (rkThis != rkRight)
      trialRight(infoVar(sum, sumCand - sum, sCount, sCountCand - sCount), idx, rkThis, rkRight);
  }
}

/**
   @brief As above, but checks monotonicity at every index.
 */
void CutAccumRegCart::splitMono(IndexT rkThis,
			    IndexT idxInit,
			    IndexT idxFinal) {
  bool nonDecreasing = monoMode > 0;
  for (int idx = static_cast<int>(idxInit); idx >= static_cast<int>(idxFinal); idx--) {
    IndexT rkRight = rkThis;
    sum -= ySumThis;
    sCount -= sCountThis;
    rkThis = sampleRank[idx].regFields(ySumThis, sCountThis);

    //    localMax(nonDecreasing);
    IndexT sCountR = sCountCand - sCount;
    double sumR = sumCand - sum;
    double infoTrial = infoVar(sum, sumR, sCount, sCountR);
    if (infoTrial > info && rkThis != rkRight) {
      bool up = (sum * sCountR <= sumR * sCount);
      if (nonDecreasing ? up : !up) {
        info = infoTrial;
        lhSCount = sCount;
        rankRH = rkRight;
        rankLH = rkThis;
        idxRight = rkRight == rankDense ? cutDense : idx + 1;
      }
    }
  }
}


CutAccumCtgCart::CutAccumCtgCart(const SplitNux* cand,
				 SFCtgCart* spCtg) :
  CutAccumCtg(cand, spCtg) {
}


void CutAccumCtgCart::split(SFCtgCart* spCtg,
			    SplitNux* cand) {
  CutAccumCtgCart cutAccum(cand, spCtg);
  cutAccum.splitCtg(spCtg, cand);
}


// Initializes from final index and loops over remaining indices.
void CutAccumCtgCart::splitCtg(const SFCtgCart* spCtg,
			       SplitNux* cand) {
  if (!resid->isEmpty()) {
    splitImpl(cand);
  }
  else {
    stateNext(idxEnd);
    splitExpl(sampleRank[idxEnd].getRank(), idxEnd-1, idxStart);
  }
  spCtg->writeCut(cand, this);
  cand->infoGain(this);
}


inline void CutAccumCtgCart::stateNext(IndexT idx) {
  PredictorT yCtg;
  (void) sampleRank[idx].ctgFields(ySumThis, sCountThis, yCtg);

  sum -= ySumThis;
  sCount -= sCountThis;
  accumCtgSS(ySumThis, yCtg);
}


void CutAccumCtgCart::splitExpl(IndexT rkThis,
				IndexT idxInit,
				IndexT idxFinal) {
  for (int idx = static_cast<int>(idxInit); idx >= static_cast<int>(idxFinal); idx--) {
    IndexT rkRight = rkThis;
    rkThis = sampleRank[idx].getRank();
    if (rkThis != rkRight)
      trialRight(infoGini(ssL, ssR, sum, sumCand - sum), idx, rkThis, rkRight);
    stateNext(idx);
  }
}


void CutAccumCtgCart::splitImpl(const SplitNux* cand) {
  if (cutDense > idxEnd) { // Far right residual:  apply and split to left.
    residualAndLeft(idxEnd, idxStart);
  }
  else { // Split far right, then residual, then possibly left.
    splitExpl(sampleRank[idxEnd].getRank(), idxEnd, cutDense);
    splitResidual(infoGini(ssL, ssR, sum, sumCand - sum), sampleRank[cutDense].getRank());
    if (cutDense > idxStart) { // Internal residual:  apply and split to left.
      residualAndLeft(cutDense - 1, idxStart);
    }
  }
}


void CutAccumCtgCart::residualAndLeft(IndexT idxLeft,
				      IndexT idxStart) {
  ySumThis = resid->sum;
  sCountThis = resid->sCount;
  applyResid(resid->ctgImpl);
  sum -= ySumThis;
  sCount -= sCountThis;
  splitExpl(rankDense, idxLeft, idxStart);
}
