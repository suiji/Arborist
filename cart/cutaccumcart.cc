// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file cutaccumcart.cc

   @brief Methods to implement CART-style splitting.

   @author Mark Seligman
 */

#include "cutaccumcart.h"
#include "splitnux.h"
#include "sfcart.h"
#include "partition.h"

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
  if (cand->getImplicitCount() != 0) {
    splitImpl(cand);
  }
  else {
    (void) splitRL(obsStart);
  }
  spReg->writeCut(cand, this);
  cand->infoGain(this);
}


IndexT CutAccumRegCart::splitRL(IndexT idxFinal,
				IndexT rkIdx) {
  // Per-sample monotonicity constraint confined to specialized method:
  if (monoMode != 0) {
    return splitMono(idxFinal, rkIdx);
  }

  ObsReg obsThis = obsCell[obsTop].unpackReg();
  for (IndexT idx = obsTop; idx-- != idxFinal; ) {
    sum -= obsThis.ySum;
    sCount -= obsThis.sCount;
    bool tiedRight = obsThis.tied;
    obsThis = obsCell[idx].unpackReg();
    if (tiedRight)
      continue;

    argmaxRL(infoVar(sum, sumCand-sum, sCount, sCountCand-sCount), idx, rkIdx);
    rkIdx++;
  }

  return rkIdx;
}


IndexT CutAccumCtgCart::splitRL(IndexT idxFinal,
				IndexT rkIdx) {
  ObsCtg obsThis = obsCell[obsTop].unpackCtg();
  for (IndexT idx = obsTop; idx-- != idxFinal; ) {
    sum -= obsThis.ySum;
    sCount -= obsThis.sCount;
    accumCtgSS(obsThis.ySum, obsThis.yCtg);
    bool tiedRight = obsThis.tied;
    obsThis = obsCell[idx].unpackCtg();
    if (tiedRight)
      continue;

    argmaxRL(infoGini(ssL, ssR, sum, sumCand-sum), idx, rkIdx);
    rkIdx++;
  }
  return rkIdx;
}


/**
   @brief As above, but checks monotonicity at every index.
 */
IndexT CutAccumRegCart::splitMono(IndexT idxFinal,
				  IndexT rkIdx) {
  bool nonDecreasing = monoMode > 0;
  ObsReg obsThis = obsCell[obsTop].unpackReg();
  for (IndexT idx = obsTop; idx-- != idxFinal; ) {
    bool tiedRight = obsThis.tied;
    sum -= obsThis.ySum;
    sCount -= obsThis.sCount;
    obsThis = obsCell[idx].unpackReg();
    if (tiedRight)
      continue;

    IndexT sCountR = sCountCand - sCount;
    double sumR = sumCand - sum;
    bool up = (sum * sCountR <= sumR * sCount);
    if (nonDecreasing ? up : !up) {
      argmaxRL(infoVar(sum, sumR, sCount, sCountR), idx, rkIdx);
    }
    rkIdx++;
  }
  return rkIdx;
}


void CutAccumRegCart::splitImpl(const SplitNux* cand) {
  IndexT rkIdx = 0;
  if (cutResidual <= obsTop) {
    // Tries obsEnd/obsEnd-1, ..., denseCut+1/denseCut.
    // Ordinary R to L, beginning at rank index zero, up to cutResidual.
    rkIdx = splitRL(cutResidual, 1);
    splitResidual(rkIdx); // Tries denseCut/resid.
  }
  // Tries resid/denseCut-1, ..., obsStart+1/obsStart, if applicable.
  // Rightmost observation is residual, with residual rank index.
  // Follow R to L with rank index beginning at current rkIdx;
  if (cutResidual > obsStart) {
    residualLR(cand, rkIdx + 1);
  }
}


void CutAccumRegCart::residualLR(const SplitNux* cand,
				 IndexT rkIdxL) {
  if (monoMode != 0) {
    splitMonoDense(cand, rkIdxL);
    return;
  }

  ObsReg obsThis = Obs::residualReg(obsCell, cand);
  IndexT idxInit = cutResidual - 1;
  IndexT rkIdxR = 0; // Right rank index initialized to residual.
  for (IndexT idx = idxInit + 1; idx-- != obsStart; ) {
    sum -= obsThis.ySum;
    sCount -= obsThis.sCount;
    bool tiedRight = obsThis.tied;
    obsThis = obsCell[idx].unpackReg();
    if (tiedRight)
      continue;

    argmaxRL(infoVar(sum, sumCand-sum, sCount, sCountCand-sCount), idx, rkIdxR, rkIdxL);
    rkIdxR = rkIdxL;
    rkIdxL++;
  }
}


void CutAccumCtgCart::residualLR(const SplitNux* cand,
				 IndexT rkIdxL) {
  vector<double> ctgResid(nodeSum);
  Obs::residualCtg(obsCell, cand, sum, sCount, ctgResid);
  for (PredictorT ctg = 0; ctg != ctgResid.size(); ctg++) {
    accumCtgSS(ctgResid[ctg], ctg);
  }

  ObsCtg obsThis(0, 0.0, 0, false); // Dummy.
  IndexT idxInit = cutResidual - 1;
  IndexT rkIdxR = 0; // Right rank, index initialized to residual.
  for (IndexT idx = idxInit + 1; idx-- != obsStart; ) {
    sum -= obsThis.ySum;
    sCount -= obsThis.sCount;
    accumCtgSS(obsThis.ySum, obsThis.yCtg);
    bool tiedRight = obsThis.tied;
    obsThis = obsCell[idx].unpackCtg();
    if (tiedRight)
      continue;

    argmaxRL(infoGini(ssL, ssR, sum, sumCand-sum), idx, rkIdxR, rkIdxL);
    rkIdxR = rkIdxL;
    rkIdxL++;
  }
}
void CutAccumRegCart::splitMonoDense(const SplitNux* cand,
				     IndexT rkIdxL) {
  bool nonDecreasing = monoMode > 0;
  IndexT idxInit = cutResidual - 1;
  ObsReg obsThis = Obs::residualReg(obsCell, cand);
  IndexT rkIdxR = 0; // Right rank index initialized to residual.
  for (IndexT idx = idxInit + 1; idx-- != obsStart; ) {
    bool tiedRight = obsThis.tied;
    sum -= obsThis.ySum;
    sCount -= obsThis.sCount;
    obsThis = obsCell[idx].unpackReg();
    if (tiedRight)
      continue;

    IndexT sCountR = sCountCand - sCount;
    double sumR = sumCand - sum;
    bool up = (sum * sCountR <= sumR * sCount);
    if (nonDecreasing ? up : !up) {
      argmaxRL(infoVar(sum, sumR, sCount, sCountR), idx, rkIdxR, rkIdxL);//, rkThis, rkRight);
    }
    rkIdxR = rkIdxL;
    rkIdxL++;
  }
}


void CutAccumRegCart::splitResidual(IndexT rkIdx) {
  ObsReg obsThis = obsCell[cutResidual].unpackReg();
  sum -= obsThis.ySum;
  sCount -= obsThis.sCount;

  IndexT sCountR = sCountCand - sCount;
  double sumR = sumCand - sum;
  bool up = (sum * sCountR <= sumR * sCount);
  if (monoMode == 0 || (monoMode > 0 && up) || (monoMode < 0 && !up)) {
    if (Accum::trialSplit(infoVar(sum, sumR, sCount, sCountR))) {
      rankIdxR = rkIdx;
      rankIdxL = 0; // Residual rank index.
      obsRight = cutResidual;
      obsLeft = cutResidual == obsStart ? cutResidual : cutResidual - 1;
    }
  }
}


void CutAccumCtgCart::splitResidual(IndexT rkIdx) {
  ObsCtg obsThis = obsCell[cutResidual].unpackCtg();
  sum -= obsThis.ySum;
  sCount -= obsThis.sCount;
  accumCtgSS(obsThis.ySum, obsThis.yCtg);

  if (Accum::trialSplit(infoGini(ssL, ssR, sum, sumCand-sum))) {
    rankIdxR = rkIdx;
    rankIdxL = 0; // Residual rank index.
    obsRight = cutResidual;
    obsLeft = cutResidual == 0 ? cutResidual : cutResidual - 1;
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
  if (cand->getImplicitCount() != 0) {
    splitImpl(cand);
  }
  else {
    (void) splitRL(obsStart);
  }
  spCtg->writeCut(cand, this);
  cand->infoGain(this);
}


void CutAccumCtgCart::splitImpl(const SplitNux* cand) {
  IndexT rkIdx = 0;
  if (cutResidual <= obsTop) {
    // Tries obsEnd/obsEnd-1, ..., denseCut+1/denseCut.
    // Ordinary R to L, beginning at rank index zero, up to cutResidual.
    IndexT rkIdx = splitRL(cutResidual, 1);
    splitResidual(rkIdx); // Tries denseCut/resid;
  }
  // Tries resid/denseCut-1, ..., obsStart+1/obsStart, if applicable.
  // Rightmost observation is residual, with residual rank index.
  // Follow R to L with rank index beginning at current rkIdx;
  if (cutResidual > obsStart) {
    residualLR(cand, rkIdx + 1);
  }
}
