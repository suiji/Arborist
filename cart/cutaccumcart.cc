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
#include "obs.h"

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
    splitRL(obsStart);
  }
  spReg->writeCut(cand, this);
  cand->infoGain(this);
}


void CutAccumRegCart::splitRL(IndexT idxFinal) {
  // Per-sample monotonicity constraint confined to specialized method:
  if (monoMode != 0) {
    splitMono(idxFinal);
  }

  for (IndexT idx = obsEnd - 1; idx != idxFinal; idx--) {
    if (!accumulateReg(obsCell[idx])) {
      argmaxRL(infoVar(sum, sumCand-sum, sCount, sCountCand-sCount), idx-1);
    }
  }
}


void CutAccumCtgCart::splitRL(IndexT idxFinal) {
  for (IndexT idx = obsEnd - 1; idx != idxFinal; idx--) {
    if (!accumulateCtg(obsCell[idx])) {
      argmaxRL(infoGini(ssL, ssR, sum, sumCand-sum), idx-1);
    }
  }
}


/**
   @brief As above, but checks monotonicity at every index.
 */
void CutAccumRegCart::splitMono(IndexT idxFinal) {
  bool nonDecreasing = monoMode > 0;
  for (IndexT idx = obsEnd - 1; idx!= idxFinal; idx--) {
    if (!accumulateReg(obsCell[idx])) {
      IndexT sCountR = sCountCand - sCount;
      double sumR = sumCand - sum;
      bool up = (sum * sCountR <= sumR * sCount);
      if (nonDecreasing ? up : !up) {
	argmaxRL(infoVar(sum, sumR, sCount, sCountR), idx-1);
      }
    }
  }
}


void CutAccumRegCart::splitImpl(const SplitNux* cand) {
  if (cutResidual < obsEnd) {
    // Tries obsEnd/obsEnd-1, ..., denseCut+1/denseCut.
    // Ordinary R to L, beginning at rank index zero, up to cutResidual.
    splitRL(cutResidual);
    splitResidual(); // Tries denseCut/resid.
  }
  // Tries resid/denseCut-1, ..., obsStart+1/obsStart, if applicable.
  // Rightmost observation is residual, with residual rank index.
  // Follow R to L with rank index beginning at current rkIdx;
  if (cutResidual > obsStart) {
    residualLR(cand);
  }
}


void CutAccumRegCart::residualLR(const SplitNux* cand) {
  if (monoMode != 0) {
    residualLRMono(cand);
    return;
  }

  residualReg(obsCell, cand);
  argmaxResidual(infoVar(sum, sumCand-sum, sCount, sCountCand-sCount), false);

  for (IndexT idx = cutResidual - 1; idx != obsStart; idx--) {
    if (!accumulateReg(obsCell[idx])) {
      argmaxRL(infoVar(sum, sumCand-sum, sCount, sCountCand-sCount), idx-1);
    }
  }
}


void CutAccumCtgCart::residualLR(const SplitNux* cand) {
  residualCtg(obsCell, cand);
  argmaxResidual(infoGini(ssL, ssR, sum, sumCand-sum), false);

  for (IndexT idx = cutResidual - 1; idx != obsStart; idx--) {
    if (!accumulateCtg(obsCell[idx])) {
      argmaxRL(infoGini(ssL, ssR, sum, sumCand-sum), idx -1);
    }
  }
}


void CutAccumRegCart::residualLRMono(const SplitNux* cand) {
  bool nonDecreasing = monoMode > 0;

  residualReg(obsCell, cand);
  IndexT sCountR = sCountCand - sCount;
  double sumR = sumCand - sum;
  bool up = (sum * sCountR <= sumR * sCount);
  if (nonDecreasing ? up : !up) {
    argmaxResidual(infoVar(sum, sumR, sCount, sCountR), false);
  }

  for (IndexT idx = cutResidual - 1; idx != obsStart; idx--) {
    if (!accumulateReg(obsCell[idx])) {
      sCountR = sCountCand - sCount;
      sumR = sumCand - sum;
      up = (sum * sCountR <= sumR * sCount);
      if (nonDecreasing ? up : !up) {
	argmaxRL(infoVar(sum, sumR, sCount, sCountR), idx-1);
      }
    }
  }
}


void CutAccumRegCart::splitResidual() {
  (void) accumulateReg(obsCell[cutResidual]);
  IndexT sCountR = sCountCand - sCount;
  double sumR = sumCand - sum;
  bool up = (sum * sCountR <= sumR * sCount);
  if (monoMode == 0 || (monoMode > 0 && up) || (monoMode < 0 && !up)) {
    argmaxResidual(infoVar(sum, sumR, sCount, sCountR), true);
  }
}


void CutAccumCtgCart::splitResidual() {
  (void) accumulateCtg(obsCell[cutResidual]);
  argmaxResidual(infoGini(ssL, ssR, sum, sumCand-sum), true);
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
    splitRL(obsStart);
  }
  spCtg->writeCut(cand, this);
  cand->infoGain(this);
}


void CutAccumCtgCart::splitImpl(const SplitNux* cand) {
  if (cutResidual < obsEnd) {
    // Tries obsEnd/obsEnd-1, ..., denseCut+1/denseCut.
    // Ordinary R to L, beginning at rank index zero, up to cutResidual.
    splitRL(cutResidual);
    splitResidual(); // Tries denseCut/resid;
  }
  // Tries resid/denseCut-1, ..., obsStart+1/obsStart, if applicable.
  // Rightmost observation is residual, with residual rank index.
  // Follow R to L with rank index beginning at current rkIdx;
  if (cutResidual > obsStart) {
    residualLR(cand);
  }
}
