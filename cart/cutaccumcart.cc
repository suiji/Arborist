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
    if (monoMode != 0)
      splitImplMono(cand);
    else
      splitImpl(cand);
  }
  else {
    if (monoMode != 0)
      splitRLMono(obsStart, obsEnd);
    else
      splitRL(obsStart, obsEnd);
  }
  spReg->writeCut(cand, this);
  cand->infoGain(this);
}


void CutAccumRegCart::splitRL(IndexT idxStart, IndexT idxEnd) {
  for (IndexT idx = idxEnd - 1; idx != idxStart; idx--) {
    if (!accumulateReg(obsCell[idx])) {
      argmaxRL(infoVar(sum, sumCand-sum, sCount, sCountCand-sCount), idx-1);
    }
  }
}


void CutAccumRegCart::splitRLMono(IndexT idxStart, IndexT idxEnd) {
  for (IndexT idx = idxEnd - 1; idx!= idxStart; idx--) {
    if (!accumulateReg(obsCell[idx])) {
      argmaxRL((senseMonotone() && infoVar(sum, sumCand - sum, sCount, sCountCand - sCount)), idx-1);
    }
  }
}


void CutAccumCtgCart::splitRL(IndexT idxStart, IndexT idxEnd) {
  for (IndexT idx = idxEnd - 1; idx != idxStart; idx--) {
    if (!accumulateCtg(obsCell[idx])) {
      argmaxRL(infoGini(ssL, ssR, sum, sumCand-sum), idx-1);
    }
  }
}


void CutAccumRegCart::splitImpl(const SplitNux* cand) {
  if (cutResidual < obsEnd) {
    // Tries obsEnd/obsEnd-1, ..., cut+1/cut.
    // Ordinary R to L, beginning at rank index zero, up to cutResidual.
    splitRL(cutResidual, obsEnd);
    splitResidual(); // Tries cut/resid.
  }
  // Tries resid/cut-1, ..., obsStart+1/obsStart, if applicable.
  // Rightmost observation is residual, with residual rank index.
  // Follow R to L with rank index beginning at current rkIdx;
  if (cutResidual > obsStart) {
    residualRL(cand);
  }
}


void CutAccumRegCart::splitImplMono(const SplitNux* cand) {
  if (cutResidual < obsEnd) {
    // Tries obsEnd/obsEnd-1, ..., cut+1/cut.
    // Ordinary R to L, beginning at rank index zero, up to cutResidual.
    splitRLMono(cutResidual, obsEnd);
    splitResidual(); // Tries cut/resid.
  }
  // Tries resid/cut-1, ..., obsStart+1/obsStart, if applicable.
  // Rightmost observation is residual, with residual rank index.
  // Follow R to L with rank index beginning at current rkIdx;
  if (cutResidual > obsStart) {
    residualRLMono(cand);
  }
}


void CutAccumRegCart::residualRL(const SplitNux* cand) {
  residualReg(obsCell, cand);
  argmaxResidual(infoVar(sum, sumCand-sum, sCount, sCountCand-sCount), false);
  splitRL(obsStart, cutResidual);
}


void CutAccumCtgCart::residualRL(const SplitNux* cand) {
  residualCtg(obsCell, cand);
  argmaxResidual(infoGini(ssL, ssR, sum, sumCand-sum), false);
  splitRL(obsStart, cutResidual);
}


void CutAccumRegCart::residualRLMono(const SplitNux* cand) {
  residualReg(obsCell, cand);
  argmaxResidual((senseMonotone() && infoVar(sum, sumCand - sum, sCount, sCountCand - sCount)), false);
  splitRLMono(obsStart, cutResidual);
}


void CutAccumRegCart::splitResidual() {
  (void) accumulateReg(obsCell[cutResidual]);
  argmaxResidual(((monoMode == 0 || senseMonotone()) && infoVar(sum, sumCand - sum, sCount, sCountCand - sCount)), true);
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
    splitRL(obsStart, obsEnd);
  }
  spCtg->writeCut(cand, this);
  cand->infoGain(this);
}


void CutAccumCtgCart::splitImpl(const SplitNux* cand) {
  if (cutResidual < obsEnd) {
    // Tries obsEnd/obsEnd-1, ..., cut+1/cut.
    // Ordinary R to L, beginning at rank index zero, up to cut.
    splitRL(cutResidual, obsEnd);
    splitResidual(); // Tries cut/resid;
  }
  // Tries resid/cut-1, ..., obsStart+1/obsStart, if applicable.
  // Rightmost observation is residual, with residual rank index.
  // Follow R to L with rank index beginning at current rkIdx;
  if (cutResidual > obsStart) {
    residualRL(cand);
  }
}
