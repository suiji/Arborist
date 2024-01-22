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


CutAccumRegCart::CutAccumRegCart(const SplitNux& cand,
				 const SFRegCart* spReg) :
  CutAccumReg(cand, spReg) {
  info = (sum * sum) / sCount;
}


void CutAccumRegCart::split(const SFRegCart* spReg,
			    SplitNux& cand) {
  CutAccumRegCart cutAccum(cand, spReg);
  cand.setInfo(cutAccum.splitReg(spReg, cand));
  spReg->writeCut(cand, cutAccum);
}


double CutAccumRegCart::splitReg(const SFRegCart* spReg,
				 const SplitNux& cand) {
  double infoCell = info;
  if (cand.getImplicitCount() != 0) {
    splitImpl();
  }
  else {
    splitRL(obsStart, obsEnd);
  }
  return info - infoCell;
}


void CutAccumRegCart::splitRL(IndexT idxStart, IndexT idxEnd) {
  for (IndexT idx = idxEnd - 1; idx != idxStart; idx--) {
    if (!accumulateReg(obsCell[idx])) {
      argmaxRL(infoVar(), idx-1);
    }
  }
}

void CutAccumRegCart::splitImpl() {
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
    residualRL();
  }
}


void CutAccumRegCart::residualRL() {
  applyResidual(obsCell);
  argmaxResidual(infoVar(), false);
  splitRL(obsStart, cutResidual);
}


void CutAccumRegCart::splitResidual() {
  (void) accumulateReg(obsCell[cutResidual]);
  argmaxResidual(infoVar(), true);
}


void CutAccumCtgCart::splitResidual() {
  (void) accumulateCtg(obsCell[cutResidual]);
  argmaxResidual(infoGini(), true);
}


CutAccumCtgCart::CutAccumCtgCart(const SplitNux& cand,
				 SFCtgCart* spCtg) :
  CutAccumCtg(cand, spCtg) {
  info = ssL / sum;
}


void CutAccumCtgCart::split(SFCtgCart* spCtg,
			    SplitNux& cand) {
  CutAccumCtgCart cutAccum(cand, spCtg);
  cand.setInfo(cutAccum.splitCtg(spCtg, cand));
  spCtg->writeCut(cand, cutAccum);
}


// Initializes from final index and loops over remaining indices.
double CutAccumCtgCart::splitCtg(const SFCtgCart* spCtg,
				 const SplitNux& cand) {
  double infoCell = info;
  if (cand.getImplicitCount() != 0) {
    splitImpl();
  }
  else {
    splitRL(obsStart, obsEnd);
  }
  return info - infoCell;
}


void CutAccumCtgCart::splitRL(IndexT idxStart, IndexT idxEnd) {
  for (IndexT idx = idxEnd - 1; idx != idxStart; idx--) {
    if (!accumulateCtg(obsCell[idx])) {
      argmaxRL(infoGini(), idx-1);
    }
  }
}


void CutAccumCtgCart::splitImpl() {
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
    residualRL();
  }
}


void CutAccumCtgCart::residualRL() {
  applyResidual(obsCell);
  argmaxResidual(infoGini(), false);
  splitRL(obsStart, cutResidual);
}
