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
#include "obsfrontier.h"

CutAccum::CutAccum(const SplitNux* cand,
		   const SplitFrontier* splitFrontier) :
  Accum(splitFrontier, cand),
  cutResidual(obsStart + cand->getPreresidual()),
  rankIdxL(-1),
  rankIdxR(-1) {
}


IndexT CutAccum::lhImplicit(const SplitNux* cand) const {
  IndexT implicitCand = cand->getImplicitCount();
  if (implicitCand == 0) // cutResidual set to 0 otherwise.
    return 0;

  // Residual lies in the left portion of the cut iff its rank is less
  // than the right rank.  This is clearly the case when the residual
  // cut is less than the right observation.  When the residual cut
  // equals the right observation, the residual lies in the left
  // portion iff the residual rank does not bound on the right.
  if (cutResidual < obsRight || (cutResidual == obsRight && rankIdxR != 0)) {
    return implicitCand;
  }
  else {
    return 0;
  }
}


double CutAccum::interpolateRank(const ObsFrontier* ofFront,
				 const SplitNux* cand) const {
  return ofFront->interpolateBackRank(cand, rankIdxL, rankIdxR);
}


CutAccumReg::CutAccumReg(const SplitNux* cand,
			 const SFReg* sfReg) :
  CutAccum(cand, sfReg),
  monoMode(sfReg->getMonoMode(cand)) {
}


CutAccumCtg::CutAccumCtg(const SplitNux* cand,
			 SFCtg* sfCtg) :
  CutAccum(cand, sfCtg),
  nCtg(sfCtg->getNCtg()),
  nodeSum(sfCtg->getSumSlice(cand)),
  ctgAccum(sfCtg->getAccumSlice(cand)),
  ssL(sfCtg->getSumSquares(cand)),
  ssR(0.0) {
}
