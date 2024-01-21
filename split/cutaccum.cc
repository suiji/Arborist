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
#include "cutfrontier.h"
#include "splitfrontier.h"
#include "partition.h"
#include "interlevel.h"


CutAccum::CutAccum(const SplitNux& cand,
		   const SplitFrontier* splitFrontier) :
  Accum(splitFrontier, cand),
  obsLeft(-1),
  obsRight(-1),
  residualLeft(false) {
}


IndexT CutAccum::lhImplicit(const SplitNux& cand) const {
  IndexT implicitCand = cand.getImplicitCount();
  if (implicitCand == 0) // cutResidual set to 0 otherwise.
    return 0;

  // Residual lies in the left portion of the cut iff its rank is less
  // than the right rank.  This is clearly the case when the residual
  // cut is less than the right observation.  When the residual cut
  // equals the right observation, the residual lies in the left
  // portion iff the residual does not bound on the right.
  if (cutResidual < obsRight || (cutResidual == obsRight && residualLeft)) {
    return implicitCand;
  }
  else {
    return 0;
  }
}


double CutAccum::interpolateRank(const InterLevel* interLevel,
				 const SplitNux& cand) const {
  if (obsRight == cutResidual) { // iff splitting residual on R/L.
    return interLevel->interpolateRank(cand, residualLeft ? obsRight : obsLeft, residualLeft);
  }
  else {
    return interLevel->interpolateRank(cand, obsLeft, obsRight);
  }
}


CutAccumReg::CutAccumReg(const SplitNux& cand,
			 const SFReg* sfReg) :
  CutAccum(cand, sfReg),
  monoMode(sfReg->getMonoMode(cand)) {
}


void CutAccum::residualReg(const Obs* obsCell) {
  double ySumObs = 0.0;
  IndexT sCountObs = 0;
  for (IndexT obsIdx = obsStart; obsIdx != obsEnd; obsIdx++) {
    const Obs& obs = obsCell[obsIdx];
    ySumObs += obs.getYSum();
    sCountObs += obs.getSCount();
  }

  sum -= (sumCount.sum - ySumObs);
  sCount -= (sumCount.sCount - sCountObs);
}


CutAccumCtg::CutAccumCtg(const SplitNux& cand,
			 SFCtg* sfCtg) :
  CutAccum(cand, sfCtg),
  ctgNux(filterMissingCtg(sfCtg, cand)),
  ctgAccum(vector<double>(ctgNux.nCtg())),
  ssL(ctgNux.sumSquares),
  ssR(0.0) {
}


void CutAccumCtg::residualCtg(const Obs* obsCell) {
  vector<double> ctgResid(ctgNux.ctgSum);
  for (PredictorT ctg = 0; ctg != ctgResid.size(); ctg++) {
    accumCtgSS(ctgResid[ctg], ctg);
  }

  double ySumExpl = 0.0;
  IndexT sCountExpl = 0;
  for (IndexT obsIdx = obsStart; obsIdx != obsEnd; obsIdx++) {
    const Obs& obs = obsCell[obsIdx];
    double ySumObs = obs.getYSum();
    ctgResid[obs.getCtg()] -= ySumObs;
    ySumExpl += ySumObs;
    sCountExpl += obs.getSCount();
  }
  sum -= (sumCount.sum - ySumExpl);
  sCount -= (sumCount.sCount - sCountExpl);
}


