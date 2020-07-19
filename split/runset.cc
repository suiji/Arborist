// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file runset.cc

   @brief Methods for maintaining runs of factor-valued predictors during splitting.

   @author Mark Seligman
 */

#include "callback.h"
#include "runaccum.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "runset.h"

IndexT RunNux::noStart = 0;


RunSet::RunSet(SplitStyle factorStyle,
	       PredictorT nCtg_,
	       IndexT nRow) :
  style(factorStyle),
  nCtg(nCtg_) {
  RunNux::noStart = nRow; // Inattainable start value, irrespective of tree.
}


IndexT RunSet::addRun(const SplitFrontier* splitFrontier,
		      const SplitNux* cand,
		      PredictorT rc) {
  runAccum.emplace_back(splitFrontier, cand, nCtg, style, rc);
  return runAccum.size() - 1; // Top position.
}


void RunSet::setOffsets() {
  if (runAccum.empty()) {
    return;
  }

  if (nCtg > 0)
    offsetsCtg();
}


void RunSet::offsetsCtg() {
  IndexT rvRuns = 0;
  for (auto accum : runAccum) {
    rvRuns += accum.countWide();
  }
  if (rvRuns == 0) {
    return;
  }

  // Economizes by pre-allocating random variates for entire frontier.
  rvWide = CallBack::rUnif(rvRuns);
  IndexT rvOff = 0;
  for (auto & accum : runAccum) {
    accum.reWide(rvWide, rvOff);
  }
}


vector<IndexRange> RunSet::getRange(const SplitNux* nux, const CritEncoding& enc) const {
  return runAccum[nux->getAccumIdx()].getRange(enc);
}


IndexRange RunSet::getTopRange(const SplitNux* nux, const CritEncoding& enc) const {
  return runAccum[nux->getAccumIdx()].getTopRange(enc);
}


IndexT RunSet::getImplicitTrue(const SplitNux* nux) const {
  return runAccum[nux->getAccumIdx()].getImplicitTrue();
}


PredictorT RunSet::getRunCount(const SplitNux* nux) const {
  return runAccum[nux->getAccumIdx()].getRunCount();
}


void RunSet::resetRunCount(PredictorT accumIdx,
		      PredictorT runCount) {
  runAccum[accumIdx].resetRunCount(runCount);
}


void RunSet::updateAccum(const SplitNux* cand) {
  runAccum[cand->getAccumIdx()].update(style);
}


vector<PredictorT> RunSet::getTrueBits(const SplitNux& nux) const {
  return runAccum[nux.getAccumIdx()].getTrueBits();
}
