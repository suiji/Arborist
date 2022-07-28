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

#include "prng.h"
#include "runaccum.h"
#include "interlevel.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "runset.h"


RunSet::RunSet(const SplitFrontier* sf,
	       IndexT nRow) :
  style(sf->getFactorStyle()),
  wideRuns(0) {
}


IndexT RunSet::addRun(const SplitFrontier* splitFrontier,
		      const SplitNux* cand) {
  runAccum.emplace_back(splitFrontier, cand, style);
  wideRuns += runAccum.back().countWide();
  return runAccum.size() - 1; // Top position.
}


void RunSet::setOffsets(const SplitFrontier* sf) {
  if (wideRuns == 0 || sf->getNCtg() <= 2)
    return;

  // Economizes by pre-allocating random variates for entire frontier.
  rvWide = PRNG::rUnif(wideRuns);
  IndexT rvOff = 0;
  for (auto & accum : runAccum) {
    accum.reWide(rvWide, rvOff);
  }
}


vector<IndexRange> RunSet::getRange(const SplitNux& nux, const CritEncoding& enc) const {
  return runAccum[nux.getAccumIdx()].getRange(enc);
}


vector<IndexRange> RunSet::getTopRange(const SplitNux& nux, const CritEncoding& enc) const {
  return runAccum[nux.getAccumIdx()].getTopRange(enc);
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


void RunSet::updateAccum(const SplitNux& cand) {
  runAccum[cand.getAccumIdx()].update(cand, style);
}


void RunSet::setTrueBits(const InterLevel* interLevel,
			 const SplitNux& nux,
			 class BV* splitBits,
			 size_t bitPos) const {
  runAccum[nux.getAccumIdx()].setTrueBits(interLevel, nux, splitBits, bitPos);
}


void RunSet::setObservedBits(const InterLevel* interLevel,
			     const SplitNux& nux,
			     class BV* splitBits,
			     size_t bitPos) const {
  runAccum[nux.getAccumIdx()].setObservedBits(interLevel, nux, splitBits, bitPos);
}
