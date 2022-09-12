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


RunSet::RunSet(const SplitFrontier* sf) :
  nAccum(0),
  style(sf->getFactorStyle()) {
}


IndexT RunSet::preIndex(const SplitFrontier*sf, const SplitNux& cand) {
  if (RunAccum::ctgWide(sf, cand))
    runWide.push_back(nAccum);

  return nAccum++;
}


void RunSet::accumPreset(const SplitFrontier* sf) {
  runSig = vector<RunSig>(nAccum);
  if (!runWide.empty())
    rvWide = PRNG::rUnif(RunAccum::maxWidth * runWide.size());
}


void RunSet::setToken(const SplitNux& nux,
		      PredictorT token) {
  runSig[nux.getAccumIdx()].splitToken = token;
}


void RunSet::setRuns(const SplitNux& cand,
		     vector<RunNux> runNux) {
  runSig[cand.getAccumIdx()].runNux = move(runNux);
}


const vector<RunNux>& RunSet::getRunNux(const SplitNux& nux) const {
  return runSig[nux.getAccumIdx()].runNux;
}


vector<IndexRange> RunSet::getRunRange(const SplitNux& nux, const CritEncoding& enc) const {
  return runSig[nux.getAccumIdx()].getRange(enc);
}


vector<IndexRange> RunSet::getTopRange(const SplitNux& nux, const CritEncoding& enc) const {
  return runSig[nux.getAccumIdx()].getTopRange(enc);
}


IndexT RunSet::getImplicitTrue(const SplitNux& nux) const {
  return runSig[nux.getAccumIdx()].getImplicitTrue();
}


PredictorT RunSet::getRunCount(const SplitNux* nux) const {
  return runSig[nux->getAccumIdx()].getRunCount();
}


void RunSet::resetRunSup(PredictorT accumIdx,
			 PredictorT runCount) {
  runSig[accumIdx].resetRunSup(runCount);
}


void RunSet::accumUpdate(const SplitNux& cand) {
  runSig[cand.getAccumIdx()].updateCriterion(cand, style);
}


void RunSet::setTrueBits(const InterLevel* interLevel,
			 const SplitNux& nux,
			 class BV* splitBits,
			 size_t bitPos) const {
  runSig[nux.getAccumIdx()].setTrueBits(interLevel, nux, splitBits, bitPos);
}


void RunSet::setObservedBits(const InterLevel* interLevel,
			     const SplitNux& nux,
			     class BV* splitBits,
			     size_t bitPos) const {
  runSig[nux.getAccumIdx()].setObservedBits(interLevel, nux, splitBits, bitPos);
}


vector<IndexRange> RunSet::getRange(const SplitNux& nux,
				    const CritEncoding& enc) const {
  if (style == SplitStyle::topSlot) {
    return getTopRange(nux, enc);
  }
  else {
    return getRunRange(nux, enc);
  }
}


