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


void RunSet::addRun(unique_ptr<RunAccum> upt,
		    const SplitNux& cand) {
  runAccum[cand.getAccumIdx()] = move(upt);
}


void RunSet::accumPreset(const SplitFrontier* sf) {
  runNux = vector<vector<RunNux>>(nAccum);
  runAccum = vector<unique_ptr<RunAccum>>(nAccum);
  if (!runWide.empty())
    rvWide = PRNG::rUnif(RunAccum::maxWidth * runWide.size());
}


vector<IndexRange> RunSet::getRunRange(const SplitNux& nux, const CritEncoding& enc) const {
  return runAccum[nux.getAccumIdx()]->getRange(enc);
}


vector<IndexRange> RunSet::getTopRange(const SplitNux& nux, const CritEncoding& enc) const {
  return runAccum[nux.getAccumIdx()]->getTopRange(enc);
}


IndexT RunSet::getImplicitTrue(const SplitNux& nux) const {
  return runAccum[nux.getAccumIdx()]->getImplicitTrue();
}


PredictorT RunSet::getRunCount(const SplitNux* nux) const {
  return runAccum[nux->getAccumIdx()]->getRunCount();
}


void RunSet::resetRunSup(PredictorT accumIdx,
			 PredictorT runCount) {
  runAccum[accumIdx]->resetRunSup(runCount);
}


void RunSet::updateAccum(const SplitNux& cand) {
  runAccum[cand.getAccumIdx()]->update(cand, style);
}


void RunSet::setTrueBits(const InterLevel* interLevel,
			 const SplitNux& nux,
			 class BV* splitBits,
			 size_t bitPos) const {
  runAccum[nux.getAccumIdx()]->setTrueBits(interLevel, nux, splitBits, bitPos);
}


void RunSet::setObservedBits(const InterLevel* interLevel,
			     const SplitNux& nux,
			     class BV* splitBits,
			     size_t bitPos) const {
  runAccum[nux.getAccumIdx()]->setObservedBits(interLevel, nux, splitBits, bitPos);
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


