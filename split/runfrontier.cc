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
#include "runfrontier.h"


RunSet::RunSet(const SplitFrontier* sf) :
  nAccum(0),
  style(sf->getFactorStyle()) {
}


IndexT RunSet::preIndex(const SplitFrontier*sf, const SplitNux& cand) {
  if (RunAccum::ctgWide(sf, cand))
    runWide.push_back(nAccum);

  return nAccum++;
}


const double* RunSet::rvSlice(IndexT sigIdx) const {
  return &rvWide[RunAccum::maxWidth * (lower_bound(runWide.begin(), runWide.end(), sigIdx) - runWide.begin())];
}

  
void RunSet::accumPreset(const SplitFrontier* sf) {
  runSig = vector<RunSig>(nAccum);
  if (!runWide.empty())
    rvWide = PRNG::rUnif(RunAccum::maxWidth * runWide.size());
}


void RunSet::setSplit(SplitNux& nux,
		      vector<RunNux> runNux,
		      const SplitRun& splitRun) {
  nux.setInfo(splitRun.gain);
  runSig[nux.getSigIdx()] = RunSig(std::move(runNux), splitRun.token, splitRun.runsSampled);
}


const vector<RunNux>& RunSet::getRunNux(const SplitNux& nux) const {
  return runSig[nux.getSigIdx()].runNux;
}


vector<IndexRange> RunSet::getRunRange(const SplitNux& nux, const CritEncoding& enc) const {
  return runSig[nux.getSigIdx()].getRange(enc);
}


vector<IndexRange> RunSet::getTopRange(const SplitNux& nux, const CritEncoding& enc) const {
  return runSig[nux.getSigIdx()].getTopRange(enc);
}


IndexT RunSet::getImplicitTrue(const SplitNux& nux) const {
  return runSig[nux.getSigIdx()].getImplicitTrue();
}


PredictorT RunSet::getRunCount(const SplitNux* nux) const {
  return runSig[nux->getSigIdx()].getRunCount();
}


void RunSet::resetRunSup(PredictorT sigIdx,
			 PredictorT runCount) {
  runSig[sigIdx].resetRunSup(runCount);
}


void RunSet::accumUpdate(const SplitNux& cand) {
  runSig[cand.getSigIdx()].updateCriterion(cand, style);
}


void RunSet::setTrueBits(const InterLevel* interLevel,
			 const SplitNux& nux,
			 class BV* splitBits,
			 size_t bitPos) const {
  runSig[nux.getSigIdx()].setTrueBits(interLevel, nux, splitBits, bitPos);
}


void RunSet::setObservedBits(const InterLevel* interLevel,
			     const SplitNux& nux,
			     class BV* splitBits,
			     size_t bitPos) const {
  runSig[nux.getSigIdx()].setObservedBits(interLevel, nux, splitBits, bitPos);
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


