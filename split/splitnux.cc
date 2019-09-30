// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitnux.cc

   @brief Methods belonging to the minimal splitting representation.

   @author Mark Seligman
 */

#include "runset.h"
#include "frontier.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "summaryframe.h"


double SplitNux::minRatio = minRatioDefault;

void SplitNux::immutables(double minRatio) {
  SplitNux::minRatio = minRatio;
}

void SplitNux::deImmutables() {
  minRatio = minRatioDefault;
}


SplitNux::SplitNux(const class SplitFrontier* splitFrontier,
		   const class Frontier* frontier,
		   const SplitCoord splitCoord_,
		   unsigned char bufIdx_,
		   IndexT noSet) :
  splitCoord(splitCoord_),
  setIdx(noSet),
  bufIdx(bufIdx_),
  lhSum(frontier->getSum(splitCoord.nodeIdx)),
  lhSCount(frontier->getSCount(splitCoord.nodeIdx)),
  info(splitFrontier->getPrebias(splitCoord))  {
}
  
  
bool SplitNux::infoGain(const SplitFrontier* splitFrontier) {
  info -= splitFrontier->getPrebias(splitCoord);
  return info > 0.0;
}


void
SplitNux::schedule(PredictorT rCount,
		   vector<PredictorT>& runCount,
		   IndexRange range,
		   IndexT implicitCount) {
  if (rCount > 1) {
    setSetIdx(runCount.size());
    runCount.push_back(rCount);
  }
  setIndexRange(range);
  setImplicit(implicitCount);
}


void SplitNux::writeBits(const SplitFrontier* splitFrontier,
			 PredictorT lhBits) {
  if (infoGain(splitFrontier)) {
    RunSet* runSet = splitFrontier->rSet(setIdx);
    lhExtent = runSet->lHBits(lhBits, lhSCount);
  }
}


void SplitNux::writeNum(const SplitFrontier* splitFrontier,
			double info,
			IndexT rankLH,
			IndexT rankRH,
			IndexT lhSCount,
			IndexT lhImplicit,
			IndexT rhMin) {
  this->info = info;
  if (infoGain(splitFrontier)) {
    rankRange.set(rankLH, rankRH - rankLH);
    this->lhSCount = lhSCount;
    this->lhImplicit = lhImplicit;
    lhExtent = lhImplicit + (rhMin - idxRange.getStart());
  }
}


PredictorT SplitNux::getCardinality(const SummaryFrame* frame) const {
  return frame->getCardinality(splitCoord.predIdx);
}


void SplitNux::consume(IndexSet* iSet) const {
  iSet->consumeCriterion(minRatio * info, lhSCount, lhExtent);
}


  /**
     @brief Writes the left-hand characterization of a factor-based
     split with numerical or binary response.

     @param runSet organizes responsed statistics by factor code.

     @param cutSlot is the LHS/RHS separator position in the vector of
     factor codes maintained by the run-set.
   */
void SplitNux::writeSlots(const SplitFrontier* splitFrontier,
                          RunSet* runSet,
                          PredictorT cutSlot) {
  if (infoGain(splitFrontier)) {
    lhExtent = runSet->lHSlots(cutSlot, lhSCount);
  }
}
