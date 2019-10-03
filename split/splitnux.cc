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


SplitNux::SplitNux(const DefCoord& preCand,
		   const SplitFrontier* splitFrontier,
		   PredictorT setIdx_,
		   IndexRange range,
		   IndexT implicitCount) :
  splitCoord(preCand.splitCoord),
  bufIdx(preCand.bufIdx),
  idxRange(range),
  setIdx(setIdx_),
  lhSum(splitFrontier->getSum(splitCoord)),
  lhSCount(splitFrontier->getSCount(splitCoord)),
  info(splitFrontier->getPrebias(splitCoord)),
  lhImplicit(implicitCount) {
}

  
bool SplitNux::infoGain(const SplitFrontier* splitFrontier) {
  info -= splitFrontier->getPrebias(splitCoord);
  return info > 0.0;
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


void SplitNux::writeSlots(const SplitFrontier* splitFrontier,
                          RunSet* runSet,
                          PredictorT cutSlot) {
  if (infoGain(splitFrontier)) {
    lhExtent = runSet->lHSlots(cutSlot, lhSCount);
  }
}
