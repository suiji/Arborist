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

#include "accum.h"
#include "frontier.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "summaryframe.h"


double SplitNux::minRatio = minRatioDefault;

vector<double> SplitNux::splitQuant;


void SplitNux::immutables(double minRatio,
			  const vector<double>& feSplitQuant) {
  SplitNux::minRatio = minRatio;
  for (auto quant : feSplitQuant) {
    splitQuant.push_back(quant);
  }
}


void SplitNux::deImmutables() {
  minRatio = minRatioDefault;
  splitQuant.clear();
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
  sum(splitFrontier->getSum(splitCoord)),
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
    lhExtent = splitFrontier->lHBits(setIdx, lhBits, lhSCount);
  }
}


void SplitNux::writeSlots(const SplitFrontier* splitFrontier,
                          PredictorT cutSlot) {
  if (infoGain(splitFrontier)) {
    lhExtent = splitFrontier->lHSlots(setIdx, cutSlot, lhSCount);
  }
}


void SplitNux::writeNum(const SplitFrontier* sf,
			const Accum* accum) {
  info = accum->info;
  if (infoGain(sf)) {
    IndexRange range = IndexRange(accum->rankLH, accum->rankRH - accum->rankLH);
    quantRank = range.interpolate(splitQuant[splitCoord.predIdx]);
    lhSCount = accum->lhSCount;
    lhImplicit = accum->lhImplicit(this);
    lhExtent = lhImplicit + (accum->rhMin - idxRange.getStart());
  }
}


PredictorT SplitNux::getCardinality(const SummaryFrame* frame) const {
  return frame->getCardinality(splitCoord.predIdx);
}


void SplitNux::consume(IndexSet* iSet) const {
  iSet->consumeCriterion(minRatio * info, lhSCount, lhExtent);
}
