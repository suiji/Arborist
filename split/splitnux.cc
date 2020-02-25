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
#include "splitfrontier.h"
#include "splitnux.h"
#include "summaryframe.h"
#include "branchsense.h"


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
  idxRange(range),
  setIdx(setIdx_),
  sum(splitFrontier->getSum(splitCoord)),
  sCount(splitFrontier->getSCount(splitCoord)),
  bufIdx(preCand.bufIdx),
  encTrue(true),
  implicitTrue(implicitCount),
  ptId(splitFrontier->getPTId(splitCoord)),
  info(splitFrontier->getPrebias(splitCoord)) {
  enc.init();
}


bool SplitNux::infoGain(const SplitFrontier* splitFrontier) {
  info -= splitFrontier->getPrebias(splitCoord);
  return info > 0.0;
}


void SplitNux::writeBits(const SplitFrontier* splitFrontier,
			 PredictorT lhBits) {
  if (infoGain(splitFrontier)) {
    splitFrontier->lHBits(this, lhBits);
  }
}


void SplitNux::writeSlots(const SplitFrontier* splitFrontier,
                          PredictorT cutSlot) {
  if (infoGain(splitFrontier)) {
    splitFrontier->lHSlots(this, cutSlot);
  }
}


void SplitNux::appendSlot(const SplitFrontier* splitFrontier) {
  if (infoGain(splitFrontier)) {
    splitFrontier->appendSlot(this);
  }
}			  


void SplitNux::writeNum(const SplitFrontier* sf,
			const Accum* accum) {
  info = accum->info;
  if (infoGain(sf)) {
    quantRank = accum->interpolateRank(splitQuant[splitCoord.predIdx]);
    implicitTrue = accum->lhImplicit(this);
    encTrue = implicitTrue == 0;
    cutExtent = implicitTrue + (accum->rhMin - idxRange.getStart());
  }
}


void SplitNux::writeNum(const Accum* accum,
			bool cutLeft,
                        bool encTrue) {
  info = accum->info;
  if (info > 0.0) {
    this->cutLeft = cutLeft;
    this->encTrue = encTrue;
    quantRank = accum->interpolateRank(splitQuant[splitCoord.predIdx]);
    implicitTrue = accum->lhImplicit(this);
    cutExtent = implicitTrue + (accum->rhMin - idxRange.getStart());
  }
}


void SplitNux::writeFac(IndexT sCountTrue,
			IndexT cutExtent,
			IndexT implicitTrue) {
  this->cutExtent = cutExtent,
  this->implicitTrue = implicitTrue;
  encTrue = implicitTrue == 0;
}
			 

PredictorT SplitNux::getCardinality(const SummaryFrame* frame) const {
  return frame->getCardinality(splitCoord.predIdx);
}
