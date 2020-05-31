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

#include "cutset.h"
#include "cutaccum.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "summaryframe.h"
#include "branchsense.h"
#include "indexset.h"
#include "defmap.h"


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
		   const DefMap* defMap,
		   PredictorT runCount) :
  splitCoord(preCand.splitCoord),
  idxRange(defMap->adjustRange(preCand, splitFrontier)),
  sum(splitFrontier->getSum(splitCoord)),
  sCount(splitFrontier->getSCount(splitCoord)),
  bufIdx(preCand.bufIdx),
  implicitCount(defMap->getImplicitCount(preCand)),
  ptId(splitFrontier->getPTId(splitCoord)),
  info(splitFrontier->getPrebias(splitCoord)) {
  accumIdx = splitFrontier->addAccumulator(this, runCount);
}


SplitNux::SplitNux(const SplitNux& parent,
		   const class IndexSet* iSet,
		   bool sense,
		   IndexT idx) :
    splitCoord(parent.splitCoord),
    idxRange(parent.idxRange),
    accumIdx(parent.accumIdx),
    sum(iSet->getSumSucc(sense)),
    sCount(iSet->getSCountSucc(sense)),
    bufIdx(parent.bufIdx),
    implicitCount(parent.implicitCount),
    ptId(parent.ptId + idx) {
}


void SplitNux::infoGain(const Accum* accum) {
  info = accum->info - info;
}


IndexRange SplitNux::cutRange(const CutSet* cutSet, bool leftRange) const {
  return leftRange ? cutRangeLeft(cutSet) : cutRangeRight(cutSet);
}


IndexRange SplitNux::cutRangeLeft(const CutSet* cutSet) const {
  return IndexRange(idxRange.getStart(), cutSet->getIdxLeft(this) - idxRange.getStart() + 1);
}


IndexRange SplitNux::cutRangeRight(const CutSet* cutSet) const {
  IndexT idxRight = cutSet->getIdxRight(this);
  return IndexRange(idxRight, idxRange.getExtent() - (idxRight - idxRange.getStart()));
}


bool SplitNux::isFactor(const SummaryFrame* frame) const {
  return frame->isFactor(splitCoord.predIdx);
}


PredictorT SplitNux::getCardinality(const SummaryFrame* frame) const {
  return frame->getCardinality(splitCoord.predIdx);
}
