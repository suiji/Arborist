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


SplitNux::SplitNux(const PreCand& preCand_,
		   const SplitFrontier* splitFrontier,
		   const DefMap* defMap,
		   PredictorT runCount) :
  preCand(preCand_),
  idxRange(splitFrontier->getRange(defMap, preCand)),
  sum(splitFrontier->getSum(preCand)),
  sCount(splitFrontier->getSCount(preCand)),
  implicitCount(defMap->getImplicitCount(preCand)),
  ptId(splitFrontier->getPTId(preCand)),
  info(splitFrontier->getPrebias(preCand)) {
  accumIdx = splitFrontier->addAccumulator(this, runCount);
}


SplitNux::SplitNux(const SplitNux& parent,
		   const class IndexSet* iSet,
		   bool sense,
		   IndexT idx) :
  preCand(parent.preCand),
  idxRange(parent.idxRange),
  accumIdx(parent.accumIdx),
  sum(iSet->getSumSucc(sense)),
  sCount(iSet->getSCountSucc(sense)),
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
  return frame->isFactor(preCand.splitCoord.predIdx);
}


PredictorT SplitNux::getCardinality(const SummaryFrame* frame) const {
  return frame->getCardinality(preCand.splitCoord.predIdx);
}
