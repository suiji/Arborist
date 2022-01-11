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

#include "deffrontier.h" // PreCand
#include "cutset.h"
#include "cutaccum.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "trainframe.h"
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


SplitNux::SplitNux(const PreCand& preCand,
		   const SplitFrontier* splitFrontier) :
  mrra(preCand.mrra),
  randVal(preCand.randVal),
  implicitCount(preCand.stageCount.idxImplicit),
  idxRange(splitFrontier->getRange(mrra)),
  sum(splitFrontier->getSum(mrra)),
  sCount(splitFrontier->getSCount(mrra)),
  ptId(splitFrontier->getPTId(mrra)),
  info(splitFrontier->getPreinfo(mrra)) {
  accumIdx = splitFrontier->addAccumulator(this, preCand);
}


SplitNux::SplitNux(const SplitNux& parent,
		   const SplitFrontier* sf,
		   bool sense,
		   IndexT idx) :
  mrra(parent.mrra),
  randVal(parent.randVal),
  implicitCount(parent.implicitCount),
  idxRange(parent.idxRange),
  accumIdx(parent.accumIdx),
  sum(sf->getSumSucc(mrra, sense)),
  sCount(sf->getSCountSucc(mrra, sense)),
  ptId(parent.ptId + idx),
  info(0.0) {
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


bool SplitNux::isFactor(const SplitFrontier* sf) const {
  return sf->isFactor(mrra.splitCoord.predIdx);
}


PredictorT SplitNux::getCardinality(const TrainFrame* frame) const {
  return frame->getCardinality(mrra.splitCoord.predIdx);
}
