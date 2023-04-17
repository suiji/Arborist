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

#include "cutfrontier.h"
#include "runsig.h"
#include "splitfrontier.h"
#include "splitnux.h"


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


SplitNux::SplitNux(const StagedCell* cell_,
		   double randVal_,
		   const SplitFrontier* splitFrontier) :
  cell(cell_),
  randVal(randVal_),
  sum(splitFrontier->getSum(cell)),
  sCount(splitFrontier->getSCount(cell)),
  ptId(splitFrontier->getPTId(cell)),
  info(0.0) {
  sigIdx = splitFrontier->accumulatorIndex(*this);
}


SplitNux::SplitNux(const SplitNux& parent,
		   const SplitFrontier* sf,
		   bool sense,
		   IndexT idx) :
  cell(parent.cell),
  randVal(parent.randVal),
  sigIdx(parent.sigIdx),
  sum(sf->getSumSucc(cell, sense)),
  sCount(sf->getSCountSucc(cell, sense)),
  ptId(parent.ptId + idx),
  info(0.0) {
}


bool SplitNux::isImplicit(const RunNux& nux) const {
  return nux.obsRange.idxStart >= getObsEnd();
}



IndexRange SplitNux::cutRange(const CutSet* cutSet, bool leftRange) const {
  return leftRange ? cutRangeLeft(cutSet) : cutRangeRight(cutSet);
}


IndexRange SplitNux::cutRangeLeft(const CutSet* cutSet) const {
  return IndexRange(getObsStart(), cutSet->getIdxLeft(*this) - getObsStart() + 1);
}


IndexRange SplitNux::cutRangeRight(const CutSet* cutSet) const {
  IndexT idxRight = cutSet->getIdxRight(*this);
  return IndexRange(idxRight, getObsExtent() - (idxRight - getObsStart()));
}
