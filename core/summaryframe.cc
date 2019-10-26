// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file summaryframe.cc

   @brief Methods associated with ranked frame representations.

   @author Mark Seligman
 */


#include "summaryframe.h"
#include "rleframe.h"
#include "coproc.h"

#include <algorithm>


SummaryFrame::SummaryFrame(const RLEFrame* rleFrame,
			   double autoCompress,
			   bool enableCoproc,
			   vector<string>& diag) : 
  nRow(rleFrame->nRow),
  nPredNum(rleFrame->nPredNum),
  cardinality(rleFrame->cardinality),
  nPredFac(cardinality.size()),
  cardExtent(nPredFac == 0 ? 0 : *max_element(cardinality.begin(), cardinality.end())),
  nPred(nPredFac + nPredNum),
  coproc(Coproc::Factory(enableCoproc, diag)),
  rankedFrame(make_unique<RankedFrame>(rleFrame->nRow,
                                       rleFrame->cardinality,
                                       nPred,
                                       rleFrame->rle,
                                       rleFrame->rleLength,
                                       autoCompress)),
  numRanked(make_unique<BlockJagged<double> >(rleFrame->numVal,
                                              rleFrame->valOff,
                                              rleFrame->nPredNum)) {
}


SummaryFrame::~SummaryFrame() {
}


IndexT SummaryFrame::safeSize(IndexT bagCount) const {
  return rankedFrame->safeSize(bagCount);
}
