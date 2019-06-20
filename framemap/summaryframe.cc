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

#include <algorithm>

#include "summaryframe.h"
#include "rleframe.h"

#include "splitnode.h"
#include "samplepred.h"


SummaryFrame::SummaryFrame(const RLEFrame* rleFrame,
                           double autoCompress,
                           const Coproc* coproc) :
  nRow(rleFrame->nRow),
  nPredNum(rleFrame->nPredNum),
  cardinality(rleFrame->cardinality),
  nPredFac(cardinality.size()),
  cardExtent(nPredFac == 0 ? 0 : *max_element(cardinality.begin(), cardinality.end())),
  nPred(nPredFac + nPredNum),
  rankedFrame(make_unique<RankedFrame>(rleFrame->nRow,
                                       rleFrame->cardinality,
                                       nPred,
                                       rleFrame->row,
                                       rleFrame->rank,
                                       rleFrame->runLength,
                                       rleFrame->rleLength,
                                       autoCompress)),
  numRanked(make_unique<BlockJagged<double> >(rleFrame->numVal,
                                              rleFrame->valOff,
                                              rleFrame->nPredNum)) {
}


/**
   @brief Static entry for sample staging.

   @return SamplePred object for tree.
 */
unique_ptr<SamplePred> SummaryFrame::SamplePredFactory(unsigned int bagCount) const {
  return make_unique<SamplePred>(nPred, bagCount, rankedFrame->safeSize(bagCount));
}


unique_ptr<SPCtg> SummaryFrame::SPCtgFactory(unsigned int bagCount,
                                             unsigned int _nCtg) const {
  return make_unique<SPCtg>(this, bagCount, _nCtg);
}


unique_ptr<SPReg> SummaryFrame::SPRegFactory(unsigned int bagCount) const {
  return make_unique<SPReg>(this, bagCount);
}
