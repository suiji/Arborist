// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file cartnode.cc

   @brief Methods implementing CART tree nodes.

   @author Mark Seligman
 */


#include "cartnode.h"
#include "summaryframe.h"
#include "bv.h"
#include "predict.h"


void CartNode::setQuantRank(const SummaryFrame* sf) {
  auto predIdx = getPredIdx();
  if (isNonterminal() && !sf->isFactor(predIdx)) {
    criterion.setQuantRank(sf, predIdx);
  }
}


IndexT CartNode::advance(const BVJagged *facSplit,
                         const IndexT rowT[],
			 unsigned int tIdx,
			 IndexT& leafIdx) const {
  auto predIdx = getPredIdx();
  if (lhDel == 0) {
    leafIdx = predIdx;
    return 0;
  }
  else {
    IndexT bitOff = getSplitBit() + rowT[predIdx];
    return facSplit->testBit(tIdx, bitOff) ? lhDel : lhDel + 1;
  }
}


IndexT CartNode::advance(const PredictFrame* blockFrame,
                         const BVJagged* facSplit,
			 const IndexT* rowFT,
			 const double* rowNT,
			 unsigned int tIdx,
			 IndexT& leafIdx) const {
  auto predIdx = getPredIdx();
  if (lhDel == 0) {
    leafIdx = predIdx;
    return 0;
  }
  else {
    bool isFactor;
    IndexT blockIdx = blockFrame->getIdx(predIdx, isFactor);
    return isFactor ?
      (facSplit->testBit(tIdx, getSplitBit() + rowFT[blockIdx]) ?
       lhDel : lhDel + 1) : (rowNT[blockIdx] <= getSplitNum() ?
                             lhDel : lhDel + 1);
  }
}
