// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file tree.cc

   @brief Methods implementing generic tree nodes.

   @author Mark Seligman
 */


#include "treenode.h"
#include "trainframe.h"
#include "bv.h"
#include "predict.h"


void TreeNode::setQuantRank(const TrainFrame* trainFrame) {
  auto predIdx = getPredIdx();
  if (isNonterminal() && !trainFrame->isFactor(predIdx)) {
    criterion.setQuantRank(trainFrame, predIdx);
  }
}


IndexT TreeNode::advance(const BVJaggedV *facSplit,
                         const IndexT rowT[],
			 unsigned int tIdx,
			 IndexT& leafIdx) const {
  auto predIdx = getPredIdx();
  if (delIdx == 0) {
    leafIdx = predIdx;
    return 0;
  }
  else {
    IndexT bitOff = getBitOffset() + rowT[predIdx];
    return facSplit->testBit(tIdx, bitOff) ? delIdx : delIdx + 1;
  }
}


IndexT TreeNode::advance(const Predict* predict,
                         const BVJaggedV* facSplit,
			 const IndexT* rowFT,
			 const double* rowNT,
			 unsigned int tIdx,
			 IndexT& leafIdx) const {
  auto predIdx = getPredIdx();
  if (delIdx == 0) {
    leafIdx = predIdx;
    return 0;
  }
  else {
    bool isFactor;
    IndexT blockIdx = predict->getIdx(predIdx, isFactor);
    return isFactor ?
      (facSplit->testBit(tIdx, getBitOffset() + rowFT[blockIdx]) ?
       delIdx : delIdx + 1) : (rowNT[blockIdx] <= getSplitNum() ?
                             delIdx : delIdx + 1);
  }
}
