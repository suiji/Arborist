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
#include "predictorframe.h"
#include "bv.h"
#include "predict.h"
#include "splitnux.h"

unsigned int TreeNode::rightBits = 0;
PredictorT TreeNode::rightMask = 0;

void TreeNode::init(PredictorT nPred) {
  rightBits = 0;
  while ((1ul << ++rightBits) < nPred);
  rightMask = (1ull << rightBits) - 1;
}


void TreeNode::deInit() {
  rightBits = 0;
  rightMask = 0ull;
}


void TreeNode::critCut(const SplitNux& nux,
		       const class SplitFrontier* splitFrontier) {
  setPredIdx(nux.getPredIdx());
  criterion.critCut(nux, splitFrontier);
}


void TreeNode::critBits(const SplitNux& nux,
			size_t bitPos) {
  setPredIdx(nux.getPredIdx());
  criterion.critBits(bitPos);
}
  

void TreeNode::setQuantRank(const PredictorFrame* frame) {
  PredictorT predIdx = getPredIdx();
  if (isNonterminal() && !frame->isFactor(predIdx)) {
    criterion.setQuantRank(frame, predIdx);
  }
}


IndexT TreeNode::advanceMixed(const Predict* predict,
			      const vector<unique_ptr<BV>>& factorBits,
			      const vector<unique_ptr<BV>>& bitsObserved,
			      const CtgT* rowFT,
			      const double* rowNT,
			      unsigned int tIdx,
			      bool trapUnobserved) const {
  bool isFactor;
  IndexT blockIdx = predict->getIdx(getPredIdx(), isFactor);
  if (isFactor) {
    return advanceFactor(factorBits[tIdx].get(), bitsObserved[tIdx].get(), getBitOffset() + rowFT[blockIdx], trapUnobserved);
  }
  else {
    return advanceNum(rowNT[blockIdx], trapUnobserved);
  }
}
