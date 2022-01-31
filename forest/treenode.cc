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


void TreeNode::critCut(const SplitNux* nux,
		       const class SplitFrontier* splitFrontier) {
  setPredIdx(nux->getPredIdx());
  criterion.critCut(nux, splitFrontier);
}


void TreeNode::critBits(const SplitNux* nux,
			size_t bitPos) {
  setPredIdx(nux->getPredIdx());
  criterion.critBits(bitPos);
}
  

void TreeNode::setQuantRank(const TrainFrame* trainFrame) {
  PredictorT predIdx = getPredIdx();
  if (isNonterminal() && !trainFrame->isFactor(predIdx)) {
    criterion.setQuantRank(trainFrame, predIdx);
  }
}


IndexT TreeNode::advance(const vector<unique_ptr<BV>>& factorBits,
			 const IndexT rowT[],
			 unsigned int tIdx) const {
  if (isTerminal()) {
    return 0;
  }
  else {
    IndexT bitOff = getBitOffset() + rowT[getPredIdx()];
    return getDelIdx() + (factorBits[tIdx]->testBit(bitOff) ? 0 : 1);
  }
}


IndexT TreeNode::advance(const Predict* predict,
			 const vector<unique_ptr<BV>>& factorBits,
			 const IndexT* rowFT,
			 const double* rowNT,
			 unsigned int tIdx) const {
  if (isTerminal()) {
    return 0;
  }
  else {
    bool isFactor;
    IndexT blockIdx = predict->getIdx(getPredIdx(), isFactor);
    bool testVal = isFactor ? factorBits[tIdx]->testBit(getBitOffset() + rowFT[blockIdx]) :
			       rowNT[blockIdx] <= getSplitNum();
    return getDelIdx() + (testVal ? 0 : 1);
  }
}
