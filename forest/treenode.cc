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
#include "predictframe.h"
#include "predictorframe.h"
#include "bv.h"
#include "splitnux.h"

unsigned int TreeNode::rightBits = 0;
PredictorT TreeNode::rightMask = 0;
bool TreeNode::trapUnobserved = false;


void TreeNode::initMasks(PredictorT nPred) {
  rightBits = 0;
  while ((1ul << ++rightBits) < nPred);
  rightMask = (1ull << rightBits) - 1;
}


void TreeNode::initTrap(bool doTrap) {
  trapUnobserved = doTrap;
}


void TreeNode::deInit() {
  rightBits = 0;
  rightMask = 0ull;
  trapUnobserved = false;
}


bool TreeNode::trapAndBail() {
  return trapUnobserved;
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


IndexT TreeNode::advanceMixed(const PredictFrame& frame,
			      const vector<BV>& factorBits,
			      const vector<BV>& bitsObserved,
			      const CtgT* rowFT,
			      const double* rowNT,
			      unsigned int tIdx) const {
  bool isFactor;
  IndexT blockIdx = frame.getIdx(getPredIdx(), isFactor);
  if (isFactor) {
    return advanceFactor(factorBits[tIdx], bitsObserved[tIdx], getBitOffset() + rowFT[blockIdx]);
  }
  else {
    return advanceNum(rowNT[blockIdx]);
  }
} // EXIT


IndexT TreeNode::advanceMixed(const PredictFrame& frame,
			      const BV& factorBits,
			      const BV& bitsObserved,
			      const CtgT* rowFT,
			      const double* rowNT) const {
  bool isFactor;
  IndexT blockIdx = frame.getIdx(getPredIdx(), isFactor);
  if (isFactor) {
    return advanceFactor(factorBits, bitsObserved, getBitOffset() + rowFT[blockIdx]);
  }
  else {
    return advanceNum(rowNT[blockIdx]);
  }
}
