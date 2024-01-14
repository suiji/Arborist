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


void TreeNode::initMasks(PredictorT nPred) {
  rightBits = 0;
  while ((1ul << ++rightBits) < nPred);
  rightMask = (1ull << rightBits) - 1;
}


void TreeNode::deInit() {
  rightBits = 0;
  rightMask = 0ull;
}


TreeNode::TreeNode(complex<double> pair) :
  packed(abs(pair.real())),
  criterion(pair.imag()),
  invert(pair.real() < 0.0) {
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
