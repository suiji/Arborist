// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file grove.cc

   @brief Main entry from front end for training.

   @author Mark Seligman
*/

#include "bv.h"
#include "grove.h"
#include "predictorframe.h"
#include "frontier.h"
#include "pretree.h"
#include "leaf.h"
#include "sampler.h"
#include "nodescorer.h"

#include <algorithm>

unsigned int Grove::trainBlock = 0;


void Grove::initBlock(unsigned int trainBlock_) {
  trainBlock = trainBlock_;
}


void Grove::deInit() {
  trainBlock = 0;
}


Grove::Grove(const PredictorFrame* frame,
	     const Sampler* sampler,
	     Forest* forest_,
	     unique_ptr<NodeScorer> nodeScorer_) :
  predInfo(vector<double>(frame->getNPred())),
  forest(forest_),
  nodeScorer(move(nodeScorer_)) {
}


void Grove::train(const PredictorFrame* frame,
		  const Sampler * sampler,
		  const IndexRange& treeRange,
		  Leaf* leaf) {
  for (unsigned treeStart = treeRange.getStart(); treeStart < treeRange.getEnd(); treeStart += trainBlock) {
    auto treeBlock = blockProduce(frame, sampler, treeStart, min(treeStart + trainBlock, static_cast<unsigned int>(treeRange.getEnd())));
    blockConsume(treeBlock, leaf);
  }
  forest->splitUpdate(frame);
}


vector<unique_ptr<PreTree>> Grove::blockProduce(const PredictorFrame* frame,
						const Sampler* sampler,
						unsigned int treeStart,
						unsigned int treeEnd) {
  vector<unique_ptr<PreTree>> block;
  for (unsigned int tIdx = treeStart; tIdx < treeEnd; tIdx++) {
    block.emplace_back(Frontier::oneTree(frame, this, sampler, tIdx));
  }

  return block;
}

 
void Grove::blockConsume(const vector<unique_ptr<PreTree>>& treeBlock,
			 Leaf* leaf) {
  for (auto & pretree : treeBlock) {
    pretree->consume(this, forest, leaf);
  }
}


void Grove::consumeInfo(const vector<double>& info) {
  for (IndexT predIdx = 0; predIdx < predInfo.size(); predIdx++) {
    predInfo[predIdx] += info[predIdx];
  }
}
