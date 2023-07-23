// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file train.cc

   @brief Main entry from front end for training.

   @author Mark Seligman
*/

#include "bv.h"
#include "train.h"
#include "predictorframe.h"
#include "frontier.h"
#include "pretree.h"
#include "leaf.h"
#include "sampler.h"
#include "sampledobs.h"

#include <algorithm>


unsigned int Train::trainBlock = 0;
unique_ptr<SampledObs> Train::sequentialObs = nullptr;


void Train::initBlock(unsigned int trainBlock_) {
  trainBlock = trainBlock_;
  sequentialObs = nullptr;
}


void Train::deInit() {
  trainBlock = 0;
  sequentialObs = nullptr;
}


unique_ptr<Train> Train::train(const PredictorFrame* frame,
			       const Sampler* sampler,
			       Forest* forest,
			       const IndexRange& treeRange,
			       Leaf* leaf) {
  auto train = make_unique<Train>(frame, sampler, forest);
  train->trainChunk(frame, sampler, treeRange, leaf);
  forest->splitUpdate(frame);

  return train;
}


Train::Train(const PredictorFrame* frame,
	     const Sampler* sampler,
	     Forest* forest_) :
  predInfo(vector<double>(frame->getNPred())),
  forest(forest_),
  nu(forest->getNu()) {
  if (sequential() && sequentialObs == nullptr) {
    sequentialObs = sampler->obsFactory(this, 0);
  }
}


void Train::trainChunk(const PredictorFrame* frame,
		       const Sampler * sampler,
		       const IndexRange& treeRange,
		       Leaf* leaf) {
  for (unsigned treeStart = treeRange.getStart(); treeStart < treeRange.getEnd(); treeStart += trainBlock) {
    auto treeBlock = blockProduce(frame, sampler, treeStart, min(treeStart + trainBlock, static_cast<unsigned int>(treeRange.getEnd())));
    blockConsume(treeBlock, leaf);
  }
}


vector<unique_ptr<PreTree>> Train::blockProduce(const PredictorFrame* frame,
						const Sampler* sampler,
						unsigned int treeStart,
						unsigned int treeEnd) {
  vector<unique_ptr<PreTree>> block;
  for (unsigned int tIdx = treeStart; tIdx < treeEnd; tIdx++) {
    block.emplace_back(Frontier::oneTree(frame, sequential() ? sequentialObs.get() : sampler->obsFactory(this, tIdx).get()));
  }

  return block;
}

 
void Train::blockConsume(const vector<unique_ptr<PreTree>>& treeBlock,
			 Leaf* leaf) {
  for (auto & pretree : treeBlock) {
    pretree->consume(this, forest, leaf);
  }
}


void Train::consumeInfo(const vector<double>& info) {
  for (IndexT predIdx = 0; predIdx < predInfo.size(); predIdx++) {
    predInfo[predIdx] += info[predIdx];
  }
}
