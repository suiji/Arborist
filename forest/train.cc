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
#include "trainframe.h"
#include "frontier.h"
#include "pretree.h"

#include <algorithm>


unsigned int Train::trainBlock = 0;

void Train::initBlock(unsigned int trainBlock_) {
  trainBlock = trainBlock_;
}


void Train::deInit() {
  trainBlock = 0;
}


unique_ptr<Train> Train::train(const TrainFrame* frame,
			       Forest* forest,
			       Sampler* sampler) {
  auto train = make_unique<Train>(frame, forest, sampler);
  train->trainChunk(frame);

  return train;
}


Train::Train(const TrainFrame* frame,
	     Forest* forest_,
	     Sampler* sampler_) :
  predInfo(vector<double>(frame->getNPred())),
  forest(forest_),
  sampler(sampler_) {
}


void Train::trainChunk(const TrainFrame* frame) {
  frame->obsLayout();
  unsigned int treeChunk = sampler->getNTree();
  for (unsigned treeStart = 0; treeStart < treeChunk; treeStart += trainBlock) {
    auto treeBlock = blockProduce(frame, treeStart, min(treeStart + trainBlock, treeChunk));
    blockConsume(treeBlock);
  }
  forest->splitUpdate(frame);
}


vector<unique_ptr<PreTree>> Train::blockProduce(const TrainFrame* frame,
						unsigned int treeStart,
						unsigned int treeEnd) {
  vector<unique_ptr<PreTree>> block;
  for (unsigned int tIdx = treeStart; tIdx < treeEnd; tIdx++) {
    block.emplace_back(move(Frontier::oneTree(frame, sampler)));
  }

  return block;
}

 
void Train::blockConsume(vector<unique_ptr<PreTree>>& treeBlock) {
  for (auto & pretree : treeBlock) {
    const vector<IndexT> leafMap = pretree->consume(forest, predInfo);
    sampler->blockSamples(leafMap);
    forest->setScores(sampler->scoreTree(leafMap));
  }
}
