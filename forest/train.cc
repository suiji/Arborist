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
#include "sample.h"
#include "train.h"
#include "trainframe.h"
#include "frontier.h"
#include "pretree.h"
#include "samplercresc.h"

#include <algorithm>


unsigned int Train::trainBlock = 0;

void Train::initBlock(unsigned int trainBlock_) {
  trainBlock = trainBlock_;
}


void Train::deInit() {
  trainBlock = 0;
}


unique_ptr<Train> Train::regression(const TrainFrame* frame,
                                    const vector<double>& y,
                                    unsigned int treeChunk) {
  auto trainReg = make_unique<Train>(frame, y, treeChunk);
  trainReg->trainChunk(frame);

  return trainReg;
}


Train::Train(const TrainFrame* frame,
             const vector<double>& y,
             unsigned int treeChunk_) :
  nRow(frame->getNRow()),
  treeChunk(treeChunk_),
  forest(make_unique<ForestCresc<DecNode> >(treeChunk)),
  predInfo(vector<double>(frame->getNPred())),
  sampler(make_unique<SamplerCresc>(y, treeChunk)) {
}


unique_ptr<Train> Train::classification(const TrainFrame* frame,
                                        const vector<unsigned int>& yCtg,
                                        const vector<double>& yProxy,
                                        unsigned int nCtg,
                                        unsigned int treeChunk,
                                        unsigned int nTree) {
  auto trainCtg = make_unique<Train>(frame, yCtg, nCtg, yProxy, nTree, treeChunk);
  trainCtg->trainChunk(frame);

  return trainCtg;
}


Train::Train(const TrainFrame* frame,
             const vector<unsigned int>& yCtg,
             unsigned int nCtg,
             const vector<double>& classWeight,
             unsigned int nTree,
             unsigned int treeChunk_) :
  nRow(frame->getNRow()),
  treeChunk(treeChunk_),
  forest(make_unique<ForestCresc<DecNode> >(treeChunk)),
  predInfo(vector<double>(frame->getNPred())),
  sampler(make_unique<SamplerCresc>(yCtg, nCtg, classWeight, treeChunk)) {
}


Train::~Train() {
}


void Train::trainChunk(const TrainFrame* frame) {
  frame->obsLayout();
  for (unsigned treeStart = 0; treeStart < treeChunk; treeStart += trainBlock) {
    unsigned int treeEnd = min(treeStart + trainBlock, treeChunk);
    auto treeBlock = blockProduce(frame, treeStart, treeEnd - treeStart);
    blockConsume(treeBlock, treeStart);
  }
  forest->splitUpdate(frame);
}


vector<unique_ptr<PreTree>> Train::blockProduce(const TrainFrame* frame,
                                     unsigned int tStart,
                                     unsigned int tCount) {
  vector<unique_ptr<PreTree>> block;
  for (unsigned int tIdx = 0; tIdx < tCount; tIdx++) {
    sampler->rootSample(frame);
    block.emplace_back(move(Frontier::oneTree(frame, sampler->getSample())));
  }

  if (tStart == 0)
    reserve(block);

  return block;
}

 
void Train::blockConsume(vector<unique_ptr<PreTree>>& treeBlock,
                         unsigned int blockStart) {
  unsigned int blockIdx = blockStart;
  for (auto & pretree : treeBlock) {
    const vector<IndexT> leafMap = pretree->consume(forest.get(), blockIdx, predInfo);
    vector<double> scores = sampler->bagLeaves(leafMap, blockIdx++);
    forest->setScores(scores);
  }
}


void Train::reserve(vector<unique_ptr<PreTree>>& treeBlock) {
  size_t blockFac;
  IndexT blockBag, blockLeaf;
  IndexT maxHeight = 0;
  (void) blockPeek(treeBlock, blockFac, blockBag, blockLeaf, maxHeight);
  PreTree::reserve(maxHeight);
}


unsigned int Train::blockPeek(vector<unique_ptr<PreTree>>& treeBlock,
                              size_t& blockFac,
                              IndexT& blockBag,
                              IndexT& blockLeaf,
                              IndexT& maxHeight) {
  IndexT blockHeight = 0;
  blockLeaf = blockFac = blockBag = 0;
  for (auto & pretree : treeBlock) {
    pretree->blockBump(blockHeight, maxHeight, blockFac, blockLeaf, blockBag);
  }

  return blockHeight;
}


void Train::cacheSamplerRaw(unsigned char blRaw[]) const {
  sampler->dumpRaw(blRaw);
}


const vector<size_t>& Train::getSamplerHeight() const {
  return sampler->getHeight();
}
