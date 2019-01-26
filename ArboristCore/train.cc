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
#include "forest.h"
#include "rowrank.h"
#include "framemap.h"
#include "index.h"
#include "pretree.h"
#include "samplepred.h"
#include "splitnode.h"
#include "splitcand.h"
#include "leaf.h"
#include "level.h"
#include "coproc.h"

#include <algorithm>


unsigned int Train::trainBlock = 0;

/**
   @brief Registers training tree-block count.

   @param trainBlock_ is the number of trees by which to block.

   @return void.
*/
void Train::initBlock(unsigned int trainBlock_) {
  trainBlock = trainBlock_;
}


/**
   @brief Registers histogram of splitting ranges.

   @return void.
 */
void Train::initCDF(const vector<double> &feSplitQuant) {
  TreeNode::Immutables(feSplitQuant);
}


/**
   @brief Registers per-node probabilities of predictor selection.
 */
void Train::initProb(unsigned int predFixed,
                     const vector<double> &predProb) {
  Level::Immutables(predFixed, predProb);
}


/**
   @brief Registers tree-shape parameters.

   @return void.
 */
void Train::initTree(unsigned int nSamp,
                     unsigned int minNode,
                     unsigned int leafMax) {
  PreTree::Immutables(nSamp, minNode, leafMax);
}


/**
   @brief Registers response-sampling parameters.

   @return void.
 */
void Train::initSample(unsigned int nSamp) {
  Sample::immutables(nSamp);
}

/**
   @brief Registers parameters governing splitting.

   @return void.
 */
void Train::initSplit(unsigned int minNode,
                      unsigned int totLevels,
                      double minRatio) {
  IndexLevel::Immutables(minNode, totLevels);
  SplitCand::immutables(minRatio);
}


/**
   @brief Registers width of categorical response.

   @return void.
 */
void Train::initCtgWidth(unsigned int ctgWidth) {
  SampleNux::immutables(ctgWidth);
}


/**
   @brief Registers monotone specifications for regression.

   @param regMono has length equal to the predictor count.  Only
   numeric predictors may have nonzero entries.
 */
void Train::initMono(const FrameTrain* frameTrain,
                     const vector<double> &regMono) {
  SPReg::Immutables(frameTrain, regMono);
}


/**
   @brief Unsets immutables.

   @return void.
*/
void Train::deInit() {
  trainBlock = 0;
  TreeNode::DeImmutables();
  SplitCand::deImmutables();
  IndexLevel::DeImmutables();
  PreTree::DeImmutables();
  Sample::deImmutables();
  SampleNux::deImmutables();
  Level::DeImmutables();
  SPReg::DeImmutables();
}


/**
   @brief Static entry for regression training.

   @param trainBlock is the maximum number of trees trainable simultaneously.

   @param minNode is the mininal number of sample indices represented by a tree node.

   @param minRatio is the minimum information ratio of a node to its parent.

   @return regression-style object.
*/
unique_ptr<Train> Train::regression(const FrameTrain *frameTrain,
                                       const RankedSet *rankedPair,
                                       const double *y,
                                       const unsigned int *row2Rank,
                                       unsigned int treeChunk) {
  auto trainReg = make_unique<Train>(frameTrain, y, row2Rank, treeChunk);
  trainReg->TrainForest(frameTrain, rankedPair);

  return trainReg;
}


/**
   @brief Regression constructor.
 */
Train::Train(const FrameTrain *frameTrain,
             const double *y,
             const unsigned int *row2Rank,
             unsigned int treeChunk_) :
  nRow(frameTrain->getNRow()),
  treeChunk(treeChunk_),
  bagRow(make_unique<BitMatrix>(treeChunk, nRow)),
  forest(make_unique<ForestTrain>(treeChunk)),
  predInfo(vector<double>(frameTrain->getNPred())),
  leaf(LFTrain::factoryReg(y, row2Rank, treeChunk)) {
}




/**
   @brief Static entry for regression training.

   @return void.
*/
unique_ptr<Train> Train::classification(const FrameTrain *frameTrain,
                                        const RankedSet *rankedPair,
                                        const unsigned int *yCtg,
                                        const double *yProxy,
                                        unsigned int nCtg,
                                        unsigned int treeChunk,
                                        unsigned int nTree) {
  auto trainCtg = make_unique<Train>(frameTrain, yCtg, nCtg, yProxy, nTree, treeChunk);
  trainCtg->TrainForest(frameTrain, rankedPair);

  return trainCtg;
}


/**
   @brief Classification constructor.
 */
Train::Train(const FrameTrain *frameTrain,
             const unsigned int *yCtg,
             unsigned int nCtg,
             const double *yProxy,
             unsigned int nTree,
             unsigned int treeChunk_) :
  nRow(frameTrain->getNRow()),
  treeChunk(treeChunk_),
  bagRow(make_unique<BitMatrix>(treeChunk, nRow)),
  forest(make_unique<ForestTrain>(treeChunk)),
  predInfo(vector<double>(frameTrain->getNPred())),
  leaf(LFTrain::factoryCtg(yCtg, yProxy, treeChunk, nRow, nCtg, nTree)) {
}


Train::~Train() {
}


/**
  @brief Trains the requisite number of trees.

  @param trainBlock is the maximum count of trees to train en banc.

  @return void.
*/
void Train::TrainForest(const FrameTrain *frameTrain,
                        const RankedSet *rankedPair) {
  for (unsigned treeStart = 0; treeStart < treeChunk; treeStart += trainBlock) {
    unsigned int treeEnd = min(treeStart + trainBlock, treeChunk); // one beyond.
    treeBlock(frameTrain, rankedPair->GetRowRank(), treeStart, treeEnd - treeStart);
  }
  forest->splitUpdate(frameTrain, rankedPair->GetNumRanked());
}


/**
   @brief  Creates a block of root samples and trains each one.

   @return void.
 */
void Train::treeBlock(const FrameTrain *frameTrain,
                      const RowRank *rowRank,
                      unsigned int tStart,
                      unsigned int tCount) {
  unsigned int tIdx = tStart;
  vector<TrainSet> block(tCount);
  for (auto & set : block) {
    auto sample = leaf->rootSample(rowRank, bagRow.get(), tIdx++);
    auto preTree = IndexLevel::oneTree(frameTrain, rowRank, sample.get());
    set = make_pair(sample, preTree);
  }

  if (tStart == 0)
    reserve(block);
  blockConsume(block, tStart);
}

 
/** 
  @brief Estimates forest heights using size parameters from the first
  trained block of trees.

  @param ptBlock is a block of PreTree references.

  @return void.
*/
void Train::reserve(vector<TrainSet> &treeBlock) {
  size_t blockFac, blockBag, blockLeaf;
  size_t maxHeight = 0;
  size_t blockHeight = blockPeek(treeBlock, blockFac, blockBag, blockLeaf, maxHeight);
  PreTree::reserve(maxHeight);
}


unsigned int Train::blockPeek(vector<TrainSet> &treeBlock,
                              size_t &blockFac,
                              size_t &blockBag,
                              size_t &blockLeaf,
                              size_t &maxHeight) {
  size_t blockHeight = 0;
  blockLeaf = blockFac = blockBag = 0;
  for (auto & set : treeBlock) {
    get<1>(set)->blockBump(blockHeight, maxHeight, blockFac, blockLeaf, blockBag);
  }

  return blockHeight;
}

 
void Train::blockConsume(vector<TrainSet> &treeBlock,
                         unsigned int blockStart) {
  unsigned int blockIdx = blockStart;
  for (auto & trainSet : treeBlock) {
    const vector<unsigned int> leafMap = get<1>(trainSet)->consume(forest.get(), blockIdx, predInfo);
    leaf->blockLeaves(get<0>(trainSet).get(), leafMap, blockIdx);

    blockIdx++;
  }
}


void Train::getBag(unsigned char *bbRaw) const {
  bagRow->Serialize(bbRaw);
}
