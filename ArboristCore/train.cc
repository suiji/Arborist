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
#include "response.h"
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
void Train::InitBlock(unsigned int trainBlock_) {
  trainBlock = trainBlock_;
}


/**
   @brief Registers preference for thin leaves.
 */
void Train::InitLeaf(bool thinLeaves) {
  LeafTrain::Immutables(thinLeaves);
}


/**
   @brief Registers histogram of splitting ranges.

   @return void.
 */
void Train::InitCDF(const vector<double> &feSplitQuant) {
  ForestNode::Immutables(feSplitQuant);
}


/**
   @brief Registers per-node probabilities of predictor selection.
 */
void Train::InitProb(unsigned int predFixed,
                     const vector<double> &predProb) {
  Level::Immutables(predFixed, predProb);
}


/**
   @brief Registers tree-shape parameters.

   @return void.
 */
void Train::InitTree(unsigned int nSamp,
                     unsigned int minNode,
                     unsigned int leafMax) {
  PreTree::Immutables(nSamp, minNode, leafMax);
}


/**
   @brief Registers response-sampling parameters.

   @return void.
 */
void Train::InitSample(unsigned int nSamp) {
  Sample::Immutables(nSamp);
}

/**
   @brief Registers parameters governing splitting.

   @return void.
 */
void Train::InitSplit(unsigned int minNode,
                      unsigned int totLevels,
                      double minRatio) {
  IndexLevel::Immutables(minNode, totLevels);
  SplitCand::immutables(minRatio);
}


/**
   @brief Registers width of categorical response.

   @return void.
 */
void Train::InitCtgWidth(unsigned int ctgWidth) {
  SampleNux::immutables(ctgWidth);
}


/**
   @brief Registers monotone specifications for regression.
 */
void Train::InitMono(const vector<double> &regMono) {
  SPReg::Immutables(regMono);
}


/**
   @brief Unsets immutables.

   @return void.
*/
void Train::DeInit() {
  trainBlock = 0;
  ForestNode::DeImmutables();
  SplitCand::deImmutables();
  IndexLevel::DeImmutables();
  LeafTrain::DeImmutables();
  PreTree::DeImmutables();
  Sample::DeImmutables();
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
unique_ptr<TrainReg> Train::Regression(const FrameTrain *frameTrain,
                                       const RankedSet *rankedPair,
                                       const double *y,
                                       const unsigned int *row2Rank,
                                       unsigned int treeChunk) {
  auto trainReg = make_unique<TrainReg>(frameTrain, y, row2Rank, treeChunk);
  trainReg->TrainForest(frameTrain, rankedPair);

  return trainReg;
}


TrainReg::TrainReg(const class FrameTrain *frameTrain,
                   const double *y,
                   const unsigned int *row2Rank,
                   unsigned int treeChunk) :
  Train(frameTrain, y, row2Rank, treeChunk),
  leafReg(make_unique<LeafTrainReg>(treeChunk)) {
}


TrainReg::~TrainReg() {
}


/**
   @brief Regression constructor.
 */
Train::Train(const FrameTrain *frameTrain,
             const double *y,
             const unsigned int *row2Rank,
             unsigned int treeChunk_) :
  nRow(frameTrain->NRow()),
  treeChunk(treeChunk_),
  bagRow(make_unique<BitMatrix>(treeChunk, nRow)),
  forest(make_unique<ForestTrain>(treeChunk)),
  predInfo(vector<double>(frameTrain->NPred())),
  response(Response::FactoryReg(y, row2Rank)) {
}




/**
   @brief Static entry for regression training.

   @return void.
*/
unique_ptr<TrainCtg> Train::Classification(const FrameTrain *frameTrain,
                                           const RankedSet *rankedPair,
                                           const unsigned int *yCtg,
                                           const double *yProxy,
                                           unsigned int nCtg,
                                           unsigned int treeChunk) {
  auto trainCtg = make_unique<TrainCtg>(frameTrain, yCtg, yProxy, nCtg, treeChunk);
  trainCtg->TrainForest(frameTrain, rankedPair);

  return trainCtg;
}


TrainCtg::TrainCtg(const class FrameTrain *frameTrain,
                   const unsigned int *yCtg,
                   const double *yProxy,
                   unsigned int nCtg,
                   unsigned int treeChunk) :
  Train(frameTrain, yCtg, yProxy, treeChunk),
  leafCtg(make_unique<LeafTrainCtg>(treeChunk, NRow(), nCtg)) {
}


TrainCtg::~TrainCtg() {
}


/**
   @brief Classification constructor.
 */
Train::Train(const FrameTrain *frameTrain,
             const unsigned int *yCtg,
             const double *yProxy,
             unsigned int treeChunk_) :
  nRow(frameTrain->NRow()),
  treeChunk(treeChunk_),
  bagRow(make_unique<BitMatrix>(treeChunk, nRow)),
  forest(make_unique<ForestTrain>(treeChunk)),
  predInfo(vector<double>(frameTrain->NPred())),
  response(Response::FactoryCtg(yCtg, yProxy)) {
}


Train::~Train() {
}


/**
  @brief Trains the requisite number of trees.

  @param trainBlock is the maximum Count of trees to train en block.

  @return void.
*/
void Train::TrainForest(const FrameTrain *frameTrain,
                        const RankedSet *rankedPair) {
  for (unsigned treeStart = 0; treeStart < treeChunk; treeStart += trainBlock) {
    unsigned int treeEnd = min(treeStart + trainBlock, treeChunk); // one beyond.
    TreeBlock(frameTrain, rankedPair->GetRowRank(), treeStart, treeEnd - treeStart);
  }
  forest->SplitUpdate(frameTrain, rankedPair->GetNumRanked());
}


/**
   @brief  Creates a block of root samples and trains each one.

   @return void.
 */
void Train::TreeBlock(const FrameTrain *frameTrain,
                      const RowRank *rowRank,
                      unsigned int tStart,
                      unsigned int tCount) {
  unsigned int tIdx = tStart;
  vector<TrainPair> treeBlock(tCount);
  for (auto & pair : treeBlock) {
    auto treeBag = bagRow->BVRow(tIdx++);
    auto sample = response->RootSample(rowRank, treeBag.get());
    auto preTree = IndexLevel::oneTree(frameTrain, sample, rowRank);
    pair = make_pair(sample, preTree);
  }

  if (tStart == 0)
    Reserve(treeBlock);
  BlockConsume(treeBlock, tStart);
}

 
/** 
  @brief Estimates forest heights using size parameters from the first
  trained block of trees.

  @param ptBlock is a block of PreTree references.

  @return void.
*/
void Train::Reserve(vector<TrainPair> &treeBlock) {
  unsigned int blockFac, blockBag, blockLeaf;
  unsigned int maxHeight = 0;
  unsigned int blockHeight = BlockPeek(treeBlock, blockFac, blockBag, blockLeaf, maxHeight);
  PreTree::Reserve(maxHeight);

  double slop = (slopFactor * treeChunk) / trainBlock;
  forest->Reserve(blockHeight, blockFac, slop);
  Reserve(slop * blockLeaf, slop * blockBag);
}


/**
   @brief Accumulates block size parameters as clues to forest-wide sizes.
   Estimates improve with larger blocks, at the cost of higher memory footprint.

   @return sum of tree sizes over block.
 */
unsigned int Train::BlockPeek(vector<TrainPair> &treeBlock,
                              unsigned int &blockFac,
                              unsigned int &blockBag,
                              unsigned int &blockLeaf,
                              unsigned int &maxHeight) {
  unsigned int blockHeight = 0;
  blockLeaf = blockFac = blockBag = 0;
  for (auto & pair : treeBlock) {
    get<1>(pair)->BlockBump(blockHeight, maxHeight, blockFac, blockLeaf, blockBag);
  }

  return blockHeight;
}

 
/**
   @brief Builds segment of decision forest for a block of trees.

   @param ptBlock is a vector of PreTree objects.

   @param blockStart is the starting tree index for the block.

   @return void, with side-effected forest.
*/
void Train::BlockConsume(vector<TrainPair> &treeBlock,
                         unsigned int blockStart) {
  unsigned int blockIdx = blockStart;
  for (auto & trainPair : treeBlock) {
    const vector<unsigned int> leafMap = get<1>(trainPair)->Consume(forest.get(), blockIdx, predInfo);
    delete get<1>(trainPair);

    Leaves(get<0>(trainPair), leafMap, blockIdx);
    delete get<0>(trainPair);
    blockIdx++;
  }
}


void TrainReg::Leaves(Sample *sample, const vector<unsigned int> &leafMap, unsigned int blockIdx) const {
  leafReg->Leaves(sample, leafMap, blockIdx);
}


void TrainReg::Reserve(unsigned int leafEst, unsigned int bagEst) const {
  leafReg->Reserve(leafEst, bagEst);
}

void TrainCtg::Leaves(Sample *sample, const vector<unsigned int> &leafMap, unsigned int blockIdx) const {
  leafCtg->Leaves(sample, leafMap, blockIdx);
}

void TrainCtg::Reserve(unsigned int leafEst, unsigned int bagEst) const {
  leafCtg->Reserve(leafEst, bagEst);
}


void Train::getBag(unsigned char *bbRaw) const {
  bagRow->Serialize(bbRaw);
}
