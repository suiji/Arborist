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

#include "sample.h"
#include "train.h"
#include "forest.h"
#include "rowrank.h"
#include "framemap.h"
#include "index.h"
#include "pretree.h"
#include "samplepred.h"
#include "splitsig.h"
#include "response.h"
#include "splitpred.h"
#include "leaf.h"
#include "level.h"
#include "coproc.h"

#include <algorithm>


unsigned int Train::trainBlock = 0;

/**
   @brief Registers training tree-block count.

   @param _trainBlock is the number of trees by which to block.

   @return void.
*/
void Train::InitBlock(unsigned int _trainBlock) {
  trainBlock = _trainBlock;
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
void Train::InitSample(unsigned int nSamp,
		       const vector<double> &sampleWeight,
		       bool withRepl) {
  Sample::Immutables(nSamp, sampleWeight, withRepl);
}

/**
   @brief Registers parameters governing splitting.

   @return void.
 */
void Train::InitSplit(unsigned int minNode,
		      unsigned int totLevels,
		      double minRatio) {
  IndexLevel::Immutables(minNode, totLevels);
  SplitSig::Immutables(minRatio);
}


/**
   @brief Registers width of categorical response.

   @return void.
 */
void Train::InitCtgWidth(unsigned int ctgWidth) {
  SampleNux::Immutables(ctgWidth);
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
  SplitSig::DeImmutables();
  IndexLevel::DeImmutables();
  LeafTrain::DeImmutables();
  PreTree::DeImmutables();
  Sample::DeImmutables();
  SampleNux::DeImmutables();
  Level::DeImmutables();
  SPReg::DeImmutables();
}


/**
   @brief Static entry for regression training.

   @param trainBlock is the maximum number of trees trainable simultaneously.

   @param minNode is the mininal number of sample indices represented by a tree node.

   @param minRatio is the minimum information ratio of a node to its parent.

   @return forest height, with output reference parameter.
*/
unique_ptr<TrainReg> Train::Regression(const FrameTrain *frameTrain,
				       const RankedSet *rankedPair,
				       const double *y,
				       const unsigned int *row2Rank,
				       unsigned int nTree) {
  auto trainReg = make_unique<TrainReg>(frameTrain, y, row2Rank, nTree);
  trainReg->TrainForest(frameTrain, rankedPair);

  return trainReg;
}


TrainReg::TrainReg(const class FrameTrain *frameTrain,
		   const double *y,
		   const unsigned int *row2Rank,
		   unsigned int nTree) :
  Train(frameTrain, y, row2Rank, nTree),
  leafReg(make_unique<LeafTrainReg>(nTree, frameTrain->NRow())) {
}


LeafTrain *TrainReg::Leaf() const {
  return SubLeaf();
}


/**
   @brief Regression constructor.
 */
Train::Train(const FrameTrain *frameTrain,
	     const double *y,
	     const unsigned int *row2Rank,
	     unsigned int _nTree) :
  nTree(_nTree),
  forest(make_unique<ForestTrain>(nTree)),
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
					   unsigned int nTree) {
  auto trainCtg = make_unique<TrainCtg>(frameTrain, yCtg, yProxy, nCtg, nTree);
  trainCtg->TrainForest(frameTrain, rankedPair);

  return trainCtg;
}


TrainCtg::TrainCtg(const class FrameTrain *frameTrain,
		   const unsigned int *yCtg,
		   const double *yProxy,
		   unsigned int nCtg,
		   unsigned int nTree) :
  Train(frameTrain, yCtg, yProxy, nTree),
  leafCtg(make_unique<LeafTrainCtg>(nTree, frameTrain->NRow(), nCtg)) {
}


LeafTrain *TrainCtg::Leaf() const {
  return SubLeaf();
}


/**
   @brief Classification constructor.
 */
Train::Train(const FrameTrain *frameTrain,
	     const unsigned int *yCtg,
	     const double *yProxy,
	     unsigned int _nTree) :
  nTree(_nTree),
  forest(make_unique<ForestTrain>(nTree)),
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
  for (unsigned treeStart = 0; treeStart < nTree; treeStart += trainBlock) {
    unsigned int treeEnd = min(treeStart + trainBlock, nTree); // one beyond.
    TreeBlock(frameTrain, rankedPair->GetRowRank(), treeStart, treeEnd - treeStart);
  }
  Forest()->SplitUpdate(frameTrain, rankedPair->GetNumRanked());
}


/**
   @brief  Creates a block of root samples and trains each one.

   @return void.
 */
void Train::TreeBlock(const FrameTrain *frameTrain,
		      const RowRank *rowRank,
		      unsigned int tStart,
		      unsigned int tCount) {
  vector<Sample*> sampleBlock(tCount);
  vector<PreTree*> ptBlock(tCount);
  // auto treeBlock = IndexLevel::TreeBlock(frameTrain. rowRank, response);
  IndexLevel::TreeBlock(frameTrain, rowRank, response.get(), sampleBlock, ptBlock);

  if (tStart == 0)
    Reserve(ptBlock);
  BlockConsume(frameTrain, sampleBlock, ptBlock, tStart);
}

 
/** 
  @brief Estimates forest heights using size parameters from the first
  trained block of trees.

  @param ptBlock is a block of PreTree references.

  @return void.
*/
void Train::Reserve(vector<PreTree*> &ptBlock) {
  unsigned int blockFac, blockBag, blockLeaf;
  unsigned int maxHeight = 0;
  unsigned int blockHeight = BlockPeek(ptBlock, blockFac, blockBag, blockLeaf, maxHeight);
  PreTree::Reserve(maxHeight);

  double slop = (slopFactor * nTree) / trainBlock;
  Forest()->Reserve(blockHeight, blockFac, slop);
  Leaf()->Reserve(slop * blockLeaf, slop * blockBag);
}


/**
   @brief Accumulates block size parameters as clues to forest-wide sizes.
   Estimates improve with larger blocks, at the cost of higher memory footprint.

   @return sum of tree sizes over block, plus output parameters.
 */
unsigned int Train::BlockPeek(vector<PreTree*> &ptBlock,
			      unsigned int &blockFac,
			      unsigned int &blockBag,
			      unsigned int &blockLeaf,
			      unsigned int &maxHeight) {
  unsigned int blockHeight = 0;
  blockLeaf = blockFac = blockBag = 0;
  for (auto & pt : ptBlock) {
    pt->BlockBump(blockHeight, maxHeight, blockFac, blockLeaf, blockBag);
  }

  return blockHeight;
}

 
/**
   @brief Builds segment of decision forest for a block of trees.

   @param ptBlock is a vector of PreTree objects.

   @param blockStart is the starting tree index for the block.

   @return void, with side-effected forest.
*/
void Train::BlockConsume(const FrameTrain *frameTrain,
			 const vector<Sample*> &sampleBlock,
			 vector<PreTree*> &ptBlock,
			 unsigned int blockStart) {
  unsigned int blockIdx = 0;
  for (auto & pt : ptBlock) {
    const vector<unsigned int> leafMap = pt->Consume(Forest(), blockStart + blockIdx, predInfo);
    delete pt;
    Leaf()->Leaves(frameTrain, sampleBlock[blockIdx], leafMap, blockStart + blockIdx);
    delete sampleBlock[blockIdx];
    blockIdx++;
  }
}


