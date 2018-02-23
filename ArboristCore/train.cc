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
#include "frameblock.h"
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
void Train::DeImmutables() {
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
TrainReg* Train::Regression(const FrameTrain *frameTrain,
			    const RowRank *rowRank,
			    const double *_y,
			    const unsigned int *_row2Rank,
			    unsigned int nTree) {
  auto trainReg = new TrainReg(frameTrain, _y, _row2Rank, nTree);
  trainReg->TrainForest(frameTrain, rowRank);

  DeImmutables();

  return trainReg;
}


TrainReg::TrainReg(const class FrameTrain *frameTrain,
		   const double *_y,
		   const unsigned int *_row2Rank,
		   unsigned int nTree) :
  Train(frameTrain, _y, _row2Rank, nTree),
  leafReg(new LeafTrainReg(nTree, frameTrain->NRow())) {
}


LeafTrain *TrainReg::Leaf() const {
  return SubLeaf();
}


/**
   @brief Regression constructor.
 */
Train::Train(const FrameTrain *frameTrain,
	     const double *_y,
	     const unsigned int *_row2Rank,
	     unsigned int _nTree) :
  nTree(_nTree),
  forest(new ForestTrain(nTree)),
  predInfo(vector<double>(frameTrain->NPred())),
  response(Response::FactoryReg(_y, _row2Rank)) {
}




/**
   @brief Static entry for regression training.

   @return void.
*/
TrainCtg *Train::Classification(const FrameTrain *frameTrain,
				const RowRank *rowRank,
			       const unsigned int *yCtg,
			       const double *yProxy,
			       unsigned int nCtg,
			       unsigned int nTree) {
  auto trainCtg = new TrainCtg(frameTrain, yCtg, yProxy, nCtg, nTree);
  trainCtg->TrainForest(frameTrain, rowRank);

  DeImmutables();

  return trainCtg;
}


TrainCtg::TrainCtg(const class FrameTrain *frameTrain,
		   const unsigned int *_yCtg,
		   const double *_yProxy,
		   unsigned int nCtg,
		   unsigned int nTree) :
  Train(frameTrain, _yCtg, _yProxy, nTree),
  leafCtg(new LeafTrainCtg(nTree, frameTrain->NRow(), nCtg)) {
}


LeafTrain *TrainCtg::Leaf() const {
  return SubLeaf();
}


/**
   @brief Classification constructor.
 */
Train::Train(const FrameTrain *frameTrain,
	     const unsigned int *_yCtg,
	     const double *_yProxy,
	     unsigned int _nTree) :
  nTree(_nTree),
  forest(new ForestTrain(nTree)),
  predInfo(vector<double>(frameTrain->NPred())),
  response(Response::FactoryCtg(_yCtg, _yProxy)) {
}


Train::~Train() {
  delete response;
}


/**
  @brief Trains the requisite number of trees.

  @param trainBlock is the maximum Count of trees to train en block.

  @return void.
*/
void Train::TrainForest(const FrameTrain *frameTrain,
			const RowRank *rowRank) {
  for (unsigned treeStart = 0; treeStart < nTree; treeStart += trainBlock) {
    unsigned int treeEnd = min(treeStart + trainBlock, nTree); // one beyond.
    TreeBlock(frameTrain, rowRank, treeStart, treeEnd - treeStart);
  }
  forest->SplitUpdate(frameTrain, rowRank);
}


/**
   @brief  Creates a block of root samples and trains each one.

   @return void.
 */
void Train::TreeBlock(const FrameTrain *frameTrain, const RowRank *rowRank, unsigned int tStart, unsigned int tCount) {
  vector<Sample*> sampleBlock(tCount);
  vector<PreTree*> ptBlock(tCount);
  IndexLevel::TreeBlock(frameTrain, rowRank, response, sampleBlock, ptBlock);

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
  forest->Reserve(blockHeight, blockFac, slop);
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
    const vector<unsigned int> leafMap = pt->Consume(forest, blockStart + blockIdx, predInfo);
    delete pt;
    Leaf()->Leaves(frameTrain, sampleBlock[blockIdx], leafMap, blockStart + blockIdx);
    delete sampleBlock[blockIdx];
    blockIdx++;
  }
}


