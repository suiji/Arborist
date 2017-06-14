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
#include "predblock.h"
#include "index.h"
#include "pretree.h"
#include "sample.h"
#include "samplepred.h"
#include "splitsig.h"
#include "response.h"
#include "splitpred.h"
#include "leaf.h"

#include <algorithm>
// Testing only:
//#include <iostream>
//using namespace std;

unsigned int Train::trainBlock = 0;


/**
   @brief Initializes immutable values for "top-level" classes directly,
   for those instances in which static initialization seems to make
   more sense.  These may, in turn, initialize subclasses or classes
   with objects strictly local to them.

   @param minNode is the minimal index node size on which to split.

   @param minRatio is a threshold ratio for determining whether to split.

   @param totLevels, if positive, limits the number of levels to build.

   @return void.
*/
void Train::Init(unsigned int _nPred, unsigned int _nTree, unsigned int _nSamp, const std::vector<double> &_feSampleWeight, bool _withRepl, unsigned int _trainBlock, unsigned int _minNode, double _minRatio, unsigned int _totLevels, unsigned int _ctgWidth, unsigned int _predFixed, const double _splitQuant[], const double _predProb[], bool _thinLeaves, const double _regMono[]) {
  trainBlock = _trainBlock;
  Sample::Immutables(_nSamp, _feSampleWeight, _withRepl, _ctgWidth, _nTree);
  SPNode::Immutables(_ctgWidth);
  SplitSig::Immutables(_minRatio);
  IndexLevel::Immutables(_minNode, _totLevels);
  Leaf::Immutables(_thinLeaves);
  PreTree::Immutables(_nSamp, _minNode);
  SplitPred::Immutables(_nPred, _ctgWidth, _predFixed, _predProb, _regMono);
  ForestNode::Immutables(_splitQuant);
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
  Leaf::DeImmutables();
  PreTree::DeImmutables();
  Sample::DeImmutables();
  SPNode::DeImmutables();
  SplitPred::DeImmutables();
}


/**
   @brief Regression constructor.
 */
Train::Train(const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank, const PMTrain *pmTrain, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<double> &_predInfo, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagLeaf> &_bagRow, std::vector<unsigned int> &_bagBits) : nTree(_origin.size()), forest(new ForestTrain(_forestNode, _origin, _facOrigin, _facSplit)), predInfo(_predInfo), response(Response::FactoryReg(_y, _row2Rank, pmTrain, _leafOrigin, _leafNode, _bagRow, _bagBits)) {
}


/**
   @brief Static entry for regression training.

   @param trainBlock is the maximum number of trees trainable simultaneously.

   @param minNode is the mininal number of sample indices represented by a tree node.

   @param minRatio is the minimum information ratio of a node to its parent.

   @return forest height, with output reference parameter.
*/
void Train::Regression(const unsigned int _feRow[], const unsigned int _feRank[], const unsigned int _numOff[], const double _numVal[], const unsigned int _feRLE[], unsigned int _feRLELength, const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<double> &_predInfo, const std::vector<unsigned int> &_feCard, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, double _autoCompress, std::vector<class BagLeaf> &_bagRow, std::vector<unsigned int> &_bagBits) {
  PMTrain *pmTrain = new PMTrain(_feCard, _predInfo.size(), _y.size());
  Train *train = new Train(_y, _row2Rank, pmTrain, _origin, _facOrigin, _predInfo, _forestNode, _facSplit, _leafOrigin, _leafNode, _bagRow, _bagBits);

  RowRank *rowRank = new RowRank(pmTrain, _feRow, _feRank, _numOff, _numVal, _feRLE, _feRLELength, _autoCompress);
  train->TrainForest(pmTrain, rowRank);

  delete rowRank;
  delete train;
  delete pmTrain;
  DeImmutables();
}


/**
   @brief Classification constructor.
 */
Train::Train(const std::vector<unsigned int> &_yCtg, unsigned int _ctgWidth, const std::vector<double> &_yProxy, const PMTrain *pmTrain, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<double> &_predInfo, std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &_bagRow, std::vector<unsigned int> &_bagBits, std::vector<double> &_weight) : nTree(_origin.size()), forest(new ForestTrain(_forestNode, _origin, _facOrigin, _facSplit)), predInfo(_predInfo), response(Response::FactoryCtg(_yCtg, _yProxy, pmTrain, _leafOrigin, _leafNode, _bagRow, _bagBits, _weight, _ctgWidth)) {
}


/**
   @brief Static entry for regression training.

   @return void.
*/
void Train::Classification(const unsigned int _feRow[], const unsigned int _feRank[], const unsigned int _numOff[], const double _numVal[], const unsigned int _feRLE[], unsigned int _rleLength, const std::vector<unsigned int>  &_yCtg, unsigned int _ctgWidth, const std::vector<double> &_yProxy, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<double> &_predInfo, const std::vector<unsigned int> &_feCard, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, double _autoCompress, std::vector<class BagLeaf> &_bagRow, std::vector<unsigned int> &_bagBits, std::vector<double> &_weight) {
  PMTrain *pmTrain = new PMTrain(_feCard, _predInfo.size(), _yCtg.size());
  Train *train = new Train(_yCtg, _ctgWidth, _yProxy, pmTrain, _origin, _facOrigin, _predInfo, _forestNode, _facSplit, _leafOrigin, _leafNode, _bagRow, _bagBits, _weight);

  RowRank *rowRank = new RowRank(pmTrain, _feRow, _feRank, _numOff, _numVal, _feRLE, _rleLength, _autoCompress);
  train->TrainForest(pmTrain, rowRank);

  delete rowRank;
  delete train;
  delete pmTrain;
  DeImmutables();
}


Train::~Train() {
  delete response;
  delete forest;
}


/**
  @brief Trains the requisite number of trees.

  @param trainBlock is the maximum Count of trees to train en block.

  @return void.
*/
void Train::TrainForest(const PMTrain *pmTrain, const RowRank *rowRank) {
  for (unsigned treeStart = 0; treeStart < nTree; treeStart += trainBlock) {
    unsigned int treeEnd = std::min(treeStart + trainBlock, nTree); // one beyond.
    Block(rowRank, treeStart, treeEnd - treeStart);
  }
    
  // Normalizes 'predInfo' to per-tree means.
  double recipNTree = 1.0 / nTree;
  for (unsigned int i = 0; i < predInfo.size(); i++) {
    predInfo[i] *= recipNTree;
  }

  forest->SplitUpdate(pmTrain, rowRank);
}


/**

   @param tEnd is one 
 */
void Train::Block(const RowRank *rowRank, unsigned int tStart, unsigned int tCount) {
  PreTree **ptBlock = response->BlockTree(rowRank, tCount);
  if (tStart == 0)
    Reserve(ptBlock, tCount);

  BlockTree(ptBlock, tStart, tCount);
  response->DeBlock(tCount);

  delete [] ptBlock;
}

 
/** 
  @brief Estimates forest heights using size parameters from the first
  trained block of trees.

  @param ptBlock is a block of PreTree references.

  @return void.
*/
void Train::Reserve(PreTree **ptBlock, unsigned int tCount) {
  unsigned int blockFac, blockBag, blockLeaf;
  unsigned int maxHeight = 0;
  unsigned int blockHeight = BlockPeek(ptBlock, tCount, blockFac, blockBag, blockLeaf, maxHeight);
  PreTree::Reserve(maxHeight);

  double slop = (slopFactor * nTree) / trainBlock;
  forest->Reserve(blockHeight, blockFac, slop);
  response->LeafReserve(slop * blockLeaf, slop * blockBag);
}


/**
   @brief Accumulates block size parameters as clues to forest-wide sizes.
   Estimates improve with larger blocks, at the cost of higher memory footprint.

   @return sum of tree sizes over block, plus output parameters.
 */
unsigned int Train::BlockPeek(PreTree **ptBlock, unsigned int tCount, unsigned int &blockFac, unsigned int &blockBag, unsigned int &blockLeaf, unsigned int &maxHeight) {
  unsigned int blockHeight = 0;
  blockLeaf = blockFac = blockBag = 0;
  for (unsigned int i = 0; i < tCount; i++) {
    ptBlock[i]->BlockBump(blockHeight, maxHeight, blockFac, blockLeaf, blockBag);
  }

  return blockHeight;
}

 
/**
   @brief Builds segment of decision forest for a block of trees.

   @param ptBlock is a vector of PreTree objects.

   @param blockStart is the starting tree index for the block.

   @param blockCount is the number of trees in the block.

   @return void, with side-effected forest.
*/
void Train::BlockTree(PreTree **ptBlock, unsigned int blockStart, unsigned int blockCount) {
  for (unsigned int blockIdx = 0; blockIdx < blockCount; blockIdx++) {
    unsigned int tIdx = blockStart + blockIdx;
    const std::vector<unsigned int> leafMap = ptBlock[blockIdx]->DecTree(forest, tIdx, predInfo);
    response->Leaves(leafMap, blockIdx, tIdx);

    delete ptBlock[blockIdx];
  }
}


