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
unsigned int Train::nTree = 0;
unsigned int Train::nRow = 0;
unsigned int Train::nPred = 0;


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
void Train::Init(const double _feNum[], const unsigned int _feCard[], unsigned int _cardMax, unsigned int _nPredNum, unsigned int _nPredFac, unsigned int _nRow, unsigned int _nTree, unsigned int _nSamp, const double _feSampleWeight[], bool _withRepl, unsigned int _trainBlock, unsigned int _minNode, double _minRatio, unsigned int _totLevels, unsigned int _ctgWidth, unsigned int _predFixed, const double _predProb[], const double _regMono[]) {
  nTree = _nTree;
  nRow = _nRow;
  nPred = _nPredNum + _nPredFac;
  trainBlock = _trainBlock;
  PBTrain::Immutables(_feNum, _feCard, _cardMax, _nPredNum, _nPredFac, nRow);
  Sample::Immutables(nRow, nPred, _nSamp, _feSampleWeight, _withRepl, _ctgWidth, nTree);
  SPNode::Immutables(_ctgWidth);
  SplitSig::Immutables(nPred, _minRatio);
  Index::Immutables(_minNode, _totLevels);
  PreTree::Immutables(nPred, _nSamp, _minNode);
  SplitPred::Immutables(nPred, _ctgWidth, _predFixed, _predProb, _regMono);
}


/**
   @brief Unsets immutables.

   @return void.
*/
void Train::DeImmutables() {
  nTree = nRow = nPred = trainBlock = 0;
  PBTrain::DeImmutables();
  SplitSig::DeImmutables();
  Index::DeImmutables();
  PreTree::DeImmutables();
  Sample::DeImmutables();
  SPNode::DeImmutables();
  SplitPred::DeImmutables();
}


/**
   @brief Regression constructor.
 */
Train::Train(const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, double _predInfo[], std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagRow> &_bagRow, std::vector<unsigned int> &_rank) : forest(new Forest(_forestNode, _origin, _facOrigin, _facSplit)), predInfo(_predInfo), response(Response::FactoryReg(_y, _row2Rank, _leafOrigin, _leafNode, _bagRow, _rank)) {
}


/**
   @brief Static entry for regression training.

   @param trainBlock is the maximum number of trees trainable simultaneously.

   @param minNode is the mininal number of sample indices represented by a tree node.

   @param minRatio is the minimum information ratio of a node to its parent.

   @return forest height, with output reference parameter.
*/
void Train::Regression(unsigned int _feRow[], unsigned int _feRank[], unsigned int _feInvNum[], const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, double _predInfo[], std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagRow> &_bagRow, std::vector<unsigned int> &_rank) {
  Train *train = new Train(_y, _row2Rank, _origin, _facOrigin, _predInfo, _forestNode, _facSplit, _leafOrigin, _leafNode, _bagRow, _rank);

  RowRank *rowRank = new RowRank(_feRow, _feRank, _feInvNum, nRow, nPred);
  train->ForestTrain(rowRank);

  delete rowRank;
  delete train;
  DeImmutables();
}


/**
   @brief Classification constructor.
 */
Train::Train(const std::vector<unsigned int> &_yCtg, unsigned int _ctgWidth, const std::vector<double> &_yProxy, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, double _predInfo[], std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, std::vector<double> &_weight) : forest(new Forest(_forestNode, _origin, _facOrigin, _facSplit)), predInfo(_predInfo), response(Response::FactoryCtg(_yCtg, _yProxy, _leafOrigin, _leafNode, _bagRow, _weight, _ctgWidth)) {
}


/**
   @brief Static entry for regression training.

   @return void.
*/
void Train::Classification(unsigned int _feRow[], unsigned int _feRank[], unsigned int _feInvNum[], const std::vector<unsigned int>  &_yCtg, unsigned int _ctgWidth, const std::vector<double> &_yProxy, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, double _predInfo[], std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagRow> &_bagRow, std::vector<double> &_weight) {
  Train *train = new Train(_yCtg, _ctgWidth, _yProxy, _origin, _facOrigin, _predInfo, _forestNode, _facSplit, _leafOrigin, _leafNode, _bagRow, _weight);

  RowRank *rowRank = new RowRank(_feRow, _feRank, _feInvNum, nRow, nPred);
  train->ForestTrain(rowRank);

  delete rowRank;
  delete train;
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
void Train::ForestTrain(const RowRank *rowRank) {
  for (unsigned treeStart = 0; treeStart < nTree; treeStart += trainBlock) {
    unsigned int treeEnd = std::min(treeStart + trainBlock, nTree); // one beyond.
    Block(rowRank, treeStart, treeEnd - treeStart);
  }
    
  // Normalizes 'predInfo' to per-tree means.
  double recipNTree = 1.0 / nTree;
  for (unsigned int i = 0; i < nPred; i++)
    predInfo[i] *= recipNTree;

  forest->SplitUpdate(rowRank);
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
    PreTree *pt = ptBlock[i];
    unsigned int height = pt->Height();
    maxHeight = std::max(height, maxHeight);
    blockHeight += height;
    blockFac += pt->BitWidth();
    blockLeaf += pt->LeafCount();
    blockBag += pt->BagCount();
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


