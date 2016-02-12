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
#include "predblock.h"
#include "index.h"
#include "pretree.h"
#include "sample.h"
#include "samplepred.h"
#include "splitsig.h"
#include "response.h"
#include "restage.h"
#include "splitpred.h"
#include "run.h"
#include "leaf.h"

// Testing only:
//#include <iostream>
using namespace std;

int Train::trainBlock = 0;
unsigned int Train::nTree = 0;
unsigned int Train::nRow = 0;
int Train::nPred = 0;

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
void Train::Init(double *_feNum, int _facCard[], int _cardMax, int _nPredNum, int _nPredFac, int _nRow, int _nTree, int _nSamp, double _feSampleWeight[], bool _withRepl, int _trainBlock, int _minNode, double _minRatio, int _totLevels, int _ctgWidth, int _predFixed, double _predProb[], int _regMono[]) {
  nTree = _nTree;
  nRow = _nRow;
  nPred = _nPredNum + _nPredFac;
  trainBlock = _trainBlock;
  PBTrain::Immutables(_feNum, _facCard, _cardMax, _nPredNum, _nPredFac, nRow);
  Sample::Immutables(nRow, nPred, _nSamp, _feSampleWeight, _withRepl, _ctgWidth, nTree);
  SPNode::Immutables(_ctgWidth);
  SplitSig::Immutables(nPred, _minRatio);
  Index::Immutables(_minNode, _totLevels);
  PreTree::Immutables(nPred, _nSamp, _minNode);
  RestageMap::Immutables(nPred);
  SplitPred::Immutables(nPred, _ctgWidth, _predFixed, _predProb, _regMono);
  Run::Immutables(nPred, _ctgWidth);
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
  Run::DeImmutables();
}


/**
   @brief Static entry for regression training.

   @param trainBlock is the maximum number of trees trainable simultaneously.

   @param minNode is the mininal number of sample indices represented by a tree node.

   @param minRatio is the minimum information ratio of a node to its parent.

   @return forest height, with output reference parameter.
*/
void Train::Regression(int _feRow[], int _feRank[], int _feInvNum[], double _y[], double _yRanked[], std::vector<unsigned int> &_inBag, std::vector<unsigned int> &_orig, std::vector<unsigned int> &_facOrig, double _predInfo[], std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_rank, std::vector<unsigned int> &_sCount) {
  TrainReg *trainReg = new TrainReg(_y, _yRanked, _inBag, _orig, _facOrig, _predInfo, _forestNode, _facSplit, _rank, _sCount);

  RowRank *rowRank = new RowRank(_feRow, _feRank, _feInvNum, nRow, nPred);
  trainReg->ForestTrain(rowRank);

  delete rowRank;
  delete trainReg;
  DeImmutables();
}


/**
   @brief Static entry for regression training.

   @param rowRank is the sorted predictor table.

   @param trainBlock is the number of trees to train per block.

   @param treeStart is the absolute tree index.  Only client is 'inBag'.

   @return void.
*/
void Train::Classification(int _feRow[], int _feRank[], int _feInvNum[], int _yCtg[], int _ctgWidth, double _yProxy[], std::vector<unsigned int> &_inBag, std::vector<unsigned int> &_orig, std::vector<unsigned int> &_facOrig, double _predInfo[], std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<double> &_weight) {

  TrainCtg *trainCtg = new TrainCtg(_yCtg, _ctgWidth, _yProxy, _inBag, _orig, _facOrig, _predInfo, _forestNode, _facSplit, _weight);
  RowRank *rowRank = new RowRank(_feRow, _feRank, _feInvNum, nRow, nPred);
  trainCtg->ForestTrain(rowRank);

  delete rowRank;
  delete trainCtg;
  DeImmutables();
}


/**
 */
TrainCtg::TrainCtg(int _yCtg[], unsigned int _ctgWidth, double _yProxy[], std::vector<unsigned int> &_inBag, std::vector<unsigned int> &_orig, std::vector<unsigned int> &_facOrig, double _predInfo[], std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<double> &_weight) : Train(_inBag, _orig, _facOrig, _predInfo, _forestNode, _facSplit), ctgWidth(_ctgWidth), weight(_weight), responseCtg(Response::FactoryCtg(_yCtg, _yProxy, nRow)) {
}


/**
 */
TrainCtg::~TrainCtg() {
  delete leafCtg;
  delete responseCtg;
}


/**
 */
TrainReg::TrainReg(double _y[], double _yRanked[], std::vector<unsigned int> &_inBag, std::vector<unsigned int> &_orig, std::vector<unsigned int> &_facOrig, double _predInfo[], std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_rank, std::vector<unsigned int> &_sCount) : Train(_inBag, _orig, _facOrig, _predInfo, _forestNode, _facSplit), rank(_rank), sCount(_sCount), responseReg(Response::FactoryReg(_y, _yRanked, nRow)) {
}


/**
 */
TrainReg::~TrainReg() {
  delete leafReg;
  delete responseReg;
}


/**
 */  
Train::Train(vector<unsigned int> &_inBag, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, double _predInfo[], std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit) : forest(new Forest(_forestNode, _origin, _facOrigin, _facSplit)), inBag(_inBag), predInfo(_predInfo) {
}


Train::~Train() {
  delete forest;
}


/**
  @brief Trains the requisite number of trees.

  @param trainBlock is the maximum Count of trees to train en block.

  @return void.
*/
void Train::ForestTrain(const RowRank *rowRank) {
  BitMatrix *forestBag = new BitMatrix(nRow, nTree);
  
  unsigned int tn;
  for (tn = 0; tn < nTree - trainBlock; tn += trainBlock) {
    Block(rowRank, forestBag, tn, trainBlock);
  }

  if (tn < nTree) {
    Block(rowRank, forestBag, tn, nTree - tn);
  }

  forestBag->Consume(inBag);
  delete forestBag;

  // Post-training fix-up to forest:
  
  // Normalizes 'predInfo' to per-tree means.
  double recipNTree = 1.0 / nTree;
  for (int i = 0; i < nPred; i++)
    predInfo[i] *= recipNTree;

  forest->ScoreUpdate(rowRank);
}


/**
 */
void TrainCtg::Block(const RowRank *rowRank, BitMatrix *forestBag, int tStart, int tCount) {
  SampleCtg **sampleBlock = responseCtg->BlockSample(rowRank, tCount);

  PreTree **ptBlock = Index::BlockTrees((Sample**) sampleBlock, tCount);
  if (tStart == 0)
    Reserve(ptBlock, tCount);

  BlockTree(ptBlock, tStart, tCount);
  BlockLeaf(ptBlock, sampleBlock, forestBag, tStart, tCount);

  delete [] ptBlock;
  delete [] sampleBlock;
}


/**
 */
void TrainReg::Block(const RowRank *rowRank, BitMatrix *forestBag, int tStart, int tCount) {
  SampleReg **sampleBlock = responseReg->BlockSample(rowRank, tCount);
  PreTree **ptBlock = Index::BlockTrees((Sample**) sampleBlock, tCount);
  if (tStart == 0)
    Reserve(ptBlock, tCount);

  BlockTree(ptBlock, tStart, tCount);
  BlockLeaf(ptBlock, sampleBlock, forestBag, tStart, tCount);

  delete [] ptBlock;
  delete [] sampleBlock;
}
 
 
/** 
  @brief Estimates forest heights using size parameters from the first
  trained block of trees.

  @param ptBlock is a block of PreTree references.

  @return void.
*/
void Train::Reserve(PreTree **ptBlock, int tCount) {
  int blockFac, blockBag;
  int maxHeight = 0;
  int blockHeight = BlockPeek(ptBlock, tCount, blockFac, blockBag, maxHeight);
  PreTree::Reserve(maxHeight);

  double slop = (slopFactor * nTree) / trainBlock;
  forest->Reserve(blockHeight, blockFac, slop);
  FactoryLeaf(blockHeight * slop, slop * blockBag);
}


/**
   @brief Initializes LeafReg with estimated vector sizes.

   @heightEst is unused.

   @bagEst is the estimated total bag count.

   @return void, with side-effected leaf object.
 */
void TrainReg::FactoryLeaf(int heightEst, int bagEst) {
  leafReg = new LeafReg(forest, sCount, rank, bagEst);
}


/**
   @brief Initializes LeafCtg with estimated vector sizes.

   @param heightEst is the estimated tree height.

   @param bagEst is unused.

   @return void, with side-effected leaf object.
 */
void TrainCtg::FactoryLeaf(int heightEst, int bagEst) {
  leafCtg = new LeafCtg(forest, weight, ctgWidth, heightEst);
}


/**
   @brief Accumulates block size parameters as clues to forest-wide sizes.
   Estimates improve with larger blocks, at the cost of higher memory footprint.

   @return sum of tree sizes over block, plus output parameters.
 */
int Train::BlockPeek(PreTree **ptBlock, int tCount, int &blockFac, int &blockBag, int &maxHeight) {
  int blockHeight = 0;
  blockFac = blockBag = 0;
  for (int i = 0; i < tCount; i++) {
    PreTree *pt = ptBlock[i];
    int height = pt->Height();
    maxHeight = height > maxHeight ? height : maxHeight;
    blockHeight += height;
    blockFac += pt->BitWidth();
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
void Train::BlockTree(PreTree **ptBlock, int blockStart, int blockCount) {
  for (int blockIdx = 0; blockIdx < blockCount; blockIdx++) {
    ptBlock[blockIdx]->DecTree(forest, blockStart + blockIdx, predInfo);
  }
}


/**
   @brief Extracts categorical leaf information from a block of PreTrees.

   @return void.
 */
void TrainCtg::BlockLeaf(PreTree **ptBlock, SampleCtg **sampleBlock, BitMatrix *forestBag, int tStart, int tCount) {
  for (int i = 0; i < tCount; i++) {
    int tIdx = tStart + i;
    BagSetTree(sampleBlock[i]->TreeBag(), forestBag, tIdx);
    leafCtg->Leaves(sampleBlock[i], ptBlock[i]->FrontierMap(), tIdx);

    delete sampleBlock[i];
    delete ptBlock[i];
  }
}


/**
   @brief Extracts regression leaf information for a block of PreTrees.
 */
void TrainReg::BlockLeaf(PreTree **ptBlock, SampleReg **sampleBlock, BitMatrix *forestBag, int tStart, int tCount) {
  for (int i = 0; i < tCount; i++) {
    int tIdx = tStart + i;
    BagSetTree(sampleBlock[i]->TreeBag(), forestBag, tIdx);
    leafReg->Leaves(sampleBlock[i], ptBlock[i]->FrontierMap(), tIdx);

    delete sampleBlock[i];
    delete ptBlock[i];
  }
}


/**
  @brief Transfers per-tree bag bits to forest-wide matrix.

  @param treeBag holds in-bag rows as compressed bits.

  @param forestBag outputs the in-bag set for the forest.

  @param tIdx is the decision tree index.
  
  @return void, with output reference bit matrix.
*/
void Train::BagSetTree(const BV *treeBag, BitMatrix *forestBag, int tIdx) {
  unsigned int slotBits = BV::SlotBits();
  int slotRow = 0;
  unsigned int slot = 0;
  for (unsigned int baseRow = 0; baseRow < nRow; baseRow += slotBits, slot++) {
    unsigned int sourceSlot = treeBag->Slot(slot);
    unsigned int mask = 1;
    unsigned int supRow = nRow < baseRow + slotBits ? nRow : baseRow + slotBits;
    for (unsigned int row = baseRow; row < supRow; row++, mask <<= 1) {
      if (sourceSlot & mask) { // row is in-bag.
	forestBag->SetBit(row, tIdx);
      }
    }
    slotRow += slotBits;
  }
}


