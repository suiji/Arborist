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

// Testing only:
//#include <iostream>
using namespace std;

int Train::trainBlock = 0;
int Train::nTree = 0;
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
void Train::Init(double *_feNum, int _facCard[], int _feMap[], int _cardMax, int _nPredNum, int _nPredFac, int _nRow, int _nTree, int _nSamp, double _feSampleWeight[], bool _withRepl, int _trainBlock, int _minNode, double _minRatio, int _totLevels, int _ctgWidth, int _predFixed, double _predProb[]) {
  nTree = _nTree;
  nRow = _nRow;
  nPred = _nPredNum + _nPredFac;
  trainBlock = _trainBlock;
  PredBlock::Immutables(_feNum, _facCard, _feMap, _cardMax, _nPredNum, _nPredFac, nRow);
  Sample::Immutables(nRow, nPred, _nSamp, _feSampleWeight, _withRepl, _ctgWidth, nTree);
  SPNode::Immutables(_ctgWidth);
  SplitSig::Immutables(nPred, _minRatio);
  Index::Immutables(_minNode, _totLevels);
  PreTree::Immutables(nPred, _nSamp, _minNode);
  RestageMap::Immutables(nPred);
  SplitPred::Immutables(nPred, _ctgWidth, _predFixed, _predProb);
  Run::Immutables(nPred, _ctgWidth);
}


/**
   @brief Unsets immutables.

   @return void.
*/
void Train::DeImmutables() {
  nTree = nRow = nPred = trainBlock = 0;
  PredBlock::DeImmutables();
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
void Train::Regression(int _feRow[], int _feRank[], int _feInvNum[], double _y[], double _yRanked[], vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], vector<int> &_pred, vector<double> &_split, vector<int> &_bump, vector<unsigned int> &_facSplit, vector<unsigned int> &_rank, vector<unsigned int> &_sCount) {
  TrainReg *trainReg = new TrainReg(_y, _yRanked, _inBag, _orig, _facOrig, _predInfo, _pred, _split, _bump, _facSplit, _rank, _sCount);

  RowRank *rowRank = new RowRank(_feRow, _feRank, _feInvNum, nRow, nPred);
  trainReg->Forest(rowRank);

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
void Train::Classification(int _feRow[], int _feRank[], int _feInvNum[], int _yCtg[], int _ctgWidth, double _yProxy[], vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], vector<int> &_pred, vector<double> &_split, vector<int> &_bump, vector<unsigned int> &_facSplit, vector<double> &_weight) {

  TrainCtg *trainCtg = new TrainCtg(_yCtg, _ctgWidth, _yProxy, _inBag, _orig, _facOrig, _predInfo, _pred, _split, _bump, _facSplit, _weight);
  RowRank *rowRank = new RowRank(_feRow, _feRank, _feInvNum, nRow, nPred);
  trainCtg->Forest(rowRank);

  delete rowRank;
  delete trainCtg;
  DeImmutables();
}


/**
 */
TrainCtg::TrainCtg(int _yCtg[], unsigned int _ctgWidth, double _yProxy[], vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], vector<int> &_pred, vector<double> &_split, vector<int> &_bump, vector<unsigned int> &_facSplit, vector<double> &_weight) : Train(_inBag, _orig, _facOrig, _predInfo, _pred, _split, _bump, _facSplit), ctgWidth(_ctgWidth), weight(_weight), responseCtg(Response::FactoryCtg(_yCtg, _yProxy, nRow)) {
}


/**
 */
TrainCtg::~TrainCtg() {
  delete responseCtg;
}


/**
 */
TrainReg::TrainReg(double _y[], double _yRanked[], vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], vector<int> &_pred, vector<double> &_split, vector<int> &_bump, vector<unsigned int> &_facSplit, vector<unsigned int> &_rank, vector<unsigned int> &_sCount) : Train(_inBag, _orig, _facOrig, _predInfo, _pred, _split, _bump, _facSplit), rank(_rank), sCount(_sCount), responseReg(Response::FactoryReg(_y, _yRanked, nRow)) {
}


/**
 */
TrainReg::~TrainReg() {
  delete responseReg;
}


/**
 */  
Train::Train(vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], vector<int> &_pred, vector<double> &_split, vector<int> &_bump, vector<unsigned int> &_facSplit) : inBag(_inBag), orig(_orig), facOrig(_facOrig), predInfo(_predInfo), pred(_pred), split(_split), bump(_bump), facSplit(_facSplit) {
}


/**
  @brief Trains the requisite number of trees.

  @param trainBlock is the maximum Count of trees to train en block.

  @return void.
*/
void Train::Forest(const RowRank *rowRank) {
  unsigned int ibLength = BV::LengthAlign(nTree * nRow);
  inBag.reserve(ibLength);
  inBag.insert(inBag.end(), ibLength, 0);
  int tn;
  for (tn = 0; tn < nTree - trainBlock; tn += trainBlock) {
    Block(rowRank, tn, trainBlock);
  }

  if (tn < nTree) {
    Block(rowRank, tn, nTree - tn);
  }

  // Post-training fix-up to forest:
  
  // Normalizes 'predInfo' to per-tree means.
  double recipNTree = 1.0 / nTree;
  for (int i = 0; i < nPred; i++)
    predInfo[i] *= recipNTree;

  // Updates numerical splitting values from ranks.
  for (unsigned int i = 0; i < pred.size(); i++) {
    int predIdx = pred[i];
    if (bump[i] > 0 && !Forest::IsFactor(predIdx)) {
      split[i] = rowRank->MeanRank(predIdx, split[i]);
    }
  }
}


/**
 */
void TrainCtg::Block(const RowRank *rowRank, int tStart, int tCount) {
  SampleCtg **sampleBlock = responseCtg->BlockSample(rowRank, tCount);

  PreTree **ptBlock = Index::BlockTrees((Sample**) sampleBlock, tCount);
  if (tStart == 0)
    Reserve(ptBlock, tCount);

  int blockOrig = pred.size();
  BlockTree(ptBlock, tStart, tCount);
  BlockLeaf(ptBlock, sampleBlock, tStart, tCount, blockOrig);

  delete [] ptBlock;
  delete [] sampleBlock;
}


/**
 */
void TrainReg::Block(const RowRank *rowRank, int tStart, int tCount) {
  SampleReg **sampleBlock = responseReg->BlockSample(rowRank, tCount);
  PreTree **ptBlock = Index::BlockTrees((Sample**) sampleBlock, tCount);
  if (tStart == 0)
    Reserve(ptBlock, tCount);

  int blockOrig = pred.size();
  BlockTree(ptBlock, tStart, tCount);
  BlockLeaf(ptBlock, sampleBlock, tStart, tCount, blockOrig);

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
  PreTree::RefineHeight(maxHeight);

  double slop = (1.2 * nTree) / trainBlock;
  int heightEst = slop * blockHeight;
  pred.reserve(heightEst);
  split.reserve(heightEst);
  bump.reserve(heightEst);
  if (blockFac > 0) {
    int facEst = slop * blockFac;
    facSplit.reserve(facEst);
  }
  int bagEst = slop * blockBag;
  LeafReserve(heightEst, bagEst);
}


/**
   @brief Reserves storage for count and rank vectors.

   @heightEst is the estimated tree height.

   @bagEst is the estimated total bag count.

   @return void.
 */
void TrainReg::LeafReserve(int heightEst, int bagEst) {
  sCount.reserve(bagEst);
  rank.reserve(bagEst);
}


/**
   @brief Reserves storage for weight vector.

   @param heightEst is the estimated tree height.

   @param bagEst is unused.

   @return void.
 */
void TrainCtg::LeafReserve(int heightEst, int bagEst) {
  weight.reserve(heightEst * ctgWidth);
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

   @param tStart is the starting tree index for the block.

   @param tCount is the number of trees in the block.

   @return void.
*/
void Train::BlockTree(PreTree **ptBlock, int tStart, int tCount) {
  for (int tIdx = 0; tIdx < tCount; tIdx++) {
    int tNum = tStart + tIdx;
    int tOrig = pred.size();
    int bitOrig = facSplit.size();
    orig[tNum] = tOrig;
    facOrig[tNum] = bitOrig;
    PreTree *pt = ptBlock[tIdx];
    Grow(pt->Height(), pt->BitWidth());
    pt->DecTree(&pred[tOrig], &split[tOrig], &bump[tOrig], &facSplit[bitOrig], predInfo);
  }
}


/**
   @brief Extends the tree vectors by initialization.

   @param height is the height of the current tree.

   @param bitWidth is the size of the splitting bit vector.

   @return void.
*/
void Train::Grow(unsigned int height, unsigned int bitWidth) {
  pred.insert(pred.end(), height, 0);
  split.insert(split.end(), height, 0.0);
  bump.insert(bump.end(), height, 0);
  facSplit.insert(facSplit.end(), bitWidth, 0);
}


/**
   @brief Extracts categorical leaf information from a block of PreTrees.

   @return void.
 */
void TrainCtg::BlockLeaf(PreTree **ptBlock, SampleCtg **sampleBlock, int tStart, int tCount, int tOrig) {
  for (int i = 0; i < tCount; i++) {
    SampleCtg *sampleCtg = sampleBlock[i];
    BagSetTree(sampleCtg->InBag(), tStart + i);
    PreTree *pt = ptBlock[i];
    int height = pt->Height();
    int leafOrig = weight.size();
    weight.insert(weight.end(), ctgWidth * height, 0.0);
    sampleCtg->Leaves(pt->FrontierMap(), height, &pred[tOrig], &split[tOrig], &bump[tOrig], &weight[leafOrig]);
    tOrig += height;
    delete sampleCtg;
    delete pt;
  }
}


/**
   @brief Extracts regression leaf information for a block of PreTrees.
 */
void TrainReg::BlockLeaf(PreTree **ptBlock, SampleReg **sampleBlock, int tStart, int tCount, int tOrig) {
  for (int i = 0; i < tCount; i++) {
    SampleReg *sampleReg = sampleBlock[i];
    BagSetTree(sampleReg->InBag(), tStart + i);
    PreTree *pt = ptBlock[i];
    int height = pt->Height();
    int bagCount = sampleReg->BagCount();
    int bagOrig = rank.size();
    rank.insert(rank.end(), bagCount, 0);
    sCount.insert(sCount.end(), bagCount, 0);
    sampleReg->Leaves(pt->FrontierMap(), height, &pred[tOrig], &split[tOrig], &bump[tOrig], &rank[bagOrig], &sCount[bagOrig]);
    tOrig += height;

    delete sampleReg;
    delete pt;
  }
}


/**
  @brief Transfers per-tree bag bits to forest-wide matrix.

  @param bagSource holds in-bag rows as compressed bits.

  @param tIdx is the decision tree index.
  
  @return void.
*/
void Train::BagSetTree(const unsigned int bagSource[], int tIdx) {
  unsigned int slotBits = BV::SlotBits();
  int slotRow = 0;
  int slot = 0;
  for (unsigned int baseRow = 0; baseRow < nRow; baseRow += slotBits, slot++) {
    unsigned int sourceSlot = bagSource[slot];
    unsigned int mask = 1;
    unsigned int supRow = nRow < baseRow + slotBits ? nRow : baseRow + slotBits;
    for (unsigned int row = baseRow; row < supRow; row++, mask <<= 1) {
      if (sourceSlot & mask) { // row is in-bag.
	Forest::BagSet((unsigned int*) &inBag[0], nTree, tIdx, row);
      }
    }
    slotRow += slotBits;
  }
}


