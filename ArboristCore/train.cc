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
#include "ompthread.h"
#include "coproc.h"

#include <algorithm>


unsigned int Train::trainBlock = 0;

void Train::initBlock(unsigned int trainBlock_) {
  trainBlock = trainBlock_;
}


void Train::initCDF(const vector<double> &feSplitQuant) {
  TreeNode::Immutables(feSplitQuant);
}


void Train::initProb(unsigned int predFixed,
                     const vector<double> &predProb) {
  Level::immutables(predFixed, predProb);
}


void Train::initTree(unsigned int nSamp,
                     unsigned int minNode,
                     unsigned int leafMax) {
  PreTree::immutables(nSamp, minNode, leafMax);
}


void Train::initOmp(unsigned int nThread) {
  OmpThread::init(nThread);
}


void Train::initSample(unsigned int nSamp) {
  Sample::immutables(nSamp);
}

void Train::initSplit(unsigned int minNode,
                      unsigned int totLevels,
                      double minRatio) {
  IndexLevel::immutables(minNode, totLevels);
  SplitCand::immutables(minRatio);
}


void Train::initCtgWidth(unsigned int ctgWidth) {
  SampleNux::immutables(ctgWidth);
}


void Train::initMono(const FrameTrain* frameTrain,
                     const vector<double> &regMono) {
  SPReg::Immutables(frameTrain, regMono);
}


void Train::deInit() {
  trainBlock = 0;
  TreeNode::DeImmutables();
  SplitCand::deImmutables();
  IndexLevel::deImmutables();
  PreTree::deImmutables();
  Sample::deImmutables();
  SampleNux::deImmutables();
  Level::deImmutables();
  SPReg::DeImmutables();
  OmpThread::deInit();
}


unique_ptr<Train> Train::regression(const FrameTrain *frameTrain,
                                       const RankedSet *rankedPair,
                                       const double *y,
                                       unsigned int treeChunk) {
  auto trainReg = make_unique<Train>(frameTrain, y, treeChunk);
  trainReg->trainChunk(frameTrain, rankedPair);

  return trainReg;
}


Train::Train(const FrameTrain *frameTrain,
             const double *y,
             unsigned int treeChunk_) :
  nRow(frameTrain->getNRow()),
  treeChunk(treeChunk_),
  bagRow(make_unique<BitMatrix>(treeChunk, nRow)),
  forest(make_unique<ForestTrain>(treeChunk)),
  predInfo(vector<double>(frameTrain->getNPred())),
  leaf(LFTrain::factoryReg(y, treeChunk)) {
}


unique_ptr<Train> Train::classification(const FrameTrain *frameTrain,
                                        const RankedSet *rankedPair,
                                        const unsigned int *yCtg,
                                        const double *yProxy,
                                        unsigned int nCtg,
                                        unsigned int treeChunk,
                                        unsigned int nTree) {
  auto trainCtg = make_unique<Train>(frameTrain, yCtg, nCtg, yProxy, nTree, treeChunk);
  trainCtg->trainChunk(frameTrain, rankedPair);

  return trainCtg;
}


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


void Train::trainChunk(const FrameTrain *frameTrain,
                       const RankedSet *rankedPair) {
  for (unsigned treeStart = 0; treeStart < treeChunk; treeStart += trainBlock) {
    unsigned int treeEnd = min(treeStart + trainBlock, treeChunk); // one beyond.
    auto treeBlock = blockProduce(frameTrain, rankedPair->getRowRank(), treeStart, treeEnd - treeStart);
    blockConsume(treeBlock, treeStart);
  }
  forest->splitUpdate(frameTrain, rankedPair->getNumRanked());
}


vector<TrainSet> Train::blockProduce(const FrameTrain *frameTrain,
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

  return move(block);
}

 
void Train::reserve(vector<TrainSet> &treeBlock) {
  size_t blockFac, blockBag, blockLeaf;
  size_t maxHeight = 0;
  (void) blockPeek(treeBlock, blockFac, blockBag, blockLeaf, maxHeight);
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
    leaf->blockLeaves(get<0>(trainSet).get(), leafMap, blockIdx++);
  }
}


void Train::cacheBagRaw(unsigned char *bbRaw) const {
  bagRow->Serialize(bbRaw);
}
