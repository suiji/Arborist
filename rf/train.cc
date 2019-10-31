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
#include "summaryframe.h"
#include "frontier.h"
#include "pretree.h"
#include "obspart.h"
#include "sfcart.h"
#include "splitnux.h"
#include "leaf.h"
#include "candrf.h"
#include "ompthread.h"
#include "coproc.h"

#include <algorithm>


unsigned int Train::trainBlock = 0;

void Train::initBlock(unsigned int trainBlock_) {
  trainBlock = trainBlock_;
}


void Train::initProb(PredictorT predFixed,
                     const vector<double> &predProb) {
  CandRF::init(predFixed, predProb);
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
                      double minRatio,
		      const vector<double>& feSplitQuant) {
  Frontier::immutables(minNode, totLevels);
  SplitNux::immutables(minRatio, feSplitQuant);
}


void Train::initCtgWidth(unsigned int ctgWidth) {
  SampleNux::immutables(ctgWidth);
}


void Train::initMono(const SummaryFrame* frame,
                     const vector<double> &regMono) {
  SFCartReg::immutables(frame, regMono);
}


void Train::deInit() {
  trainBlock = 0;
  SplitNux::deImmutables();
  Frontier::deImmutables();
  PreTree::deImmutables();
  Sample::deImmutables();
  SampleNux::deImmutables();
  CandRF::deInit();
  SFCartReg::deImmutables();
  OmpThread::deInit();
}


unique_ptr<Train> Train::regression(const SummaryFrame* frame,
                                    const double* y,
                                    unsigned int treeChunk) {
  auto trainReg = make_unique<Train>(frame, y, treeChunk);
  trainReg->trainChunk(frame);

  return trainReg;
}


Train::Train(const SummaryFrame* frame,
             const double* y,
             unsigned int treeChunk_) :
  cand(make_unique<CandRF>()),
  nRow(frame->getNRow()),
  treeChunk(treeChunk_),
  bagRow(make_unique<BitMatrix>(treeChunk, nRow)),
  forest(make_unique<ForestCresc<DecNode> >(treeChunk)),
  predInfo(vector<double>(frame->getNPred())),
  leaf(LFTrain::factoryReg(y, treeChunk)) {
}


unique_ptr<Train> Train::classification(const SummaryFrame* frame,
                                        const unsigned int* yCtg,
                                        const double* yProxy,
                                        unsigned int nCtg,
                                        unsigned int treeChunk,
                                        unsigned int nTree) {
  auto trainCtg = make_unique<Train>(frame, yCtg, nCtg, yProxy, nTree, treeChunk);
  trainCtg->trainChunk(frame);

  return trainCtg;
}


Train::Train(const SummaryFrame* frame,
             const unsigned int* yCtg,
             unsigned int nCtg,
             const double* yProxy,
             unsigned int nTree,
             unsigned int treeChunk_) :
  cand(make_unique<CandRF>()),
  nRow(frame->getNRow()),
  treeChunk(treeChunk_),
  bagRow(make_unique<BitMatrix>(treeChunk, nRow)),
  forest(make_unique<ForestCresc<DecNode> >(treeChunk)),
  predInfo(vector<double>(frame->getNPred())),
  leaf(LFTrain::factoryCtg(yCtg, yProxy, treeChunk, nRow, nCtg, nTree)) {
}


Train::~Train() {
}


void Train::trainChunk(const SummaryFrame* frame) {
  for (unsigned treeStart = 0; treeStart < treeChunk; treeStart += trainBlock) {
    unsigned int treeEnd = min(treeStart + trainBlock, treeChunk); // one beyond.
    auto treeBlock = blockProduce(frame, treeStart, treeEnd - treeStart);
    blockConsume(treeBlock, treeStart);
  }
  forest->splitUpdate(frame);
}


vector<TrainSet> Train::blockProduce(const SummaryFrame* frame,
                                     unsigned int tStart,
                                     unsigned int tCount) {
  unsigned int tIdx = tStart;
  vector<TrainSet> block(tCount);
  for (auto & set : block) {
    unique_ptr<Sample> sample(leaf->rootSample(frame, bagRow.get(), tIdx++));
    unique_ptr<PreTree> preTree(Frontier::oneTree(this, frame, sample.get()));
    set = make_pair(move(sample), move(preTree));
  }

  if (tStart == 0)
    reserve(block);

  return block;
}

 
void Train::blockConsume(vector<TrainSet>& treeBlock,
                         unsigned int blockStart) {
  unsigned int blockIdx = blockStart;
  for (auto & trainSet : treeBlock) {
    const vector<IndexT> leafMap = get<1>(trainSet)->consume(forest.get(), blockIdx, predInfo);
    leaf->blockLeaves(get<0>(trainSet).get(), leafMap, blockIdx++);
  }
}


void Train::reserve(vector<TrainSet>& treeBlock) {
  size_t blockFac;
  IndexT blockBag, blockLeaf;
  IndexT maxHeight = 0;
  (void) blockPeek(treeBlock, blockFac, blockBag, blockLeaf, maxHeight);
  PreTree::reserve(maxHeight);
}


unsigned int Train::blockPeek(vector<TrainSet>& treeBlock,
                              size_t& blockFac,
                              IndexT& blockBag,
                              IndexT& blockLeaf,
                              IndexT& maxHeight) {
  IndexT blockHeight = 0;
  blockLeaf = blockFac = blockBag = 0;
  for (auto & set : treeBlock) {
    get<1>(set)->blockBump(blockHeight, maxHeight, blockFac, blockLeaf, blockBag);
  }

  return blockHeight;
}

 
void Train::cacheBagRaw(unsigned char* bbRaw) const {
  bagRow->Serialize(bbRaw);
}


unique_ptr<SplitFrontier>
Train::splitFactory(const SummaryFrame* frame,
		    Frontier* frontier,
		    const Sample* sample,
		    PredictorT nCtg) const {
  return SFCart::splitFactory(cand.get(), frame, frontier, sample, nCtg);
}
