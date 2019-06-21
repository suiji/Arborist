// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file trainbridge.cc

   @brief Exportable classes and methods from the Train class.

   @author Mark Seligman
*/

#include "trainbridge.h"

#include "forest.h"
#include "leaf.h"
#include "train.h"

TrainBridge::TrainBridge(unique_ptr<Train> train_) : train(move(train_)) {
}


TrainBridge::~TrainBridge() {
}


unique_ptr<TrainBridge> TrainBridge::classification(const class SummaryFrame* frame,
                                                    const unsigned int *yCtg,
                                                    const double *yProxy,
                                                    unsigned int nCtg,
                                                    unsigned int treeChunk,
                                                    unsigned int nTree) {
  auto train = Train::classification(frame, yCtg, yProxy, nCtg, treeChunk, nTree);

  return make_unique<TrainBridge>(move(train));
}

unique_ptr<TrainBridge> TrainBridge::regression(const class SummaryFrame* frame,
                                                const double* y,
                                                unsigned int treeChunk) {
  auto train = Train::regression(frame, y, treeChunk);
  return make_unique<TrainBridge>(move(train));
}


void TrainBridge::writeHeight(unsigned int height[], unsigned int tIdx) const {
  unsigned int idx = tIdx;
  for (auto th : getLeafHeight()) {
    height[idx++] = th + (tIdx == 0 ? 0 : height[tIdx - 1]);
  }
}

void TrainBridge::writeBagHeight(unsigned int bagHeight[], unsigned int tIdx) const {
  unsigned int idx = tIdx;
  for (auto th : leafBagHeight()) {
    bagHeight[idx++] = th + (tIdx == 0 ? 0 : bagHeight[tIdx - 1]);
  }
}


bool TrainBridge::leafFits(unsigned int height[], unsigned int tIdx, size_t capacity, size_t& offset, size_t& bytes) const {
  offset = tIdx == 0 ? 0 : height[tIdx - 1] * sizeof(Leaf);
  bytes = getLeafHeight().back() * sizeof(Leaf);
  return offset + bytes <= capacity;
}


bool TrainBridge::bagSampleFits(unsigned int height[], unsigned int tIdx, size_t capacity, size_t& offset, size_t& bytes) const {
  offset = tIdx == 0 ? 0 : height[tIdx - 1] * sizeof(BagSample);
  bytes = leafBagHeight().back() * sizeof(BagSample);
  return offset + bytes <= capacity;
}


const vector<size_t>& TrainBridge::getForestHeight() const {
  return train->getForest()->getNodeHeight();
}


const vector<size_t>& TrainBridge::getFactorHeight() const {
  return train->getForest()->getFacHeight();
}


void TrainBridge::dumpTreeRaw(unsigned char treeOut[]) const {
  train->getForest()->cacheNodeRaw(treeOut);
}


void TrainBridge::dumpFactorRaw(unsigned char facOut[]) const {
  train->getForest()->cacheFacRaw(facOut);
}


const vector<size_t>& TrainBridge::getLeafHeight() const {
  return train->getLeaf()->getLeafHeight();
}


void TrainBridge::dumpLeafRaw(unsigned char leafOut[]) const {
  train->getLeaf()->cacheNodeRaw(leafOut);
}


const vector<size_t>& TrainBridge::leafBagHeight() const {
  return train->getLeaf()->getBagHeight();
}


void TrainBridge::dumpBagLeafRaw(unsigned char blOut[]) const {
  train->getLeaf()->cacheBLRaw(blOut);
}


size_t TrainBridge::getWeightSize() const {
  return train->getLeaf()->getWeightSize();
}


void TrainBridge::dumpLeafWeight(double weightOut[]) const {
  train->getLeaf()->dumpWeight(weightOut);
}


void TrainBridge::consumeBag() {
}


void TrainBridge::initBlock(unsigned int trainBlock) {
  Train::initBlock(trainBlock);
}


void TrainBridge::initCDF(const vector<double> &splitQuant){
  Train::initCDF(splitQuant);
}


void TrainBridge::initProb(unsigned int predFixed,
                           const vector<double> &predProb) {
  Train::initProb(predFixed, predProb);
}


void TrainBridge::initTree(unsigned int nSamp,
                           unsigned int minNode,
                           unsigned int leafMax) {
  Train::initTree(nSamp, minNode, leafMax);
}

void TrainBridge::initOmp(unsigned int nThread) {
  Train::initOmp(nThread);
}
  

void TrainBridge::initSample(unsigned int nSamp) {
  Train::initSample(nSamp);
}

void TrainBridge::initCtgWidth(unsigned int ctgWidth) {
  Train::initCtgWidth(ctgWidth);
}


void TrainBridge::initSplit(unsigned int minNode,
                            unsigned int totLevels,
                            double minRatio) {
  Train::initSplit(minNode, totLevels, minRatio);
}
  

void TrainBridge::initMono(const class SummaryFrame* frame,
                           const vector<double> &regMono) {
  Train::initMono(frame, regMono);
}

void TrainBridge::deInit() {
  Train::deInit();
}


void TrainBridge::dumpBagRaw(unsigned char bbRaw[]) const {
  train->cacheBagRaw(bbRaw);
}


const class LFTrain* TrainBridge::getLeaf() const {
  return train->getLeaf();
}


const vector<double>& TrainBridge::getPredInfo() const {
  return train->getPredInfo();
}
