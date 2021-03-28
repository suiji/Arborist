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

#include "sampler.h"
#include "leaf.h"
#include "train.h"
#include "rftrain.h"
#include "trainframe.h"


TrainBridge::TrainBridge(const RLEFrame* rleFrame, double autoCompress, bool enableCoproc, vector<string>& diag) : trainFrame(make_unique<TrainFrame>(rleFrame, autoCompress, enableCoproc, diag)) {
}


TrainBridge::~TrainBridge() {
}


vector<PredictorT> TrainBridge::getPredMap() const {
  vector<PredictorT> predMap(trainFrame->getPredMap());
  return predMap;
}


unique_ptr<TrainChunk> TrainBridge::classification(const vector<unsigned int>& yCtg,
                                                   const vector<double>& classWeight,
                                                   unsigned int nCtg,
                                                   unsigned int treeChunk,
                                                   unsigned int nTree) const {
  auto train = Train::classification(trainFrame.get(), yCtg, classWeight, nCtg, treeChunk, nTree);

  return make_unique<TrainChunk>(move(train));
}


unique_ptr<TrainChunk> TrainBridge::regression(const vector<double>& y,
                                               unsigned int treeChunk) const {
  auto train = Train::regression(trainFrame.get(), y, treeChunk);
  return make_unique<TrainChunk>(move(train));
}


void TrainBridge::initBlock(unsigned int trainBlock) {
  Train::initBlock(trainBlock);
}


void TrainBridge::initProb(unsigned int predFixed,
                           const vector<double> &predProb) {
  RfTrain::initProb(predFixed, predProb);
}


void TrainBridge::initTree(unsigned int nSamp,
                           unsigned int minNode,
                           unsigned int leafMax) {
  RfTrain::initTree(nSamp, minNode, leafMax);
}

void TrainBridge::initOmp(unsigned int nThread) {
  RfTrain::initOmp(nThread);
}


void TrainBridge::initSample(unsigned int nSamp) {
  RfTrain::initSample(nSamp);
}

void TrainBridge::initCtgWidth(unsigned int ctgWidth) {
  RfTrain::initCtgWidth(ctgWidth);
}


void TrainBridge::initSplit(unsigned int minNode,
                            unsigned int totLevels,
                            double minRatio,
			    const vector<double>& feSplitQuant) {
  RfTrain::initSplit(minNode, totLevels, minRatio, feSplitQuant);
}
  

void TrainBridge::initMono(const vector<double> &regMono) {
  RfTrain::initMono(trainFrame.get(), regMono);
}


void TrainBridge::deInit() {
  RfTrain::deInit();
  Train::deInit();
}


TrainChunk::TrainChunk(unique_ptr<Train> train_) : train(move(train_)) {
}


TrainChunk::~TrainChunk() {
}


void TrainChunk::writeSamplerBlockHeight(vector<size_t>& outHeight, unsigned int tIdx) const {
  size_t baseHeight = tIdx == 0 ? 0 : outHeight[tIdx - 1];
  unsigned int idx = tIdx;
  for (auto th : samplerHeight()) {
    outHeight[idx++] = th + baseHeight;
  }
}


bool TrainChunk::samplerBlockFits(const vector<size_t>& height, unsigned int tIdx, size_t capacity, size_t& offset, size_t& bytes) const {
  offset = tIdx == 0 ? 0 : height[tIdx - 1] * sizeof(SamplerNux);
  bytes = samplerHeight().back() * sizeof(SamplerNux);
  return offset + bytes <= capacity;
}


const vector<size_t>& TrainChunk::getForestHeight() const {
  return train->getForest()->getNodeHeight();
}


const vector<size_t>& TrainChunk::getFactorHeight() const {
  return train->getForest()->getFacHeight();
}


void TrainChunk::dumpTreeRaw(unsigned char treeOut[]) const {
  train->getForest()->cacheNodeRaw(treeOut);
}


void TrainChunk::dumpFactorRaw(unsigned char facOut[]) const {
  train->getForest()->cacheFacRaw(facOut);
}


const vector<size_t>& TrainChunk::samplerHeight() const {
  return train->getSamplerHeight();
}


void TrainChunk::dumpSamplerBlockRaw(unsigned char blOut[]) const {
  train->cacheSamplerRaw(blOut);
}


const vector<double>& TrainChunk::getPredInfo() const {
  return train->getPredInfo();
}
