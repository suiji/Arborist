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

#include "forestbridge.h"
#include "trainbridge.h"
#include "samplerbridge.h"

#include "leaf.h"
#include "train.h"
#include "rftrain.h"
#include "trainframe.h"


TrainBridge::TrainBridge(const RLEFrame* rleFrame, double autoCompress, bool enableCoproc, vector<string>& diag) : trainFrame(make_unique<TrainFrame>(rleFrame, autoCompress, enableCoproc, diag)) {
  Forest::init(rleFrame->getNPred());
}


TrainBridge::~TrainBridge() {
}


vector<PredictorT> TrainBridge::getPredMap() const {
  vector<PredictorT> predMap(trainFrame->getPredMap());
  return predMap;
}


unique_ptr<TrainedChunk> TrainBridge::train(const ForestBridge& forest,
					  const SamplerBridge& sampler) const {
  auto trained = Train::train(trainFrame.get(), forest.getForest(), sampler.getSampler());

  return make_unique<TrainedChunk>(move(trained));
}


void TrainBridge::initBlock(unsigned int trainBlock) {
  Train::initBlock(trainBlock);
}


void TrainBridge::initProb(unsigned int predFixed,
                           const vector<double> &predProb) {
  RfTrain::initProb(predFixed, predProb);
}


void TrainBridge::initTree(IndexT leafMax) {
  RfTrain::initTree(leafMax);
}


void TrainBridge::initOmp(unsigned int nThread) {
  RfTrain::initOmp(nThread);
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
  Forest::deInit();
  RfTrain::deInit();
  Train::deInit();
}


TrainedChunk::TrainedChunk(unique_ptr<Train> train_) : train(move(train_)) {
}


TrainedChunk::~TrainedChunk() {
}


const vector<double>& TrainedChunk::getPredInfo() const {
  return train->getPredInfo();
}
