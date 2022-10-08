// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictbridge.cc

   @brief Exportable classes and methods from the Predict class.

   @author Mark Seligman
*/

#include "response.h"
#include "samplerbridge.h"
#include "sampler.h"
#include "leafbridge.h"
#include "predictbridge.h"
#include "predict.h"
#include "quant.h"
#include "forestbridge.h"
#include "forest.h"
#include "rleframe.h"
#include "ompthread.h"


PredictRegBridge::PredictRegBridge(unique_ptr<RLEFrame> rleFrame_,
				   unique_ptr<ForestBridge> forestBridge_,
				   unique_ptr<SamplerBridge> samplerBridge_,
				   unique_ptr<LeafBridge> leafBridge_,
				   vector<double> yTest,
				   unsigned int nPermute_,
				   bool trapUnobserved,
				   unsigned int nThread,
				   vector<double> quantile) :
  PredictBridge(std::move(rleFrame_), std::move(forestBridge_), nPermute_, nThread),
  samplerBridge(std::move(samplerBridge_)),
  leafBridge(std::move(leafBridge_)),
  predictRegCore(make_unique<PredictReg>(forestBridge->getForest(), samplerBridge->getSampler(), leafBridge->getLeaf(), rleFrame.get(), std::move(yTest), nPermute, std::move(quantile), trapUnobserved)) {
}


PredictRegBridge::~PredictRegBridge() {
}


PredictCtgBridge::PredictCtgBridge(unique_ptr<RLEFrame> rleFrame_,
				   unique_ptr<ForestBridge> forestBridge_,
				   unique_ptr<SamplerBridge> samplerBridge_,
				   unique_ptr<LeafBridge> leafBridge_,
				   vector<unsigned int> yTest,
				   unsigned int nPermute_,
				   bool doProb,
				   bool trapUnobserved,
				   unsigned int nThread) :
  PredictBridge(std::move(rleFrame_), std::move(forestBridge_), nPermute_, nThread),
  samplerBridge(std::move(samplerBridge_)),
  predictCtgCore(make_unique<PredictCtg>(forestBridge->getForest(), samplerBridge->getSampler(), rleFrame.get(), std::move(yTest), nPermute, doProb, trapUnobserved)) {
}


PredictCtgBridge::~PredictCtgBridge() {
}


PredictBridge::PredictBridge(unique_ptr<RLEFrame> rleFrame_,
                             unique_ptr<ForestBridge> forestBridge_,
			     unsigned int nPermute_,
			     unsigned int nThread) :
  rleFrame(std::move(rleFrame_)),
  forestBridge(std::move(forestBridge_)),
  nPermute(nPermute_) {
  Forest::init(rleFrame->getNPred());
  OmpThread::init(nThread);
}


PredictBridge::~PredictBridge() {
  Forest::deInit();
  OmpThread::deInit();
}


size_t PredictBridge::getNRow() const {
  return rleFrame->getNRow();
}


bool PredictBridge::permutes() const {
  return nPermute > 0;
}


void PredictRegBridge::predict() const {
  predictRegCore->predict(rleFrame.get());
}


void PredictCtgBridge::predict() const {
  predictCtgCore->predict(rleFrame.get());
}


const vector<unsigned int>& PredictCtgBridge::getYPred() const {
  return predictCtgCore->getYPred();
}


const vector<size_t>& PredictCtgBridge::getConfusion() const {
  return predictCtgCore->getConfusion();
}


const vector<double>& PredictCtgBridge::getMisprediction() const {
  return predictCtgCore->getMisprediction();
}


const vector<vector<double>>& PredictCtgBridge::getMispredPermuted() const {
  return predictCtgCore->getMispredPermuted();
}


double PredictCtgBridge::getOOBError() const {
  return predictCtgCore->getOOBError();
}


const vector<double>& PredictCtgBridge::getOOBErrorPermuted() const {
  return predictCtgCore->getOOBErrorPermuted();
}


unsigned int PredictCtgBridge::ctgIdx(unsigned int ctgTest,
				      unsigned int ctgPred) const {
  return predictCtgCore->ctgIdx(ctgTest, ctgPred);
}


const unsigned int* PredictCtgBridge::getCensus() const {
  return predictCtgCore->getCensus();
}


const vector<double>& PredictCtgBridge::getProb() const {
  return predictCtgCore->getProb();
}


double PredictRegBridge::getSAE() const {
  return predictRegCore->getSAE();
}


double PredictRegBridge::getSSE() const {
  return predictRegCore->getSSE();
}


const vector<double>& PredictRegBridge::getSSEPermuted() const {
  return predictRegCore->getSSEPermuted();
}


const vector<double>& PredictRegBridge::getYTest() const {
  return predictRegCore->getYTest();
}


const vector<double>& PredictRegBridge::getYPred() const {
  return predictRegCore->getYPred();
}


const vector<double> PredictRegBridge::getQPred() const {
  return predictRegCore->getQPred();
}


const vector<double> PredictRegBridge::getQEst() const {
  return predictRegCore->getQEst();
}
