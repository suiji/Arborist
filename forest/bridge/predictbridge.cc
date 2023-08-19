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

#include "predictbridge.h"
#include "predict.h"
#include "quant.h"
#include "rleframe.h"
#include "ompthread.h"

// Type completion only:
#include "response.h"
#include "sampler.h"
#include "forestscorer.h"


PredictRegBridge::PredictRegBridge(unique_ptr<RLEFrame> rleFrame_,
				   ForestBridge forestBridge_,
				   SamplerBridge samplerBridge_,
				   LeafBridge leafBridge_,
				   //				   const pair<double, double>& scoreDesc,
				   vector<double> yTest,
				   unsigned int nPermute_,
				   bool indexing,
				   bool trapUnobserved,
				   unsigned int nThread,
				   vector<double> quantile) :
  PredictBridge(std::move(rleFrame_), std::move(forestBridge_), nPermute_, nThread),
  samplerBridge(std::move(samplerBridge_)),
  leafBridge(std::move(leafBridge_)),
  predictRegCore(make_unique<PredictReg>(forestBridge.getForest(), samplerBridge.getSampler(), leafBridge.getLeaf(), rleFrame.get(), /*ScoreDesc(scoreDesc),*/ std::move(yTest), PredictOption(nPermute, indexing, trapUnobserved), std::move(quantile))) {
}


PredictRegBridge::~PredictRegBridge() = default;


PredictCtgBridge::PredictCtgBridge(unique_ptr<RLEFrame> rleFrame_,
				   ForestBridge forestBridge_,
				   SamplerBridge samplerBridge_,
				   LeafBridge leafBridge_,
				   //				   const pair<double, double>& scoreDesc,
				   vector<unsigned int> yTest,
				   unsigned int nPermute_,
				   bool doProb,
				   bool indexing,
				   bool trapUnobserved,
				   unsigned int nThread) :
  PredictBridge(std::move(rleFrame_), std::move(forestBridge_), nPermute_, nThread),
  samplerBridge(std::move(samplerBridge_)),
  leafBridge(std::move(leafBridge_)),
  predictCtgCore(make_unique<PredictCtg>(forestBridge.getForest(), samplerBridge.getSampler(), rleFrame.get(), /*ScoreDesc(scoreDesc),*/ std::move(yTest), PredictOption(nPermute, indexing, trapUnobserved), doProb)) {
}


PredictCtgBridge::~PredictCtgBridge() = default;


PredictBridge::PredictBridge(unique_ptr<RLEFrame> rleFrame_,
                             ForestBridge forestBridge_,
			     unsigned int nPermute_,
			     unsigned int nThread) :
  rleFrame(std::move(rleFrame_)),
  forestBridge(std::move(forestBridge_)),
  nPermute(nPermute_) {
  OmpThread::init(nThread);
}


PredictBridge::~PredictBridge() {
  OmpThread::deInit();
}


size_t PredictBridge::getNRow() const {
  return rleFrame->getNRow();
}


unsigned int PredictBridge::getNTree() const {
  return forestBridge.getNTree();
}


bool PredictBridge::permutes() const {
  return nPermute > 0;
}


const vector<size_t>& PredictCtgBridge::getIndices() const {
  return predictCtgCore->getIndices();
}


const vector<size_t>& PredictRegBridge::getIndices() const {
  return predictRegCore->getIndices();
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


const vector<unsigned int>& PredictCtgBridge::getCensus() const {
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


const vector<double>& PredictRegBridge::getQPred() const {
  return predictRegCore->getQPred();
}


const vector<double>& PredictRegBridge::getQEst() const {
  return predictRegCore->getQEst();
}


vector<double> PredictBridge::forestWeight(const ForestBridge& forestBridge,
						   const SamplerBridge& samplerBridge,
						   const LeafBridge& leafBridge,
						   const double indices[],
						   size_t nObs,
						   unsigned int nThread) {
  return Predict::forestWeight(forestBridge.getForest(), samplerBridge.getSampler(), leafBridge.getLeaf(), nObs, indices, nThread);
}
