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
#include "samplerbridge.h"
#include "forestbridge.h"
#include "fepredict.h"
#include "forest.h"
#include "predict.h"
#include "sampler.h"

// Type completion only:
#include "dectree.h"
#include "response.h"
#include "quant.h"


unique_ptr<PredictRegBridge> PredictRegBridge::predict(const Sampler* sampler,
						       Forest* forest,
						       vector<double> yTest) {
  return make_unique<PredictRegBridge>(sampler->predictReg(forest, yTest));
}


PredictRegBridge::PredictRegBridge(unique_ptr<SummaryReg> summary_) :
  PredictBridge(),
  summary(std::move(summary_)) {
}


PredictRegBridge::~PredictRegBridge() = default;


unique_ptr<PredictCtgBridge> PredictCtgBridge::predict(const Sampler* sampler,
						       Forest* forest,
						       vector<unsigned int> yTest) {
  return make_unique<PredictCtgBridge>(sampler->predictCtg(forest, yTest));
}


PredictCtgBridge::PredictCtgBridge(unique_ptr<SummaryCtg> summary_) :
  PredictBridge(),
  summary(std::move(summary_)) {
}


PredictCtgBridge::~PredictCtgBridge() = default;


PredictBridge::PredictBridge() {
}


PredictBridge::~PredictBridge() {
  SamplerNux::unsetMasks();
}


void PredictBridge::initPredict(bool indexing,
				bool bagging,
				unsigned int nPermute,
				bool trapUnobserved) {
  FEPredict::initPredict(indexing, bagging, nPermute, trapUnobserved);
}


void PredictBridge::initQuant(vector<double> quantile) {
  FEPredict::initQuant(std::move(quantile));
}


void PredictBridge::initCtgProb(bool doProb) {
  FEPredict::initCtgProb(doProb);
}


vector<double> PredictBridge::forestWeight(const ForestBridge& forestBridge,
					   const SamplerBridge& samplerBridge,
					   const double indices[],
					   size_t nObs) {
  return Predict::forestWeight(forestBridge.getForest(), samplerBridge.getSampler(), nObs, indices);
}


bool PredictCtgBridge::permutes() const {
  return Predict::permutes();
}


bool PredictRegBridge::permutes() const {
  return Predict::permutes();
}


const vector<size_t>& PredictCtgBridge::getIndices() const {
  return summary->getIndices();
}


const vector<size_t>& PredictRegBridge::getIndices() const {
  return summary->getIndices();
}


size_t PredictRegBridge::getNObs() const {
  return summary->getNObs();
}


size_t PredictCtgBridge::getNObs() const {
  return summary->getNObs();
}


const vector<double>& PredictRegBridge::getYPred() const {
  return summary->getYPred();
}


const vector<unsigned int>& PredictCtgBridge::getYPred() const {
  return summary->getYPred();
}

// Classification summaries:

const vector<size_t>& PredictCtgBridge::getConfusion() const {
  return summary->getConfusion();
}


const vector<double>& PredictCtgBridge::getMisprediction() const {
  return summary->getMisprediction();
}


double PredictCtgBridge::getOOBError() const {
  return summary->getOOBError();
}


const vector<unsigned int>& PredictCtgBridge::getCensus() const {
  return summary->getCensus();
}


unsigned int PredictCtgBridge::ctgIdx(unsigned int ctgTest,
				      unsigned int ctgPred) const {
  return summary->ctgIdx(ctgTest, ctgPred);
}


// Classification auxiliaries:
const vector<double>& PredictCtgBridge::getProb() const {
  return summary->getProb();
}

// Regression statistics:
double PredictRegBridge::getSAE() const {
  return summary->getSAE();
}


double PredictRegBridge::getSSE() const {
  return summary->getSSE();
}


// Regression auxiliaries:
const vector<double>& PredictRegBridge::getQPred() const {
  return summary->getQPred();
}


const vector<double>& PredictRegBridge::getQEst() const {
  return summary->getQEst();
}


// Permutation summaries:
vector<vector<vector<double>>> PredictCtgBridge::getMispredPermuted() const {
  return summary->getMispredPermuted();
}


vector<vector<double>> PredictCtgBridge::getOOBErrorPermuted() const {
  return summary->getOOBErrorPermuted();
}


vector<vector<double>> PredictRegBridge::getSSEPermuted() const {
  return summary->getSSEPermuted();
}


vector<vector<double>> PredictRegBridge::getSAEPermuted() const {
  return summary->getSAEPermuted();
}
