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
#include "bagbridge.h"
#include "forestbridge.h"
#include "forest.h"
#include "leafbridge.h"
#include "leafpredict.h"
#include "rleframe.h"
#include "ompthread.h"


PredictRegBridge::PredictRegBridge(unique_ptr<RLEFrame> rleFrame_,
				   unique_ptr<ForestBridge> forestBridge_,
				   unique_ptr<BagBridge> bagBridge_,
				   unique_ptr<LeafBridge> leafBridge_,
				   vector<double> yTrain,
				     double meanTrain,
				     vector<double> yTest,
				     bool oob_,
				     unsigned int nPermute_,
				     unsigned int nThread,
				     vector<double> quantile) :
  PredictBridge(move(rleFrame_), move(forestBridge_), move(bagBridge_), move(leafBridge_), oob_, nPermute_, nThread),
  predictRegCore(make_unique<PredictReg>(bagBridge->getBag(), forestBridge->getForest(), leafBridge->getLeaf(), rleFrame.get(), move(yTrain), meanTrain, move(yTest), oob, nPermute, move(quantile))) {
}


PredictRegBridge::~PredictRegBridge() {
}


PredictCtgBridge::PredictCtgBridge(unique_ptr<RLEFrame> rleFrame_,
				   unique_ptr<ForestBridge> forestBridge_,
				   unique_ptr<BagBridge> bagBridge_,
				   unique_ptr<LeafBridge> leafBridge_,
				   const double* leafProb,
				   unsigned int nCtgTrain,
				   vector<unsigned int> yTest,
				   bool oob_,
				   unsigned int nPermute_,
				   bool doProb,
				   unsigned int nThread) :
  PredictBridge(move(rleFrame_), move(forestBridge_), move(bagBridge_), move(leafBridge_), oob_, nPermute_, nThread),
  predictCtgCore(make_unique<PredictCtg>(bagBridge->getBag(), forestBridge->getForest(), leafBridge->getLeaf(), rleFrame.get(), leafProb, nCtgTrain, move(yTest), oob, nPermute, doProb)) {
}


PredictCtgBridge::~PredictCtgBridge() {
}


PredictBridge::PredictBridge(unique_ptr<RLEFrame> rleFrame_,
                             unique_ptr<ForestBridge> forestBridge_,
                             unique_ptr<BagBridge> bagBridge_,
                             unique_ptr<LeafBridge> leafBridge_,
			     bool oob_,
			     unsigned int nPermute_,
			     unsigned int nThread) :
  rleFrame(move(rleFrame_)),
  bagBridge(move(bagBridge_)),
  forestBridge(move(forestBridge_)),
  leafBridge(move(leafBridge_)),
  oob(oob_),
  nPermute(nPermute_) {
  OmpThread::init(nThread);
}


PredictBridge::~PredictBridge() {
  OmpThread::deInit();
}


size_t PredictBridge::getNRow() const {
  return rleFrame->getNRow();
}


bool PredictBridge::permutes() const {
  return nPermute > 0;
}


void PredictRegBridge::predict() const {
  predictRegCore->predict();
}


void PredictCtgBridge::predict() const {
  predictCtgCore->predict();
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


const vector<vector<double>>& PredictCtgBridge::getMispredPermute() const {
  return predictCtgCore->getMispredPermute();
}


double PredictCtgBridge::getOOBError() const {
  return predictCtgCore->getOOBError();
}


const vector<double>& PredictCtgBridge::getOOBErrorPermute() const {
  return predictCtgCore->getOOBErrorPermute();
}


unsigned int PredictCtgBridge::getNCtgTrain() const {
  return predictCtgCore->getNCtgTrain();
}


LeafBridge* PredictBridge::getLeaf() const {
  return leafBridge.get();
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


const vector<double>& PredictRegBridge::getSSEPermute() const {
  return predictRegCore->getSSEPermute();
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
