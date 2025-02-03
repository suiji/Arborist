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
#include "fetrain.h"
#include "predictorframe.h"
#include "coproc.h"

TrainBridge::TrainBridge(unique_ptr<RLEFrame> rleFrame, double autoCompress, bool enableCoproc, vector<string>& diag) : frame(make_unique<PredictorFrame>(std::move(rleFrame), autoCompress, enableCoproc, diag)) {
  init(frame->getNPred());
}


TrainBridge::~TrainBridge() = default;


void TrainBridge::init(unsigned int nPred) {
  FETrain::initDecNode(nPred);
}


void TrainBridge::initGrove(bool thinLeaves,
			    unsigned int trainBlock) {
  FETrain::initGrove(thinLeaves, trainBlock);
}


void TrainBridge::initProb(unsigned int predFixed,
                           const vector<double> &predProb) {
  FETrain::initProb(predFixed, predProb);
}


void TrainBridge::initTree(size_t leafMax) {
  FETrain::initTree(leafMax);
}


void TrainBridge::initSamples(vector<double> obsWeight) {
  FETrain::initSamples(std::move(obsWeight));
}


void TrainBridge::initCtg(vector<double> classWeight) {
  FETrain::initCtg(std::move(classWeight));
}


void TrainBridge::initBooster(const string& loss, const string& scorer) {
  FETrain::initBooster(loss, scorer);
}


void TrainBridge::initBooster(const string& loss,
			      const string& scorer,
			      double nu,
			      bool trackFit,
			      unsigned int stopLag) {
  FETrain::initBooster(loss, scorer, nu, trackFit, stopLag);
}


void TrainBridge::getScoreDesc(double& nu,
			       double& baseScore,
			       string& forestScorer) {
  FETrain::listScoreDesc(nu, baseScore, forestScorer);
}


void TrainBridge::initNodeScorer(const string& scorer) {
  FETrain::initNodeScorer(scorer);
}


void TrainBridge::initSplit(unsigned int minNode,
                            unsigned int totLevels,
                            double minRatio,
			    const vector<double>& feSplitQuant) {
  FETrain::initSplit(minNode, totLevels, minRatio, feSplitQuant);
}
  

void TrainBridge::initMono(const vector<double> &regMono) {
  FETrain::initMono(frame.get(), regMono);
}


void TrainBridge::deInit() {
  FETrain::deInit();
}
