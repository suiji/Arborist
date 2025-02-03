// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file fetrain.cc

   @brief Bridge entry to static initializations.

   @author Mark Seligman
*/

#include "fetrain.h"
#include "fecore.h"
#include "bv.h"
#include "booster.h"
#include "grove.h"
#include "predictorframe.h"
#include "frontier.h"
#include "pretree.h"
#include "partition.h"
#include "sfcart.h"
#include "splitnux.h"
#include "sampledobs.h"
#include "algparam.h"
#include "coproc.h"

#include <algorithm>

void FETrain::initProb(PredictorT predFixed,
                     const vector<double> &predProb) {
  CandType::init(predFixed, predProb);
}


void FETrain::initTree(IndexT leafMax) {
  PreTree::init(leafMax);
}


void FETrain::initSplit(unsigned int minNode,
                      unsigned int totLevels,
                      double minRatio,
		      const vector<double>& feSplitQuant) {
  IndexSet::immutables(minNode);
  Frontier::immutables(totLevels);
  SplitNux::immutables(minRatio, feSplitQuant);
}


void FETrain::initMono(const PredictorFrame* frame,
		       const vector<double> &regMono) {
  SFRegCart::immutables(frame, regMono);
}


void FETrain::initBooster(const string& loss, const string& scorer) {
  Booster::init(loss, scorer);
}


void FETrain::initBooster(const string& loss, const string& scorer,
			  double nu,
			  bool trackFit,
			  unsigned int stopLag) {
  Booster::init(loss, scorer, nu, trackFit, stopLag);
}


void FETrain::initNodeScorer(const string& scorer) {
  NodeScorer::init(scorer);
}


void FETrain::initGrove(bool thinLeaves, unsigned int trainBlock) {
  Grove::init(thinLeaves, trainBlock);
}


void FETrain::initDecNode(unsigned int nPred) {
  DecNode::initMasks(nPred);
}


void FETrain::initSamples(vector<double> obsWeight) {
  SampledObs::init(std::move(obsWeight));
}


void FETrain::initCtg( vector<double> classWeight) {
  SampledCtg::init(std::move(classWeight));
}


void FETrain::listScoreDesc(double& nu,
			    double& baseScore,
			    string& forestScore) {
  Booster::listScoreDesc(nu, baseScore, forestScore);
}


void FETrain::deInit() {
  DecNode::deInit();
  Booster::deInit();
  Grove::deInit();
  NodeScorer::deInit();
  SplitNux::deImmutables();
  IndexSet::deImmutables();
  Frontier::deInit();
  PreTree::deInit();
  SampleNux::deImmutables();
  SampledObs::deInit();
  SamplerNux::unsetMasks();
  CandType::deInit();
  SFRegCart::deImmutables();
  FECore::deInit();
}
