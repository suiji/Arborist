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
#include "bv.h"
#include "booster.h"
#include "grove.h"
#include "predictorframe.h"
#include "frontier.h"
#include "pretree.h"
#include "partition.h"
#include "sfcart.h"
#include "splitnux.h"
#include "algparam.h"
#include "ompthread.h"
#include "coproc.h"

#include <algorithm>

void FETrain::initProb(PredictorT predFixed,
                     const vector<double> &predProb) {
  CandType::init(predFixed, predProb);
}


void FETrain::initTree(IndexT leafMax) {
  PreTree::init(leafMax);
}


void FETrain::initOmp(unsigned int nThread) {
  OmpThread::init(nThread);
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


void FETrain::initBooster(const string& loss, const string& scorer, double nu) {
  Booster::init(loss, scorer, nu);
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
  SamplerNux::unsetMasks();
  CandType::deInit();
  SFRegCart::deImmutables();
  OmpThread::deInit();
}
