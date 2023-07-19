// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file fetrain.cc

   @brief Bridge entry to training.

   @author Mark Seligman
*/

#include "fetrain.h"
#include "bv.h"
#include "train.h"
#include "predictorframe.h"
#include "frontier.h"
#include "pretree.h"
#include "partition.h"
#include "sfcart.h"
#include "splitnux.h"
#include "sampler.h"
#include "candrf.h"
#include "ompthread.h"
#include "coproc.h"

#include <algorithm>

void FETrain::initProb(PredictorT predFixed,
                     const vector<double> &predProb) {
  CandRF::init(predFixed, predProb);
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


void FETrain::initSamples(double nu) {
  SampledObs::init(0.0);
}


void FETrain::deInit() {
  SplitNux::deImmutables();
  IndexSet::deImmutables();
  Frontier::deImmutables();
  PreTree::deInit();
  SampleNux::deImmutables();
  SampledObs::deInit();
  CandRF::deInit();
  SFRegCart::deImmutables();
  OmpThread::deInit();
}
