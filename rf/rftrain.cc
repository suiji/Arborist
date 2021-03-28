// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rftrain.cc

   @brief Bridge entry to training.

   @author Mark Seligman
*/

#include "rftrain.h"
#include "bv.h"
#include "sample.h"
#include "train.h"
#include "trainframe.h"
#include "frontier.h"
#include "pretree.h"
#include "obspart.h"
#include "sfcart.h"
#include "splitnux.h"
#include "sampler.h"
#include "leaf.h"
#include "candrf.h"
#include "ompthread.h"
#include "coproc.h"

#include <algorithm>

void RfTrain::initProb(PredictorT predFixed,
                     const vector<double> &predProb) {
  CandRF::init(predFixed, predProb);
}


void RfTrain::initTree(unsigned int nSamp,
                     unsigned int minNode,
                     unsigned int leafMax) {
  PreTree::immutables(nSamp, minNode, leafMax);
}


void RfTrain::initOmp(unsigned int nThread) {
  OmpThread::init(nThread);
}


void RfTrain::initSample(unsigned int nSamp) {
  Sample::immutables(nSamp);
}


void RfTrain::initSplit(unsigned int minNode,
                      unsigned int totLevels,
                      double minRatio,
		      const vector<double>& feSplitQuant) {
  IndexSet::immutables(minNode);
  Frontier::immutables(totLevels);
  SplitNux::immutables(minRatio, feSplitQuant);
}


void RfTrain::initCtgWidth(unsigned int ctgWidth) {
  SampleNux::immutables(ctgWidth);
}


void RfTrain::initMono(const TrainFrame* frame,
                     const vector<double> &regMono) {
  SFRegCart::immutables(frame, regMono);
}


void RfTrain::deInit() {
  SplitNux::deImmutables();
  IndexSet::deImmutables();
  Frontier::deImmutables();
  PreTree::deImmutables();
  Sample::deImmutables();
  SampleNux::deImmutables();
  CandRF::deInit();
  SFRegCart::deImmutables();
  OmpThread::deInit();
}
