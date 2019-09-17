// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitfrontier.cc

   @brief Methods to implement splitting of index-tree levels.

   @author Mark Seligman
 */


#include "frontier.h"
#include "sfcart.h"
#include "splitcand.h"
#include "splitnux.h"
#include "level.h"
#include "runset.h"
#include "samplenux.h"
#include "obspart.h"
#include "callback.h"
#include "summaryframe.h"
#include "rankedframe.h"
#include "sample.h"
#include "ompthread.h"

// Post-split consumption:
#include "pretree.h"

vector<double> SFCartReg::mono; // Numeric monotonicity constraints.


void SFCartReg::immutables(const SummaryFrame* frame,
                       const vector<double>& bridgeMono) {
  auto numFirst = frame->getNumFirst();
  auto numExtent = frame->getNPredNum();
  auto monoCount = count_if(bridgeMono.begin() + numFirst, bridgeMono.begin() + numExtent, [] (double prob) { return prob != 0.0; });
  if (monoCount > 0) {
    mono = vector<double>(frame->getNPredNum());
    mono.assign(bridgeMono.begin() + frame->getNumFirst(), bridgeMono.begin() + frame->getNumFirst() + frame->getNPredNum());
  }
}


void SFCartReg::deImmutables() {
  mono.clear();
}


SFCartReg::SFCartReg(const SummaryFrame* frame,
             Frontier* frontier,
	     const Sample* sample) :
  SplitFrontier(frame, frontier, sample),
  ruMono(vector<double>(0)) {
  run = make_unique<Run>(0, frame->getNRow());
}


/**
   @brief Constructor.
 */
SFCartCtg::SFCartCtg(const SummaryFrame* frame,
             Frontier* frontier,
	     const Sample* sample,
	     PredictorT nCtg_):
  SplitFrontier(frame, frontier, sample),
  nCtg(nCtg_) {
  run = make_unique<Run>(nCtg, frame->getNRow());
}


/**
   @brief Sets quick lookup offets for Run object.

   @return void.
 */
void SFCartReg::setRunOffsets(const vector<unsigned int>& runCount) {
  run->offsetsReg(runCount);
}


/**
   @brief Sets quick lookup offsets for Run object.
 */
void SFCartCtg::setRunOffsets(const vector<unsigned int>& runCount) {
  run->offsetsCtg(runCount);
}


double SFCartCtg::getSumSquares(const SplitCand *cand) const {
  return sumSquares[cand->getSplitCoord().nodeIdx];
}


const vector<double>& SFCartCtg::getSumSlice(const SplitCand* cand) {
  return ctgSum[cand->getSplitCoord().nodeIdx];
}


double* SFCartCtg::getAccumSlice(const SplitCand *cand) {
  return &ctgSumAccum[getNumIdx(cand->getSplitCoord().predIdx) * splitCount * nCtg + cand->getSplitCoord().nodeIdx * nCtg];
}

/**
   @brief Run objects should not be deleted until after splits have been consumed.
 */
void SFCartReg::clear() {
  SplitFrontier::clear();
}


SFCartReg::~SFCartReg() {
}


SFCartCtg::~SFCartCtg() {
}


void SFCartCtg::clear() {
  SplitFrontier::clear();
}


/**
   @brief Sets level-specific values for the subclass.
*/
void SFCartReg::levelPreset() {
  if (!mono.empty()) {
    ruMono = CallBack::rUnif(splitCount * mono.size());
  }
}


void SFCartCtg::levelPreset() {
  levelInitSumR(frame->getNPredNum());
  ctgSum = vector<vector<double> >(splitCount);

  sumSquares = frontier->sumsAndSquares(ctgSum);
}


void SFCartCtg::levelInitSumR(PredictorT nPredNum) {
  if (nPredNum > 0) {
    ctgSumAccum = vector<double>(nPredNum * nCtg * splitCount);
    fill(ctgSumAccum.begin(), ctgSumAccum.end(), 0.0);
  }
}


int SFCartReg::getMonoMode(const SplitCand* cand) const {
  if (mono.empty())
    return 0;

  PredictorT numIdx = getNumIdx(cand->getSplitCoord().predIdx);
  double monoProb = mono[numIdx];
  double prob = ruMono[cand->getSplitCoord().nodeIdx * mono.size() + numIdx];
  if (monoProb > 0 && prob < monoProb) {
    return 1;
  }
  else if (monoProb < 0 && prob < -monoProb) {
    return -1;
  }
  else {
    return 0;
  }
}


void SFCartCtg::split(SplitCand& cand) {
  cand.split(this);
}


void SFCartReg::split(SplitCand& cand) {
  cand.split(this);
}
