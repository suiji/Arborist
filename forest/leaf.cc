// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leaf.cc

   @brief Methods to train and score leaf components for crescent forest.

   @author Mark Seligman
 */

#include "predict.h"
#include "leaf.h"
#include "samplemap.h"
#include "sample.h"
#include "sampler.h"
#include "callback.h"
#include "ompthread.h"

#include <algorithm>


/**
   @brief Crescent constructor.
 */
Leaf::Leaf() {
}


LeafReg::LeafReg(const vector<double>& y_) :
  Leaf(),
  yTrain(y_),
  defaultPrediction(meanTrain()) {
}


LeafCtg::LeafCtg(const vector<PredictorT>& yCtg_,
		 PredictorT nCtg_,
		 const vector<double>& classWeight_) :
  Leaf(),
  yCtg(yCtg_),
  nCtg(nCtg_),
  classWeight(classWeight_),
  defaultPrediction(ctgDefault()) {
}


LeafCtg::LeafCtg(const vector<PredictorT>& yCtg_,
		 PredictorT nCtg_) :
  Leaf(),
  yCtg(yCtg_),
  nCtg(nCtg_),
  classWeight(vector<double>(0)),
  defaultPrediction(ctgDefault()) {
}


unique_ptr<LeafCtg> Leaf::factoryCtg(const vector<PredictorT>& yCtg,
				     PredictorT nCtg,
				     const vector<double>& classWeight) {
  return make_unique<LeafCtg>(yCtg, nCtg, classWeight);
}


unique_ptr<LeafCtg> Leaf::factoryCtg(const vector<PredictorT>& yCtg,
				     PredictorT nCtg) {
  return make_unique<LeafCtg>(yCtg, nCtg);
}


unique_ptr<LeafReg> Leaf::factoryReg(const vector<double>& yTrain) {
  return make_unique<LeafReg>(yTrain);
}

  
PredictorT LeafCtg::ctgDefault() const {
  vector<double> probDefault = defaultProb();
  return max_element(probDefault.begin(), probDefault.end()) - probDefault.begin();
}

  

unique_ptr<Sample> LeafReg::rootSample(const TrainFrame* frame,
				       const Sampler* sampler) const {
  return Sample::factoryReg(sampler, yTrain, frame);
}


unique_ptr<Sample> LeafCtg::rootSample(const TrainFrame* frame,
				       const Sampler* sampler) const {
  return Sample::factoryCtg(sampler, classWeight, frame, yCtg);
}


double LeafReg::predictObs(const Predict* predict, size_t row) const {
  double sumScore = 0.0;
  unsigned int nEst = 0;
  for (unsigned int tIdx = 0; tIdx < predict->getNTree(); tIdx++) {
    double score;
    if (predict->isLeafIdx(row, tIdx, score)) {
      nEst++;
      sumScore += score;
    }
  }
  return nEst > 0 ? sumScore / nEst : defaultPrediction;
}


PredictorT LeafCtg::predictObs(const Predict* predict, size_t row, PredictorT* census) const {
  unsigned int nEst = 0;
  vector<double> ctgJitter(nCtg);
  for (unsigned int tIdx = 0; tIdx < predict->getNTree(); tIdx++) {
    double score;
    if (predict->isLeafIdx(row, tIdx, score)) {
      nEst++;
      PredictorT ctg = floor(score); // Truncates jittered score for indexing.
      census[ctg]++;
      ctgJitter[ctg] += score - ctg; // Accumulates category jitters.
    }
  }
  if (nEst == 0) { // Default category unity, all others zero.
    census[defaultPrediction] = 1;
  }

  return argMaxJitter(census, &ctgJitter[0]);
}


PredictorT LeafCtg::argMaxJitter(const IndexT* census,
				 const double* ctgJitter) const {
  PredictorT argMax = 0;
  IndexT countMax = 0;
  // Assumes at least one slot has nonzero count.
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    IndexT count = census[ctg];
    if (count == 0)
      continue;
    else if (count > countMax) {
      countMax = count;
      argMax = ctg;
    }
    else if (count == countMax) {
      if (ctgJitter[ctg] > ctgJitter[argMax]) {
	argMax = ctg;
      }
    }
  }
  return argMax;
}


CtgProb::CtgProb(const Predict* predict,
		 const LeafCtg* leaf,
		 const class Sampler* sampler,
		 bool doProb) :
  nCtg(leaf->getNCtg()),
  probDefault(leaf->defaultProb()),
  probs(vector<double>(doProb ? predict->getNRow() * nCtg : 0)) {
}


void CtgProb::predictRow(const Predict* predict, size_t row, PredictorT* ctgRow) {
  unsigned int nEst = accumulate(ctgRow, ctgRow + nCtg, 0ul);
  double* probRow = &probs[row * nCtg];
  if (nEst == 0) {
    applyDefault(probRow);
  }
  else {
    double scale = 1.0 / nEst;
    for (PredictorT ctg = 0; ctg < nCtg; ctg++)
      probRow[ctg] = ctgRow[ctg] * scale;
  }
}


vector<double> LeafCtg::defaultProb() const {
  // Uses the ECDF as the default distribution.
  vector<IndexT> ctgTot(nCtg);
  for (auto ctg : yCtg) {
    ctgTot[ctg]++;
  }

  vector<double> ctgDefault(nCtg);
  double scale = 1.0 / yCtg.size();
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    ctgDefault[ctg] = ctgTot[ctg] * scale;
  }
  return ctgDefault;
}


void CtgProb::applyDefault(double probPredict[]) const {
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    probPredict[ctg] = probDefault[ctg];
  }
}


void CtgProb::dump() const {
  // TODO
}
