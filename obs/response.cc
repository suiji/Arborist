// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file response.cc

   @brief Methods involving response type.

   @author Mark Seligman
 */

#include "predict.h"
#include "response.h"
#include "sampledobs.h"
#include "sampler.h"
#include "samplernux.h"
#include "quant.h"

#include "prng.h"
#include <algorithm>

ResponseReg::ResponseReg(const vector<double>& y) :
  Response(),
  yTrain(y),
  defaultPrediction(meanTrain()) {
}


double ResponseReg::getDefaultPrediction() const {
  return defaultPrediction;
}


ResponseCtg::ResponseCtg(const vector<PredictorT>& yCtg_,
			 PredictorT nCtg_,
			 const vector<double>& classWeight_) :
  Response(),
  yCtg(yCtg_),
  nCtg(nCtg_),
  classWeight(classWeight_),
  defaultPrediction(ctgDefault()) {
}


double ResponseCtg::getDefaultPrediction() const {
  return defaultPrediction; // upcasts.
}


ResponseCtg::ResponseCtg(const vector<PredictorT>& yCtg_,
			 PredictorT nCtg_) :
  Response(),
  yCtg(yCtg_),
  nCtg(nCtg_),
  classWeight(vector<double>(0)),
  defaultPrediction(ctgDefault()) {
}


unique_ptr<ResponseCtg> Response::factoryCtg(const vector<PredictorT>& yCtg,
				     PredictorT nCtg,
				     const vector<double>& classWeight) {
  return make_unique<ResponseCtg>(yCtg, nCtg, classWeight);
}


unique_ptr<ResponseCtg> Response::factoryCtg(const vector<PredictorT>& yCtg,
				     PredictorT nCtg) {
  return make_unique<ResponseCtg>(yCtg, nCtg);
}


unique_ptr<ResponseReg> Response::factoryReg(const vector<double>& yTrain) {
  return make_unique<ResponseReg>(yTrain);
}

  
PredictorT ResponseCtg::ctgDefault() const {
  vector<double> probDefault = ctgProb();
  return max_element(probDefault.begin(), probDefault.end()) - probDefault.begin();
}


vector<double> ResponseCtg::ctgProb() const {
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


unique_ptr<SampledObs> ResponseReg::getObs(const Sampler* sampler,
					  unsigned int tIdx) const {
  return make_unique<SampledReg>(sampler, this, tIdx);
}


unique_ptr<SampledObs> ResponseCtg::getObs(const Sampler* sampler,
					   unsigned int tIdx) const {
  return make_unique<SampledCtg>(sampler, this, tIdx);
}
