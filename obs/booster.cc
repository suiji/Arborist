// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file booster.cc

   @brief Paramtrized treatment of tree boosting.

   @author Mark Seligman
 */

#include "booster.h"
#include "indexset.h"
#include "scoredesc.h"
#include "sampledobs.h"
#include "sampler.h"
#include "response.h"
#include "samplemap.h"
#include "nodescorer.h"

#include <cmath>


unique_ptr<Booster> Booster::booster = nullptr;
bool Booster::trackFit = false;
unsigned int Booster::stopLag = 0;


Booster::Booster(double (Booster::* baseScorer_)(const Response*) const,
		 void (Booster::* updater_)(NodeScorer*, SampledObs*, double&),
		 double nu) :
  scoreDesc(ScoreDesc(nu)),
  baseScorer(baseScorer_),
  updater(updater_) {
}


double Booster::zero(const Response* response) const {
  return 0.0;
}


void Booster::noUpdate(NodeScorer* nodeScorer,
		       SampledObs* sampledObs,
		       double& bagSum) {
}


void Booster::setEstimate(const Sampler* sampler) {
  if (boosting()) {
    booster->baseEstimate(sampler);
  }
}


void Booster::updateResidual(NodeScorer* nodeScorer,
			     SampledObs* sampledObs,
			     double& bagSum) {
  if (boosting()) {
    booster->update(nodeScorer, sampledObs, bagSum);
  }
}


double Booster::mean(const Response* response) const {
  return reinterpret_cast<const ResponseReg*>(response)->mean();
}


void Booster::baseEstimate(const Sampler* sampler) {
  scoreDesc.baseScore = (this->*baseScorer)(sampler->getResponse());
  estimate = vector<double>(sampler->getNObs(), scoreDesc.baseScore);
}


void Booster::updateL2(NodeScorer* nodeScorer,
		       SampledObs* sampledObs,
		       double& bagSum) {
  bagSum = 0.0;
  IndexT obsIdx = 0;
  for (const double& est : estimate) {
    IndexT sIdx;
    SampleNux* nux;
    if (sampledObs->isSampled(obsIdx++, sIdx, nux)) {
      bagSum += nux->decrementSum(est);
    }
  }
}


double Booster::logit(const Response* response) const {
  vector<double> binaryProb = reinterpret_cast<const ResponseCtg*>(response)->ctgProb();
  return log(binaryProb[1] / binaryProb[0]);
}


void Booster::updateLogOdds(NodeScorer* nodeScorer,
			    SampledObs* sampledObs,
			    double& bagSum) {
  bagSum = 0.0;
  vector<double> pq(sampledObs->getBagCount());
  IndexT obsIdx = 0;
  for (const double& est : estimate) {
    IndexT sIdx;
    SampleNux* nux;
    if (sampledObs->isSampled(obsIdx++, sIdx, nux)) {
      double prob = 1.0 / (1.0 + exp(-est)); // logistic
      bagSum += nux->decrementSum(prob);
      pq[sIdx] = prob * (1.0 - prob) * nux->getSCount();
    }
  }

  nodeScorer->setGamma(std::move(pq));
}


void Booster::updateEstimate(const SampledObs* sampledObs,
			     const PreTree*  pretree,
			     const SampleMap& terminalMap) {
  if (boosting()) {
    booster->scoreSamples(sampledObs, pretree, terminalMap);
  }
}


void Booster::scoreSamples(const SampledObs* sampledObs,
			   const PreTree* preTree,
			   const SampleMap& terminalMap) {
  vector<double> sampleScore = terminalMap.scaleSampleScores(sampledObs, preTree, scoreDesc.nu);
  IndexT obsIdx = 0;
  for (double& est : estimate) {
    IndexT sIdx;
    if (sampledObs->isSampled(obsIdx++, sIdx)) {
      est += sampleScore[sIdx];
    }
  }
}


void Booster::init(const string& loss,
		    const string& scorer,
		    double nu) {
  if (loss == "l2") {
    booster = make_unique<Booster>(&Booster::mean, &Booster::updateL2, nu);
  }
  else if (loss == "logistic") {
    booster = make_unique<Booster>(&Booster::logit, &Booster::updateLogOdds, nu);
  }
  else {
    booster = make_unique<Booster>(&Booster::zero, &Booster::noUpdate, 0.0);
  }
  booster->scoreDesc.scorer = scorer;
}


void Booster::init(const string& loss,
		   const string& scorer,
		   double nu,
		   bool trackFit,
		   unsigned int stopLag) {
  init(loss, scorer, nu);
  Booster::trackFit = trackFit;
  Booster::stopLag = stopLag;
}


void Booster::deInit() {
  booster = nullptr;
}
