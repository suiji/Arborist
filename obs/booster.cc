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
#include "samplemap.h"
#include "nodescorer.h"

#include <cmath>


unique_ptr<Booster> Booster::booster = nullptr;


Booster::Booster(double (Booster::* baseScorer_)(const IndexSet&) const,
		 void (Booster::* updater_)(NodeScorer*, SampledObs*, double&),
		 double nu) :
  scoreDesc(ScoreDesc(nu)),
  baseScorer(baseScorer_),
  updater(updater_) {
}


void Booster::makeZero() {
  booster = make_unique<Booster>(&Booster::zero, &Booster::noUpdate, 0.0);
}


void Booster::setMean() {
  booster->scoreDesc.scorer = "mean";
}


void Booster::setPlurality() {
  booster->scoreDesc.scorer = "plurality";
}


void Booster::setSum() {
  booster->scoreDesc.scorer = "sum";
}


void Booster::setLogistic() {
  booster->scoreDesc.scorer = "logistic";
}


double Booster::zero(const IndexSet& iRoot) const {
  return 0.0;
}


void Booster::noUpdate(NodeScorer* nodeScorer,
		       SampledObs* sampledObs,
		       double& bagSum) {
}


void Booster::setEstimate(const SampledObs* sampledObs) {
  if (boosting())
    booster->baseEstimate(sampledObs);
}


void Booster::updateResidual(NodeScorer* nodeScorer,
			     SampledObs* sampledObs,
			     double& bagSum) {
  if (boosting()) {
    booster->update(nodeScorer, sampledObs, bagSum);
  }
}


void Booster::makeL2(double nu) {
  booster =  make_unique<Booster>(&Booster::mean, &Booster::updateL2, nu);
}


double Booster::mean(const IndexSet& iRoot) const {
  return iRoot.getSum() / iRoot.getSCount();
}


void Booster::baseEstimate(const SampledObs* sampledObs) {
  baseSamples = sampledObs->getSamples();
  scoreDesc.baseScore = (this->*baseScorer)(IndexSet(sampledObs));
  estimate = vector<double>(sampledObs->getBagCount(), scoreDesc.baseScore);
}


void Booster::updateL2(NodeScorer* nodeScorer,
		       SampledObs* sampledObs,
		       double& bagSum) {
  IndexT sIdx = 0;
  bagSum = 0.0;
  vector<SampleNux> residual = baseSamples;
  for (SampleNux& nux : residual) { // sCount applied internally.
    bagSum += nux.decrementSum(estimate[sIdx++]);
  }
  sampledObs->setSamples(std::move(residual));
}


void Booster::makeLogOdds(double nu) {
  booster =  make_unique<Booster>(&Booster::logit, &Booster::updateLogOdds, nu);
}


double Booster::logit(const IndexSet& iRoot) const {
  return log(iRoot.getCategoryCount(1) / double(iRoot.getCategoryCount(0)));
}


void Booster::updateLogOdds(NodeScorer* nodeScorer,
			    SampledObs* sampledObs,
			    double& bagSum) {
  IndexT sIdx = 0;
  bagSum = 0.0;
  vector<SampleNux> residual = baseSamples;
  vector<double> p = logistic(estimate);
  vector<double> pq = scaleComplement(p);
  for (SampleNux& nux : residual) {
    bagSum += nux.decrementSum(p[sIdx]); // sCount applied internally.
    pq[sIdx] *= nux.getSCount();
    sIdx++;
  }
  sampledObs->setSamples(std::move(residual));
  nodeScorer->setGamma(std::move(pq));
}


vector<double> Booster::scaleComplement(const vector<double>& p) {
  vector<double> pq(p.size());
  for (IndexT i = 0; i != p.size(); i++) {
    pq[i] = p[i] * (1.0 - p[i]);
  }
  
  return pq;
}


vector<double> Booster::logistic(const vector<double>& logOdds) {
  vector<double> p(logOdds.size());
  for (IndexT i = 0; i != logOdds.size(); i++) {
    p[i] = 1.0 / (1.0 + exp(-logOdds[i]));
  }

  return p;
}


void Booster::updateEstimate(const PreTree*  pretree,
			     const SampleMap& terminalMap) {
  if (boosting()) {
    booster->scoreSamples(pretree, terminalMap);
  }
}


void Booster::scoreSamples(const PreTree*  pretree,
			   const SampleMap& terminalMap) {
  terminalMap.scaleSampleScores(pretree, estimate, scoreDesc.nu);
}


void Booster::listScoreDesc(double& nu, double& baseScore, string& scorer) {
  nu = booster->scoreDesc.nu;
  baseScore = booster->scoreDesc.baseScore;
  scorer = booster->scoreDesc.scorer;
}


void Booster::deInit() {
  booster = nullptr;
}
