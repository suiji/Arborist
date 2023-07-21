// This file is part of ArboristBase.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sampledobs.cc

   @brief Observations filtered by sampling.

   @author Mark Seligman
 */

#include "samplernux.h"
#include "sampler.h"
#include "response.h"
#include "sampledobs.h"
#include "frontier.h"
#include "predictorframe.h"
#include "ompthread.h"
#include "pretree.h"
#include "samplemap.h"

#include <numeric>

vector<SampleNux> SampledObs::sampleNux(0);
double SampledObs::bagSum = 0.0;
double SampledObs::learningRate = 0.0;
vector<double> SampledObs::preScore(0);
double SampledObs::rootScore = 0.0;
unsigned int SampledObs::seqIdx = 0;
vector<IndexT> SampledObs::row2Sample(0);
vector<vector<IndexT>> SampledObs::sample2Rank(0);
vector<IndexT> SampledObs::runCount(0);


void SampledObs::init(double nu) {
  learningRate = nu;
}


void SampledObs::deInit() {
  learningRate = 0.0;
  bagSum = 0.0;
  seqIdx = 0;
  rootScore = 0.0;
  preScore.clear();
  sampleNux.clear();
  row2Sample.clear();
  sample2Rank.clear();
  runCount.clear();
}


SampledObs::SampledObs(const Sampler* sampler,
		       const Response* response,
		       unsigned int samplerIdx,
		       double (SampledObs::* adder_)(double, const SamplerNux&, PredictorT)) :
  nSamp(sampler->getNSamp()),
  bagCount(sampler->getExtent(samplerIdx) == 0 ? nSamp : sampler->getExtent(samplerIdx)),
  adder(adder_),
  ctgRoot(vector<SumCount>(response->getNCtg())) {
  //  row2Sample(vector<IndexT>(sampler->getNObs())) {
}

    
unique_ptr<SampledCtg> SampledObs::factoryCtg(const Sampler* sampler,
					      const ResponseCtg* response,
					      unsigned int tIdx) {
  return make_unique<SampledCtg>(sampler, response, tIdx);
}


void SampledCtg::sampleRoot(const vector<SamplerNux>& nux,
			    const Frontier* frontier,
			    const PredictorFrame* frame) {
  bagSamples(response->getYCtg(), response->getClassWeight(), nux);
  setRanks(frame);
}


void SampledReg::sampleRoot(const vector<SamplerNux>& nux,
			    const Frontier* frontier,
			    const PredictorFrame* frame) {
  bagSamples(response->getYTrain(), frame, frontier, nux);
}


unique_ptr<SampledReg> SampledObs::factoryReg(const Sampler* sampler,
					      const ResponseReg* response,
					      unsigned int tIdx) {
  return make_unique<SampledReg>(sampler, response, tIdx);
}


SampledReg::SampledReg(const Sampler* sampler,
		       const ResponseReg* response_,
		       unsigned int tIdx) :
  SampledObs(sampler, response_, tIdx, static_cast<double (SampledObs::*)(double, const SamplerNux&, PredictorT)>(&SampledReg::addNode)),
  response(response_) {
}


void SampledReg::bagSamples(const vector<double>& y,
			    const PredictorFrame* frame,
			    const Frontier* frontier,
			    const vector<SamplerNux>& nux) {
  if (seqIdx == 0) { // Also true if nonsequential.
    SampledObs::bagSamples(y, vector<PredictorT>(y.size()), nux);
    setRanks(frame);
  }
  if (sequential()) {
    if (seqIdx == 0) {
      preScore = vector<double>(bagCount);
      rootScore = frontier->getRootScore(this);
      fill(preScore.begin(), preScore.end(), rootScore);
    }

    IndexT sIdx = 0;
    for (SampleNux& nux : sampleNux) { // sCount applied internally.
      nux.decrementSum(learningRate * preScore[sIdx++]);
    }
    bagSum *= (1.0 - learningRate);

    seqIdx++;
  }
}


void SampledObs::scoreSamples(const PreTree*  pretree,
			      const SampleMap& terminalMap) {
  if (sequential()) {
    preScore = terminalMap.scoreSamples(pretree);
  }
}



SampledCtg::SampledCtg(const Sampler* sampler,
		       const ResponseCtg* response_,
		       unsigned int tIdx) :
  SampledObs(sampler, response_, tIdx, static_cast<double (SampledObs::*)(double, const SamplerNux&, PredictorT)>(&SampledCtg::addNode)),
  response(response_) {
  SumCount scZero;
  fill(ctgRoot.begin(), ctgRoot.end(), scZero);
}


// Same as for regression case, but allocates and sets 'ctg' value, as well.
// Full row count is used to avoid the need to rewalk.
//
void SampledCtg::bagSamples(const vector<PredictorT>& yCtg,
			    const vector<double>& y,
			    const vector<SamplerNux>& nux) {
  SampledObs::bagSamples(y, yCtg, nux);
}


void SampledObs::bagSamples(const vector<double>&  y,
			    const vector<PredictorT>& yCtg,
			    const vector<SamplerNux>& nux) {
  bagSum = 0.0;
  sampleNux.clear();
  row2Sample = vector<IndexT>(y.size());
  if (nux.empty()) {
    bagTrivial(y, yCtg);
    return;
  }
  else {
    IndexT sIdx = 0;
    IndexT row = 0;
    fill(row2Sample.begin(), row2Sample.end(), bagCount);
    for (const SamplerNux& nx : nux) {
      row += nx.getDelRow();
      bagSum += (this->*adder)(y[row], nx, yCtg[row]);
      row2Sample[row] = sIdx++;
    }
  }
}


void SampledObs::bagTrivial(const vector<double>& y,
			    const vector<PredictorT>& yCtg) {
  iota(row2Sample.begin(), row2Sample.end(), 0);
  SamplerNux nux(1, 1);
  for (IndexT row = 0; row < bagCount; row++) {
    bagSum += (this->*adder)(y[row], nux, yCtg[row]);
  }
}


void SampledObs::setRanks(const PredictorFrame* layout) {
  sample2Rank = vector<vector<IndexT>>(layout->getNPred());
  runCount = vector<IndexT>(layout->getNPred());

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound predIdx = 0; predIdx < layout->getNPred(); predIdx++)
      sample2Rank[predIdx] = sampleRanks(layout, predIdx);
  }
}


vector<IndexT> SampledObs::sampleRanks(const PredictorFrame* layout, PredictorT predIdx) {
  vector<IndexT> sampledRanks(bagCount);
  const vector<IndexT>& row2Rank = layout->getRanks(predIdx);
  IndexT sIdx = 0;
  vector<unsigned char> rankSeen(row2Rank.size());
  for (IndexT row = 0; row != row2Rank.size(); row++) {
    if (row2Sample[row] < bagCount) {
      IndexT rank = row2Rank[row];
      sampledRanks[sIdx++] = rank;
      rankSeen[rank] = 1;
    }
  }
  runCount[predIdx] = accumulate(rankSeen.begin(), rankSeen.end(), 0);

  return sampledRanks;
}
