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
#include "predictorframe.h"
#include "ompthread.h"
#include "nodescorer.h"
#include "indexset.h"
#include "booster.h"

#include <numeric>


SampledObs::SampledObs(const Sampler* sampler,
		       unsigned int samplerIdx,
		       double (SampledObs::* adder_)(double, const SamplerNux&, PredictorT)) :
  nSamp(sampler->getNSamp()),
  nux(sampler->getSamples(samplerIdx)),
  bagCount(nux.size() == 0 ? nSamp : nux.size()),
  adder(adder_),
  bagSum(0.0),
  obs2Sample(vector<IndexT>(sampler->getNObs())),
  ctgRoot(vector<SumCount>(sampler->getNCtg())) {
}


SampledObs::~SampledObs() = default;


void SampledObs::sampleRoot(const PredictorFrame* frame,
			    NodeScorer* scorer) {
  bagSamples(frame);
  setRanks(frame);
  Booster::updateResidual(scorer, this, bagSum);
}


void SampledCtg::bagSamples(const PredictorFrame* frame) {
  bagSamples(frame, response->getYCtg(), response->getClassWeight());
}


void SampledReg::bagSamples(const PredictorFrame* frame) {
  bagSamples(frame, response->getYTrain());
}


SampledReg::SampledReg(const Sampler* sampler,
		       const ResponseReg* response_,
		       unsigned int tIdx) :
  SampledObs(sampler, tIdx, static_cast<double (SampledObs::*)(double, const SamplerNux&, PredictorT)>(&SampledReg::addNode)),
  response(response_) {
}


SampledReg::~SampledReg() = default;


void SampledReg::bagSamples(const PredictorFrame* frame,
			    const vector<double>& y) {
  SampledObs::bagSamples(y, vector<PredictorT>(y.size()));
}


void SampledCtg::bagSamples(const PredictorFrame* frame,
			    const vector<PredictorT>& yCtg,
			    const vector<double>& y) {
  SampledObs::bagSamples(y, yCtg);
}


SampledCtg::SampledCtg(const Sampler* sampler,
		       const ResponseCtg* response_,
		       unsigned int tIdx) :
  SampledObs(sampler, tIdx, static_cast<double (SampledObs::*)(double, const SamplerNux&, PredictorT)>(&SampledCtg::addNode)),
  response(response_) {
  SumCount scZero;
  fill(ctgRoot.begin(), ctgRoot.end(), scZero);
}

SampledCtg::~SampledCtg() = default;


void SampledObs::bagSamples(const vector<double>&  y,
			    const vector<PredictorT>& yCtg) {
  if (nux.empty()) {
    bagTrivial(y, yCtg);
    return;
  }
  else {
    IndexT sIdx = 0;
    IndexT row = 0;
    fill(obs2Sample.begin(), obs2Sample.end(), bagCount);
    for (const SamplerNux& nx : nux) {
      row += nx.getDelRow();
      bagSum += (this->*adder)(y[row], nx, yCtg[row]);
      obs2Sample[row] = sIdx++;
    }
  }
}


void SampledObs::bagTrivial(const vector<double>& y,
			    const vector<PredictorT>& yCtg) {
  iota(obs2Sample.begin(), obs2Sample.end(), 0);
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
  const vector<IndexT>& obs2Rank = layout->getRanks(predIdx);
  IndexT sIdx = 0;
  vector<unsigned char> rankSeen(obs2Rank.size());
  for (IndexT row = 0; row != obs2Rank.size(); row++) {
    if (obs2Sample[row] < bagCount) {
      IndexT rank = obs2Rank[row];
      sampledRanks[sIdx++] = rank;
      rankSeen[rank] = 1;
    }
  }
  runCount[predIdx] = accumulate(rankSeen.begin(), rankSeen.end(), 0);

  return sampledRanks;
}
