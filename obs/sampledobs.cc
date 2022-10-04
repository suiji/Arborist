// This file is part of ArboristCore.

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

#include <numeric>


SampledObs::SampledObs(const Sampler* sampler,
	       const Response* response,
	       double (SampledObs::* adder_)(double, const SamplerNux&, PredictorT)) :
  nSamp(sampler->getNSamp()),
  adder(adder_),
  ctgRoot(vector<SumCount>(response->getNCtg())),
  row2Sample(vector<IndexT>(sampler->getNObs())),
  bagSum(0.0) {
}

    
unique_ptr<SampleCtg> SampledObs::factoryCtg(const Sampler* sampler,
					     const Response* response,
					     const vector<double>& y,
					     const vector<PredictorT>& yCtg,
					     unsigned int tIdx) {
  unique_ptr<SampleCtg> sampleCtg = make_unique<SampleCtg>(sampler, response);
  sampleCtg->bagSamples(sampler, yCtg, y, tIdx);

  return sampleCtg;
}


unique_ptr<SampleReg> SampledObs::factoryReg(const Sampler* sampler,
					     const Response* response,
					     const vector<double>& y,
					     unsigned int tIdx) {
  unique_ptr<SampleReg> sampleReg = make_unique<SampleReg>(sampler, response);
  sampleReg->bagSamples(sampler, y, tIdx);
  return sampleReg;
}


SampleReg::SampleReg(const Sampler* sampler,
		     const Response* response) :
  SampledObs(sampler, response, static_cast<double (SampledObs::*)(double, const SamplerNux&, PredictorT)>(&SampleReg::addNode)) {
}



void SampleReg::bagSamples(const class Sampler* sampler,
			   const vector<double>& y,
			   unsigned int tIdx) {
  vector<PredictorT> ctgProxy(row2Sample.size());
  SampledObs::bagSamples(sampler, y, ctgProxy, tIdx);
}


SampleCtg::SampleCtg(const Sampler* sampler,
		     const Response* response) :
  SampledObs(sampler, response, static_cast<double (SampledObs::*)(double, const SamplerNux&, PredictorT)>(&SampleCtg::addNode)) {
  SumCount scZero;
  fill(ctgRoot.begin(), ctgRoot.end(), scZero);
}


// Same as for regression case, but allocates and sets 'ctg' value, as well.
// Full row count is used to avoid the need to rewalk.
//
void SampleCtg::bagSamples(const Sampler* sampler,
			   const vector<PredictorT>& yCtg,
			   const vector<double>& y,
			   unsigned int tIdx) {
  SampledObs::bagSamples(sampler, y, yCtg, tIdx);
}


void SampledObs::bagSamples(const Sampler* sampler,
			    const vector<double>&  y,
			    const vector<PredictorT>& yCtg,
			    unsigned int tIdx) {
  /*
  if (!sampler->isBagging()) { // Wrong test.
    bagTrivial(y, yCtg);
    return;
  }
  else {
  */
  IndexT sIdx = 0;
  IndexT row = 0;
  bagCount = sampler->getExtent(tIdx);
  fill(row2Sample.begin(), row2Sample.end(), bagCount);
  for (SamplerNux nux : sampler->getSamples(tIdx)) {
    row += nux.getDelRow();
    bagSum += (this->*adder)(y[row], nux, yCtg[row]);
    row2Sample[row] = sIdx++;
  }
 //}
}


void SampledObs::bagTrivial(const vector<double>& y,
			const vector<PredictorT>& yCtg) {
  bagCount = row2Sample.size();
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
