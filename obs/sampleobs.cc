// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sample.cc

   @brief Methods for sampling from the response to begin training an individual tree.

   @author Mark Seligman
 */

#include "samplernux.h"
#include "sampler.h"
#include "response.h"
#include "sampleobs.h"

#include <numeric>


SampleObs::SampleObs(const Sampler* sampler,
	       const Response* response,
	       double (SampleObs::* adder_)(double, const SamplerNux&, PredictorT)) :
  nSamp(sampler->getNSamp()),
  adder(adder_),
  ctgRoot(vector<SumCount>(response->getNCtg())),
  row2Sample(vector<IndexT>(sampler->getNObs())),
  bagSum(0.0) {
}

    
unique_ptr<SampleCtg> SampleObs::factoryCtg(const Sampler* sampler,
					 const Response* response,
					 const vector<double>& y,
                                         const vector<PredictorT>& yCtg,
					 unsigned int tIdx) {
  unique_ptr<SampleCtg> sampleCtg = make_unique<SampleCtg>(sampler, response);
  sampleCtg->bagSamples(sampler, yCtg, y, tIdx);

  return sampleCtg;
}


unique_ptr<SampleReg> SampleObs::factoryReg(const Sampler* sampler,
					 const Response* response,
					 const vector<double>& y,
					 unsigned int tIdx) {
  unique_ptr<SampleReg> sampleReg = make_unique<SampleReg>(sampler, response);
  sampleReg->bagSamples(sampler, y, tIdx);
  return sampleReg;
}


SampleReg::SampleReg(const Sampler* sampler,
		     const Response* response) :
  SampleObs(sampler, response, static_cast<double (SampleObs::*)(double, const SamplerNux&, PredictorT)>(&SampleReg::addNode)) {
}



void SampleReg::bagSamples(const class Sampler* sampler,
			   const vector<double>& y,
			   unsigned int tIdx) {
  vector<PredictorT> ctgProxy(row2Sample.size());
  SampleObs::bagSamples(sampler, y, ctgProxy, tIdx);
}


SampleCtg::SampleCtg(const Sampler* sampler,
		     const Response* response) :
  SampleObs(sampler, response, static_cast<double (SampleObs::*)(double, const SamplerNux&, PredictorT)>(&SampleCtg::addNode)) {
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
  SampleObs::bagSamples(sampler, y, yCtg, tIdx);
}


void SampleObs::bagSamples(const Sampler* sampler,
			const vector<double>&  y,
			const vector<PredictorT>& yCtg,
			unsigned int tIdx) {
  /*
  if (!sampler->isBagging()) { // Wrong test.
    bagTrivial(y, yCtg);
    return;
  }
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
}


void SampleObs::bagTrivial(const vector<double>& y,
			const vector<PredictorT>& yCtg) {
  bagCount = row2Sample.size();
  iota(row2Sample.begin(), row2Sample.end(), 0);
  SamplerNux nux(1, 1);
  for (IndexT row = 0; row < bagCount; row++) {
    bagSum += (this->*adder)(y[row], nux, yCtg[row]);
  }
}
