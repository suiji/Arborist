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


#include "sampler.h"
#include "response.h"
#include "sample.h"

#include <numeric>


Sample::Sample(const Sampler* sampler,
	       const Response* response,
	       double (Sample::* adder_)(IndexT, double, IndexT, PredictorT)) :
  nSamp(sampler->getNSamp()),
  adder(adder_),
  ctgRoot(vector<SumCount>(response->getNCtg())),
  row2Sample(vector<IndexT>(sampler->getNObs())),
  bagSum(0.0) {
}

    
unique_ptr<SampleCtg> Sample::factoryCtg(const class Sampler* sampler,
					 const class Response* response,
					 const vector<double>& y,
                                         const vector<PredictorT>& yCtg) {
  unique_ptr<SampleCtg> sampleCtg = make_unique<SampleCtg>(sampler, response);
  sampleCtg->bagSamples(sampler, yCtg, y);

  return sampleCtg;
}


unique_ptr<SampleReg> Sample::factoryReg(const Sampler* sampler,
					 const class Response* response,
					 const vector<double>& y) {
  unique_ptr<SampleReg> sampleReg = make_unique<SampleReg>(sampler, response);
  sampleReg->bagSamples(sampler, y);
  return sampleReg;
}


SampleReg::SampleReg(const Sampler* sampler,
		     const Response* response) :
		     Sample(sampler, response, static_cast<double (Sample::*)(IndexT, double, IndexT, PredictorT)>(&SampleReg::addNode)) {
}



void SampleReg::bagSamples(const class Sampler* sampler,
			   const vector<double>& y) {
  vector<PredictorT> ctgProxy(row2Sample.size());
  Sample::bagSamples(sampler, y, ctgProxy);
}


SampleCtg::SampleCtg(const Sampler* sampler,
		     const Response* response) :
  Sample(sampler, response, static_cast<double (Sample::*)(IndexT, double, IndexT, PredictorT)>(&SampleCtg::addNode)) {
  SumCount scZero;
  fill(ctgRoot.begin(), ctgRoot.end(), scZero);
}


// Same as for regression case, but allocates and sets 'ctg' value, as well.
// Full row count is used to avoid the need to rewalk.
//
void SampleCtg::bagSamples(const Sampler* sampler,
			   const vector<PredictorT>& yCtg,
			   const vector<double>& y) {
  Sample::bagSamples(sampler, y, yCtg);
}


void Sample::bagSamples(const Sampler* sampler,
			const vector<double>&  y,
			const vector<PredictorT>& yCtg) {
  if (!sampler->isBagging()) {
    bagTrivial(y, yCtg);
    return;
  }

  bagCount = 0;
  IndexT countMax = 0;
  const vector<IndexT>& sampledRows = sampler->getSampledRows();
  for (IndexT count : sampledRows) {// Temporary.
    bagCount += (count > 0);
    if (count > countMax)
      countMax = count;
  }
  SampleNux::setShifts(getNCtg(), countMax);
  
  // Copies contents of sampled outcomes and builds mapping vectors.
  //
  fill(row2Sample.begin(), row2Sample.end(), bagCount);
  IndexT sIdx = 0;
  IndexT rowPrev = 0;
  for (IndexT row = 0; row < row2Sample.size(); row++) {
    if (sampledRows[row] > 0) {
      bagSum += (this->*adder)(row - exchange(rowPrev, row), y[row], sampledRows[row], yCtg[row]);
      row2Sample[row] = sIdx++;
    }
  }
}


void Sample::bagTrivial(const vector<double>& y,
			const vector<PredictorT>& yCtg) {
  SampleNux::setShifts(getNCtg(), 1);
  bagCount = row2Sample.size();
  iota(row2Sample.begin(), row2Sample.end(), 0);
  for (IndexT row = 0; row < bagCount; row++) {
    bagSum += (this->*adder)(1, y[row], 1, yCtg[row]);
  }
}
