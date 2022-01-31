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
#include "sample.h"
#include "trainframe.h"

#include <numeric>


Sample::Sample(const TrainFrame* frame,
	       const Sampler* sampler,
	       double (Sample::* adder_)(IndexT, double, IndexT, PredictorT)) :
  nSamp(sampler->getNSamp()),
  bagging(sampler->isBagging()),
  sampledRows(sampler->getSampledRows()),
  adder(adder_),
  ctgRoot(vector<SumCount>(sampler->getNCtg())),
  row2Sample(vector<IndexT>(frame->getNRow())),
  bagSum(0.0) {
}

    
unique_ptr<SampleCtg> Sample::factoryCtg(const class Sampler* sampler,
					 const vector<double>& y,
                                         const TrainFrame* frame,
                                         const vector<PredictorT>& yCtg) {
  unique_ptr<SampleCtg> sampleCtg = make_unique<SampleCtg>(frame, sampler);
  sampleCtg->bagSamples(yCtg, y);

  return sampleCtg;
}


unique_ptr<SampleReg> Sample::factoryReg(const Sampler* sampler,
					 const vector<double>& y,
                                         const TrainFrame* frame) {
  unique_ptr<SampleReg> sampleReg = make_unique<SampleReg>(frame, sampler);
  sampleReg->bagSamples(y);
  return sampleReg;
}


SampleReg::SampleReg(const TrainFrame *frame,
		     const Sampler* sampler) :
  Sample(frame, sampler, static_cast<double (Sample::*)(IndexT, double, IndexT, PredictorT)>(&SampleReg::addNode)) {
}



void SampleReg::bagSamples(const vector<double>& y) {
  vector<PredictorT> ctgProxy(row2Sample.size());
  Sample::bagSamples(y, ctgProxy);
}


SampleCtg::SampleCtg(const TrainFrame* frame,
		     const Sampler* sampler) :
  Sample(frame, sampler, static_cast<double (Sample::*)(IndexT, double, IndexT, PredictorT)>(&SampleCtg::addNode)) {
  SumCount scZero;
  fill(ctgRoot.begin(), ctgRoot.end(), scZero);
}


// Same as for regression case, but allocates and sets 'ctg' value, as well.
// Full row count is used to avoid the need to rewalk.
//
void SampleCtg::bagSamples(const vector<PredictorT>& yCtg,
			   const vector<double>& y) {
  Sample::bagSamples(y, yCtg);
}


void Sample::bagSamples(const vector<double>&  y,
			const vector<PredictorT>& yCtg) {
  if (!bagging) {
    bagTrivial(y, yCtg);
    return;
  }

  bagCount = 0;
  IndexT countMax = 0;
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
