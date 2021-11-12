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
#include "callback.h"
#include "trainframe.h"

#include <numeric>


Sample::Sample(const TrainFrame* frame,
	       const Sampler* sampler) :
  nSamp(sampler->getNSamp()),
  bagging(sampler->isBagging()),
  ctgRoot(vector<SumCount>(SampleNux::getNCtg())),
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
		     const Sampler* sampler) : Sample(frame, sampler) {
}



void SampleReg::bagSamples(const vector<double>& y) {
  vector<PredictorT> ctgProxy(row2Sample.size());
  Sample::bagSamples(y, ctgProxy);
}


SampleCtg::SampleCtg(const TrainFrame* frame,
		     const Sampler* sampler) : Sample(frame, sampler) {
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

  // Samples row indices and counts occurrences.
  //
  const IndexT nRow = row2Sample.size();
  vector<IndexT> sCountRow(nRow);
  IndexT bagCount = countSamples(sCountRow, nSamp);

  // Copies contents of sampled outcomes and builds mapping vectors.
  //
  fill(row2Sample.begin(), row2Sample.end(), bagCount);
  delRow = vector<IndexT>(bagCount);
  IndexT sIdx = 0;
  IndexT rowPrev = 0;
  for (IndexT row = 0; row < nRow; row++) {
    if (sCountRow[row] > 0) {
      row2Sample[row] = sIdx;
      delRow[sIdx] = row - rowPrev;
      bagSum += addNode(delRow[sIdx], y[row], sCountRow[row], yCtg[row]);
      rowPrev = row;
      sIdx++;
    }
  }
}


void Sample::bagTrivial(const vector<double>& y,
			const vector<PredictorT>& yCtg) {
  IndexT bagCount = row2Sample.size();
  delRow = vector<IndexT>(bagCount);
  fill(delRow.begin() + 1, delRow.end(), 1); // Saturates bag row.
  iota(row2Sample.begin(), row2Sample.end(), 0);
  for (IndexT row = 0; row < bagCount; row++) {
    bagSum += addNode(delRow[row], y[row], 1, yCtg[row]);
  }
}


// Sample counting is sensitive to locality.  In the absence of
// binning, access is random.  Larger bins improve locality, but
// performance begins to degrade when bin size exceeds available
// cache.
IndexT Sample::countSamples(vector<IndexT>& sc,
			    IndexT nSamp) {
  vector<IndexT> idx(CallBack::sampleRows(nSamp));
  if (binIdx(sc.size()) > 0) {
    idx = binIndices(idx);
  }
    
  IndexT nz = 0;
  for (auto index : idx) {
    nz += (sc[index] == 0 ? 1 : 0);
    sc[index]++;
  }

  return nz;
}


vector<unsigned int> Sample::binIndices(const vector<unsigned int>& idx) {
  // Sets binPop to respective bin population, then accumulates population
  // of bins to the left.
  // Performance not sensitive to bin width.
  //
  vector<unsigned int> binPop(1 + binIdx(idx.size()));
  for (auto val : idx) {
    binPop[binIdx(val)]++;
  }
  for (unsigned int i = 1; i < binPop.size(); i++) {
    binPop[i] += binPop[i-1];
  }

  // Available index initialzed to one less than total population left of and
  // including bin.  Empty bins have same initial index as bin to the left.
  // This is not a problem, as empty bins are not (re)visited.
  //
  vector<int> idxAvail(binPop.size());
  for (unsigned int i = 0; i < idxAvail.size(); i++) {
    idxAvail[i] = static_cast<int>(binPop[i]) - 1;
  }

  // Writes to the current available index for bin, which is then decremented.
  //
  // Performance degrades if bin width exceeds available cache.
  //
  vector<unsigned int> idxBinned(idx.size());
  for (auto index : idx) {
    int destIdx = idxAvail[binIdx(index)]--;
    idxBinned[destIdx] = index;
  }

  return idxBinned;
}
