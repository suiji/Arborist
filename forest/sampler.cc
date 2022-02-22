// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#include "predict.h"
#include "sampler.h"
#include "callback.h"
#include "response.h"
#include "samplernux.h"

#include <cmath>


PackedT SamplerNux::delMask = 0;
unsigned int SamplerNux::rightBits = 0;


Sampler::Sampler(IndexT nSamp_,
		 IndexT nObs_,
		 unsigned int nTree_) :
    nTree(nTree_),
    nObs(nObs_),
    nSamp(nSamp_) {
  }

  
Sampler::Sampler(const vector<double>& yTrain,
		 IndexT nSamp_,
		 vector<vector<SamplerNux>> samples_) :
  nTree(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryReg(yTrain)),
  samples(samples_) {
}


Sampler::Sampler(const vector<PredictorT>& yTrain,
		 IndexT nSamp_,
		 vector<vector<SamplerNux>> samples_,
		 PredictorT nCtg,
		 const vector<double>& classWeight) :
  nTree(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryCtg(yTrain, nCtg, classWeight)),
  samples(move(samples_)) {
}


Sampler::Sampler(const vector<double>& yTrain,
		 vector<vector<SamplerNux>> samples_,
		 IndexT nSamp_,
		 bool bagging) :
  nTree(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryReg(yTrain)),
  samples(move(samples_)),
  bagMatrix(bagRows(bagging)) {
}


Sampler::Sampler(const vector<PredictorT>& yTrain,
		 vector<vector<SamplerNux>> samples_,
		 IndexT nSamp_,
		 PredictorT nCtg,
		 bool bagging) :
  nTree(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryCtg(yTrain, nCtg)),
  samples(move(samples_)),
  bagMatrix(bagRows(bagging)) {
}


Sampler::~Sampler() {
}


unique_ptr<BitMatrix> Sampler::bagRows(bool bagging) {
  if (!bagging)
    return make_unique<BitMatrix>(0, 0);

  unique_ptr<BitMatrix> matrix = make_unique<BitMatrix>(nTree, nObs);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT row = 0;
    for (IndexT sIdx = 0; sIdx != getBagCount(tIdx); sIdx++) {
      row += getDelRow(tIdx, sIdx);
      matrix->setBit(tIdx, row);
    }
  }
  return matrix;
}


void Sampler::sample(unsigned int nRep) {
  for (unsigned int i = 0; i < nRep; i++)
    sample();
}


void Sampler::sample() {
  vector<IndexT> sCountRow = countSamples(nObs, nSamp);
  IndexT rowPrev = 0;
  for (IndexT row = 0; row < nObs; row++) {
    if (sCountRow[row] > 0) {
      sbCresc.emplace_back(row - exchange(rowPrev, row), sCountRow[row]);
    }
  }
}


// Sample counting is sensitive to locality.  In the absence of
// binning, access is random.  Larger bins improve locality, but
// performance begins to degrade when bin size exceeds available
// cache.
vector<IndexT> Sampler::countSamples(IndexT nObs,
				     IndexT nSamp) {
  vector<IndexT> sc(nObs);
  vector<IndexT> idx(CallBack::sampleRows(nSamp));
  if (binIdx(sc.size()) > 0) {
    idx = binIndices(idx);
  }
    
  for (auto index : idx) {
    sc[index]++;
  }

  return sc;
}


vector<unsigned int> Sampler::binIndices(const vector<unsigned int>& idx) {
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


unique_ptr<Sample> Sampler::rootSample(unsigned int tIdx) const {
  return response->rootSample(this, tIdx);
}


vector<IndexT> Sampler::sampledRows(unsigned int tIdx) const {
  vector<IndexT> rowsSampled(sbCresc.size());

  IndexT sIdx = 0;
  IndexT row = 0;
  for (auto nux : sbCresc) {
    row += nux.getDelRow();
    rowsSampled[sIdx++] = row;
  }

  return rowsSampled;
}

# ifdef restore
// RECAST:
void SamplerBlock::dump(const Sampler* sampler,
			vector<vector<size_t> >& rowTree,
			vector<vector<IndexT> >& sCountTree) const {
  size_t bagIdx = 0; // Absolute sample index.
  for (unsigned int tIdx = 0; tIdx < raw->getNMajor(); tIdx++) {
    IndexT row = 0;
    while (bagIdx != getHeight(tIdx)) {
      row += getDelRow(bagIdx);
      rowTree[tIdx].push_back(row);
      sCountTree[tIdx].push_back(getSCount(bagIdx));
	//	extentTree[tIdx].emplace_back(getExtent(leafIdx)); TODO
    }
  }
}
#endif
