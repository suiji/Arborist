// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#include "predict.h"
#include "sampler.h"
#include "train.h"
#include "response.h"
#include "samplernux.h"
#include "frontier.h"
#include "predictorframe.h"
#include "booster.h"
#include "prng.h"


PackedT SamplerNux::delMask = 0;
unsigned int SamplerNux::rightBits = 0;


Sampler::Sampler(IndexT nSamp_,
		 IndexT nObs_,
		 unsigned int nRep_,
		 bool replace,
		 const double weight[]) :
    nRep(nRep_),
    nObs(nObs_),
    nSamp(nSamp_),
    response(nullptr) {
  setCoefficients(weight, replace);
}

  
void Sampler::setCoefficients(const double weight[],
			      bool replace) {
  if (weight != nullptr) {
    if (replace)
      walker = make_unique<Sample::Walker<size_t>>(weight, nObs);
    else {
      weightNoReplace = vector<double>(weight, weight + nObs);
    }
  }
  else if (!replace) { // Uniform non-replace scaling vector.
    coeffNoReplace = vector<size_t>(nSamp);
    iota(coeffNoReplace.begin(), coeffNoReplace.end(), nObs - nSamp + 1);
    reverse(coeffNoReplace.begin(), coeffNoReplace.end());
  }
}


Sampler::Sampler(IndexT nObs_,
		 IndexT nSamp_,
		 const vector<vector<SamplerNux>>& samples_) :
  nRep(samples_.size()),
  nObs(nObs_),
  nSamp(nSamp_),
  response(nullptr),
  samples(samples_) {
}

  
Sampler::Sampler(const vector<double>& yTrain,
		 IndexT nSamp_,
		 vector<vector<SamplerNux>> samples_) :
  nRep(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryReg(yTrain)),
  samples(samples_) {
  Booster::setEstimate(this);
}


Sampler::Sampler(const vector<PredictorT>& yTrain,
		 IndexT nSamp_,
		 vector<vector<SamplerNux>> samples_,
		 PredictorT nCtg,
		 const vector<double>& classWeight) :
  nRep(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryCtg(yTrain, nCtg, classWeight)),
  samples(std::move(samples_)) {
  Booster::setEstimate(this);
}


Sampler::Sampler(const vector<double>& yTrain,
		 vector<vector<SamplerNux>> samples_,
		 IndexT nSamp_,
		 bool bagging) :
  nRep(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryReg(yTrain)),
  samples(std::move(samples_)),
  bagMatrix(bagRows(bagging)) {
}


Sampler::Sampler(const vector<PredictorT>& yTrain,
		 vector<vector<SamplerNux>> samples_,
		 IndexT nSamp_,
		 PredictorT nCtg,
		 bool bagging) :
  nRep(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryCtg(yTrain, nCtg)),
  samples(std::move(samples_)),
  bagMatrix(bagRows(bagging)) {
}


Sampler::~Sampler() = default;


unique_ptr<BitMatrix> Sampler::bagRows(bool bagging) {
  if (!bagging)
    return make_unique<BitMatrix>(0, 0);

  unique_ptr<BitMatrix> matrix = make_unique<BitMatrix>(nRep, nObs);
  for (unsigned int tIdx = 0; tIdx < nRep; tIdx++) {
    IndexT row = 0;
    for (IndexT sIdx = 0; sIdx != getBagCount(tIdx); sIdx++) {
      row += getDelRow(tIdx, sIdx);
      matrix->setBit(tIdx, row);
    }
  }
  return matrix;
}


unique_ptr<SampledObs> Sampler::getObs(unsigned int tIdx) const {
  return response->getObs(this, tIdx);
}


vector<IdCount> Sampler::obsExpand(const vector<SampleNux>& nuxen) const {
  vector<IdCount> idCount;

  IndexT obsIdx = 0;
  for (auto nux : nuxen) {
    obsIdx += nux.getDelRow();
    idCount.emplace_back(obsIdx, nux.getSCount());
  }

  return idCount;
}


void Sampler::sample() {
  vector<size_t> idxOut;
  if (walker != nullptr) {
    idxOut = walker->sample(nSamp);
  }
  else if (!weightNoReplace.empty()) {
    idxOut = Sample::sampleEfraimidis<size_t>(weightNoReplace, nSamp);
  }
  else if (!coeffNoReplace.empty()) {
    idxOut = Sample::sampleUniform<size_t>(coeffNoReplace, nObs);
  }
  else {
    idxOut = PRNG::rUnifIndex(nSamp, nObs);
  }

  appendSamples(idxOut);
}


void Sampler::appendSamples(const vector<size_t>& idx) {
  vector<IndexT> sCountRow = binIdx(nObs) > 0 ? countSamples(binIndices(nObs, idx)) : countSamples(idx);
  IndexT rowPrev = 0;
  for (IndexT row = 0; row < nObs; row++) {
    if (sCountRow[row] > 0) {
      sbCresc.emplace_back(row - exchange(rowPrev, row), sCountRow[row]);
    }
  }
}


vector<IndexT> Sampler::countSamples(const vector<size_t>& idx) {
  vector<IndexT> sc(nObs);
  for (auto index : idx) {
    sc[index]++;
  }

  return sc;
}


// Sample counting is sensitive to locality.  In the absence of
// binning, access is random.  Larger bins improve locality, but
// performance begins to degrade when bin size exceeds available
// cache.
vector<size_t> Sampler::binIndices(size_t nObs,
				   const vector<size_t>& idx) {
  // Sets binPop to respective bin population, then accumulates population
  // of bins to the left.
  // Performance not sensitive to bin width.
  //
  vector<size_t> binPop(1 + binIdx(nObs));
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
  vector<size_t> idxBinned(idx.size());
  for (auto index : idx) {
    int destIdx = idxAvail[binIdx(index)]--;
    idxBinned[destIdx] = index;
  }

  return idxBinned;
}


CtgT Sampler::getNCtg() const {
  return response->getNCtg();
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
