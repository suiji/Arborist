// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#include "predict.h"
#include "sampler.h"
#include "response.h"
#include "samplernux.h"
#include "frontier.h"
#include "predictorframe.h"
#include "booster.h"
#include "forest.h"
#include "quant.h"
#include "rleframe.h"
#include "bv.h"
#include "prng.h"


PackedT SamplerNux::delMask = 0;
unsigned int SamplerNux::rightBits = 0;


Sampler::Sampler(size_t nSamp_,
		 size_t nObs_,
		 unsigned int nRep_,
		 bool replace_,
		 const vector<double>& weight,
		 size_t nHoldout,
		 const vector<size_t>& unobserved_) :
  nRep(nRep_),
  nObs(nObs_),
  unobserved(unobserved_),
  holdout(makeHoldout(nObs, nHoldout, unobserved)),
  noSample(makeNoSample(unobserved, holdout)),
  replace(replace_),
  omitMap(makeOmitMap(nObs, noSample, replace)),
  prob(makeProbability(weight, noSample)),
  nSamp(sampleCount(nSamp_, nObs, replace, noSample, prob)),
  trivial(false),
  response(nullptr) {
  walker = (prob.empty() || !replace) ? nullptr : make_unique<Sample::Walker<size_t>>(prob, nObs);
}


vector<size_t> Sampler::makeHoldout(size_t nObs,
				    size_t nHoldout,
				    const vector<size_t>& undefined) {
  return Sample::sampleWithout<size_t>(nObs, undefined, nHoldout);
}


vector<size_t> Sampler::makeNoSample(const vector<size_t>& unobserved,
				     const vector<size_t>& holdout) {
  vector<size_t> noSample = holdout;
  noSample.insert(noSample.end(), unobserved.begin(), unobserved.end());
  sort(noSample.begin(), noSample.end());
  return noSample;
}

  
vector<double> Sampler::makeProbability(const vector<double>& weight,
					const vector<size_t>& noSample) {
  vector<double> prob = weight;
  if (!prob.empty()) {
    for (const size_t& idx : noSample) {
      prob[idx] = 0.0;
    }
    double totWeight = accumulate(prob.begin(), prob.end(), 0.0);
    if (totWeight == 0.0)
      prob = vector<double>(0);
    else {
      double scale = 1.0 / totWeight;
      for (double& probability : prob) {
	probability *= scale;
      }
    }
  }
  
  return prob;
}


size_t Sampler::sampleCount(size_t nSpecified,
			    size_t nObs,
			    bool replace,
			    const vector<size_t>& noSample,
			    const vector<double>& prob) {
  size_t sCount, nAvail;
  if (!prob.empty()) { // noSample included with zero-valued slots.
    nAvail = count_if(prob.begin(), prob.end(), [] (double probability) { return probability > 0.0;});
  }
  else if (!noSample.empty()) {
    nAvail = nObs - noSample.size();
  }
  else
    nAvail = nObs;

  if (nSpecified == 0) {
    sCount = replace ? nAvail : round(1-exp(-1)*nAvail);
  }
  else if (!replace)
    sCount = min(nSpecified, nAvail);
  else
    sCount = nSpecified;

  return sCount;
}


vector<size_t> Sampler::makeOmitMap(size_t nObs, const vector<size_t>& noSample, bool replace) {
  if (noSample.empty() || !replace)
    return vector<size_t>(0);

  vector<size_t> omitMap;
  size_t omitIdx = 0;
  size_t omitVal = noSample[0];

  for (size_t mapIdx = 0; mapIdx != nObs; mapIdx++) {
    if (mapIdx == omitVal) { // 'nObs' is inattainable.
      omitIdx++;
      omitVal = (omitIdx == noSample.size() ? nObs : noSample[omitIdx]);
    }
    else { // Appends only the non-withheld indices.
      omitMap.push_back(mapIdx);
    }
  }

  return omitMap;
}


Sampler::Sampler(size_t nObs_,
		 size_t nSamp_,
		 const vector<vector<SamplerNux>>& samples_) :
  nRep(samples_.size()),
  nObs(nObs_),
  nSamp(nSamp_),
  response(nullptr),
  samples(samples_) {
}

  
Sampler::Sampler(const vector<double>& yTrain,
		 size_t nSamp_,
		 vector<vector<SamplerNux>> samples_) :
  nRep(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryReg(yTrain)),
  samples(samples_),
  predict(Predict::makeReg(this, nullptr)) {
  Booster::setEstimate(this);
}


Sampler::Sampler(const vector<PredictorT>& yTrain,
		 size_t nSamp_,
		 vector<vector<SamplerNux>> samples_,
		 PredictorT nCtg) :
  nRep(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryCtg(yTrain, nCtg)),
  samples(std::move(samples_)),
  predict(Predict::makeCtg(this, nullptr)) {
  Booster::setEstimate(this);
}


Sampler::Sampler(const vector<double>& yTrain,
		 vector<vector<SamplerNux>> samples_,
		 size_t nSamp_,
		 unique_ptr<RLEFrame> rleFrame) :
  nRep(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryReg(yTrain)),
  samples(std::move(samples_)),
  predict(Predict::makeReg(this, std::move(rleFrame))) {
}


Sampler::Sampler(const vector<PredictorT>& yTrain,
		 vector<vector<SamplerNux>> samples_,
		 size_t nSamp_,
		 PredictorT nCtg,
		 unique_ptr<RLEFrame> rleFrame) :
  nRep(samples_.size()),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  response(Response::factoryCtg(yTrain, nCtg)),
  samples(std::move(samples_)),
  predict(Predict::makeCtg(this, std::move(rleFrame))) {
}


Sampler::~Sampler() = default;


unique_ptr<BitMatrix> Sampler::makeBag(bool bagging) const {
  if (!bagging)
    return make_unique<BitMatrix>(0, 0);

  unique_ptr<BitMatrix> matrix = make_unique<BitMatrix>(nRep, nObs);
  for (unsigned int tIdx = 0; tIdx < nRep; tIdx++) {
    size_t obsIdx = 0;
    for (size_t sIdx = 0; sIdx != getBagCount(tIdx); sIdx++) {
      obsIdx += getDelRow(tIdx, sIdx);
      matrix->setBit(tIdx, obsIdx);
    }
  }
  return matrix;
}


unique_ptr<SampledObs> Sampler::makeObs(unsigned int tIdx) const {
  return response->getObs(this, tIdx);
}


vector<IdCount> Sampler::obsExpand(const vector<SampleNux>& nuxen) const {
  vector<IdCount> idCount;

  size_t obsIdx = 0;
  for (const SampleNux& nux : nuxen) {
    obsIdx += nux.getDelRow();
    idCount.emplace_back(obsIdx, nux.getSCount());
  }

  return idCount;
}


void Sampler::sample() {
  vector<size_t> idxOut;
  if (trivial) { // No sampling:  use entire index set.
    idxOut = vector<size_t>(nObs);
    iota(idxOut.begin(), idxOut.end(), 0);
  }
  else if (walker != nullptr) { // Weighted, replacement.
    idxOut = walker->sample(nSamp, noSample);
  }
  else if (!prob.empty()) { // Weighted, no replacement.
    idxOut = Sample::sampleEfraimidis<size_t>(prob, noSample, nSamp);
  }
  else if (!replace) { // Uniform, no replacement.
    idxOut = Sample::sampleWithout<size_t>(nObs, noSample, nSamp);
  }
  else { // Uniform, replacement.
    idxOut = Sample::sampleWith<size_t>(nObs, omitMap, nSamp);
  }

  appendSamples(idxOut);
}


void Sampler::appendSamples(const vector<size_t>& idx) {
  vector<IndexT> sCountRow = binIdx(nObs) > 0 ? countSamples(binIndices(nObs, idx)) : countSamples(idx);
  size_t obsPrev = 0;
  for (size_t obsIdx = 0; obsIdx < nObs; obsIdx++) {
    if (sCountRow[obsIdx] > 0) {
      sbCresc.emplace_back(obsIdx - exchange(obsPrev, obsIdx), sCountRow[obsIdx]);
    }
  }
}


vector<IndexT> Sampler::countSamples(const vector<size_t>& idx) {
  vector<IndexT> sampleCount(nObs);
  for (auto index : idx) {
    sampleCount[index]++;
  }

  return sampleCount;
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


unique_ptr<SummaryReg> Sampler::predictReg(Forest* forest,
					   const vector<double>& yTest) const {
  return predict->predictReg(this, forest, yTest);
}


unique_ptr<SummaryCtg> Sampler::predictCtg(Forest* forest,
					   const vector<unsigned int>& yTest) const {
  return predict->predictCtg(this, forest, yTest);
}



# ifdef restore
// RECAST:
void SamplerBlock::dump(const Sampler* sampler,
			vector<vector<size_t> >& rowTree,
			vector<vector<IndexT> >& sCountTree) const {
  size_t bagIdx = 0; // Absolute sample index.
  for (unsigned int tIdx = 0; tIdx < raw->getNMajor(); tIdx++) {
    size_t row = 0;
    while (bagIdx != getHeight(tIdx)) {
      row += getDelRow(bagIdx);
      rowTree[tIdx].push_back(row);
      sCountTree[tIdx].push_back(getSCount(bagIdx));
	//	extentTree[tIdx].emplace_back(getExtent(leafIdx)); TODO
    }
  }
}
#endif
