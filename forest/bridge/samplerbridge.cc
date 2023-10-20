// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplerbridge.cc

   @brief Front-end wrapper for core-level Sampler objects.

   @author Mark Seligman
 */

#include "samplerbridge.h"
#include "predictbridge.h"
#include "forestbridge.h"
#include "sampler.h"
#include "samplerrw.h"
#include "rleframe.h"

#include <memory>
using namespace std;


size_t SamplerBridge::getNObs() const {
  return sampler->getNObs();
}


size_t SamplerBridge::getNSamp() const {
  return sampler->getNSamp();
}


unsigned int SamplerBridge::getNRep() const {
  return sampler->getNRep();
}


SamplerBridge::SamplerBridge(size_t nSamp,
			     size_t nObs,
			     unsigned int nTree,
			     bool replace,
			     const double weight[]) {
  SamplerNux::setMasks(nObs);
  sampler = make_unique<Sampler>(nSamp, nObs, nTree, replace, weight);
}


SamplerBridge::SamplerBridge(vector<double> yTrain,
			     size_t nSamp,
			     unsigned int nTree,
			     const double samples[]) {
  SamplerNux::setMasks(yTrain.size());
  vector<vector<SamplerNux>> nux = SamplerRW::unpack(samples, nSamp, nTree);
  sampler = make_unique<Sampler>(yTrain, nSamp, std::move(nux));
}


SamplerBridge::SamplerBridge(vector<double> yTrain,
			     size_t nSamp,
			     unsigned int nTree,
			     const double samples[],
			     unique_ptr<RLEFrame> rleFrame) {
  SamplerNux::setMasks(yTrain.size());
  vector<vector<SamplerNux>> nux = SamplerRW::unpack(samples, nSamp, nTree);
  sampler = make_unique<Sampler>(yTrain, std::move(nux), nSamp, std::move(rleFrame));
}


SamplerBridge::SamplerBridge(vector<unsigned int> yTrain,
			     size_t nSamp,
			     unsigned int nTree,
			     const double samples[],
			     unsigned int nCtg,
			     const vector<double>& classWeight) {
  SamplerNux::setMasks(yTrain.size());
  vector<vector<SamplerNux>> nux = SamplerRW::unpack(samples, nSamp, nTree, nCtg);
  sampler = make_unique<Sampler>(yTrain, nSamp, std::move(nux), nCtg, classWeight);
}


SamplerBridge::SamplerBridge(vector<unsigned int> yTrain,
			     unsigned int nCtg,
			     size_t nSamp,
			     unsigned int nTree,
			     const double samples[],
			     unique_ptr<RLEFrame> rleFrame) {
  SamplerNux::setMasks(yTrain.size());
  vector<vector<SamplerNux>> nux = SamplerRW::unpack(samples, nSamp, nTree, nCtg);
  sampler = make_unique<Sampler>(yTrain, std::move(nux), nSamp, nCtg, std::move(rleFrame));
}


SamplerBridge::SamplerBridge(size_t nObs,
			     const double samples[],
			     size_t nSamp,
			     unsigned int nTree) {
  SamplerNux::setMasks(nObs);
  sampler = make_unique<Sampler>(nObs, nSamp, SamplerRW::unpack(samples, nSamp, nTree));
}


SamplerBridge::SamplerBridge(SamplerBridge&& sb) :
  sampler(std::exchange(sb.sampler, nullptr)) {
}


SamplerBridge::~SamplerBridge() = default;


void SamplerBridge::sample() {
  sampler->sample();
}


Sampler* SamplerBridge::getSampler() const {
  return sampler.get();
}


size_t SamplerBridge::getNuxCount() const {
  return sampler->crescCount();
}


void SamplerBridge::dumpNux(double nuxOut[]) const {
  sampler->dumpNux(nuxOut);
}


bool SamplerBridge::categorical() const {
  return sampler->getNCtg() > 0;
}


unique_ptr<PredictRegBridge> SamplerBridge::predictReg(ForestBridge& forestBridge,
					   vector<double> yTest)  const {
  return PredictRegBridge::predict(getSampler(), forestBridge.getForest(), yTest);
}


unique_ptr<PredictCtgBridge> SamplerBridge::predictCtg(ForestBridge& forestBridge,
					   vector<unsigned int> yTest) const {
  return PredictCtgBridge::predict(getSampler(), forestBridge.getForest(), yTest);
}
