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
#include "sampler.h"
#include "samplerrw.h"

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


SamplerBridge::SamplerBridge(const vector<double>& yTrain,
			     size_t nSamp,
			     unsigned int nTree,
			     const double samples[],
			     bool bagging) {
  SamplerNux::setMasks(yTrain.size());
  vector<vector<SamplerNux>> nux = SamplerRW::unpack(samples, nSamp, nTree);
  sampler = make_unique<Sampler>(yTrain, std::move(nux), nSamp, bagging);
}


SamplerBridge::SamplerBridge(const vector<unsigned int>& yTrain,
			     size_t nSamp,
			     unsigned int nTree,
			     const double samples[],
			     unsigned int nCtg,
			     const vector<double>& classWeight) {
  SamplerNux::setMasks(yTrain.size());
  vector<vector<SamplerNux>> nux = SamplerRW::unpack(samples, nSamp, nTree, nCtg);
  sampler = make_unique<Sampler>(yTrain, nSamp, std::move(nux), nCtg, classWeight);
}


SamplerBridge::SamplerBridge(const vector<unsigned int>& yTrain,
			     unsigned int nCtg,
			     size_t nSamp,
			     unsigned int nTree,
			     const double samples[],
			     bool bagging) {
  SamplerNux::setMasks(yTrain.size());
  vector<vector<SamplerNux>> nux = SamplerRW::unpack(samples, nSamp, nTree, nCtg);
  sampler = make_unique<Sampler>(yTrain, std::move(nux), nSamp, nCtg, bagging);
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


SamplerBridge::~SamplerBridge() {
  SamplerNux::unsetMasks();
}


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
