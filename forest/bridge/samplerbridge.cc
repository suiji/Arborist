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


unsigned int SamplerBridge::getNTree() const {
  return sampler->getNTree();
}


unique_ptr<SamplerBridge> SamplerBridge::preSample(size_t nSamp,
						   size_t nObs,
						   unsigned int nTree,
						   bool replace,
						   const double weight[]) {
  SamplerNux::setMasks(nObs);
  return make_unique<SamplerBridge>(nSamp, nObs, nTree, replace, weight);
}


SamplerBridge::SamplerBridge(size_t nSamp,
			     size_t nObs,
			     unsigned int nTree,
			     bool replace,
			     const double weight[]) :
  sampler(make_unique<Sampler>(nSamp, nObs, nTree, replace, weight)) {
}

// EXIT:  EFfects indistinguishable from readReg().
unique_ptr<SamplerBridge> SamplerBridge::trainReg(const vector<double>& yTrain,
						  size_t nSamp,
						  unsigned int nTree,
						  const double samples[]) {
  SamplerNux::setMasks(yTrain.size());
  vector<vector<SamplerNux>> nux = SamplerRW::unpack(samples, nSamp, nTree);
  return make_unique<SamplerBridge>(yTrain, nSamp, std::move(nux));
}


SamplerBridge::SamplerBridge(const vector<double>& yTrain,
			     size_t nSamp,
			     vector<vector<SamplerNux>> samples) :
  sampler(make_unique<Sampler>(yTrain, nSamp, std::move(samples))) {
}


void SamplerBridge::sample() {
  sampler->sample();
}


void SamplerBridge::appendSamples(const vector<size_t>& idx) { // EXIT
  sampler->appendSamples(idx);
}


unique_ptr<SamplerBridge> SamplerBridge::readReg(const vector<double>& yTrain,
						 size_t nSamp,
						 unsigned int nTree,
						 const double samples[],
						 bool bagging) {
  SamplerNux::setMasks(yTrain.size());
  vector<vector<SamplerNux>> nux = SamplerRW::unpack(samples, nSamp, nTree);
  return make_unique<SamplerBridge>(yTrain, nSamp, std::move(nux), bagging);
}


SamplerBridge::SamplerBridge(const vector<double>& yTrain,
			     size_t nSamp,
			     vector<vector<SamplerNux>> samples,
			     bool bagging) :
  sampler(make_unique<Sampler>(yTrain, std::move(samples), nSamp, bagging)) {
}


unique_ptr<SamplerBridge> SamplerBridge::trainCtg(const vector<unsigned int>& yTrain,
						  size_t nSamp,
						  unsigned int nTree,
						  const double samples[],
						  unsigned int nCtg,
						  const vector<double>& classWeight) {
  SamplerNux::setMasks(yTrain.size());
  vector<vector<SamplerNux>> nux = SamplerRW::unpack(samples, nSamp, nTree, nCtg);
  return make_unique<SamplerBridge>(yTrain, nSamp, std::move(nux), nCtg, classWeight);
}


SamplerBridge::SamplerBridge(const vector<unsigned int>& yTrain,
			     size_t nSamp,
			     vector<vector<SamplerNux>> nux,
			     unsigned int nCtg,
			     const vector<double>& classWeight) :
  sampler(make_unique<Sampler>(yTrain, nSamp, std::move(nux), nCtg, classWeight)) {
}
  


unique_ptr<SamplerBridge> SamplerBridge::readCtg(const vector<unsigned int>& yTrain,
						 unsigned int nCtg,
						 size_t nSamp,
						 unsigned int nTree,
						 const double samples[],
						 bool bagging) {
  SamplerNux::setMasks(yTrain.size());
  vector<vector<SamplerNux>> nux = SamplerRW::unpack(samples, nSamp, nTree, nCtg);
  return make_unique<SamplerBridge>(yTrain, nSamp, std::move(nux), nCtg, bagging);
}


SamplerBridge::SamplerBridge(const vector<unsigned int>& yTrain,
			     size_t nSamp,
			     vector<vector<SamplerNux>> samples,
			     unsigned int nCtg,
			     bool bagging) :
  sampler(make_unique<Sampler>(yTrain, std::move(samples), nSamp, nCtg, bagging)) {
}


SamplerBridge::~SamplerBridge() {
  SamplerNux::unsetMasks();
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
