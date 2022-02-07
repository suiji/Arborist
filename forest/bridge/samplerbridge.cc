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


size_t SamplerBridge::strideBytes(size_t nObs) {
  return BitMatrix::strideBytes(nObs);
}


unsigned int SamplerBridge::getNObs() const {
  return sampler->getNObs();
}


unsigned int SamplerBridge::getNTree() const {
  return sampler->getNTree();
}


unique_ptr<SamplerBridge> SamplerBridge::crescReg(const vector<double>& yTrain,
						  unsigned int nSamp,
						  unsigned int treeChunk) {
  SamplerNux::setMasks(yTrain.size());
  return make_unique<SamplerBridge>(yTrain, nSamp, treeChunk);
}


SamplerBridge::SamplerBridge(const vector<double>& yTrain,
			     unsigned int nSamp,
			     unsigned int treeChunk) :
  sampler(make_unique<Sampler>(yTrain, nSamp, treeChunk)) {
}


unique_ptr<SamplerBridge> SamplerBridge::readReg(const vector<double>& yTrain,
						 unsigned int nSamp,
						 unsigned int nTree,
						 const double samples[],
						 bool bagging) {
  SamplerNux::setMasks(yTrain.size());
  return make_unique<SamplerBridge>(yTrain, nSamp, move(SamplerRW::unpack(samples, nTree, nSamp)), bagging);
}


SamplerBridge::SamplerBridge(const vector<double>& yTrain,
			     unsigned int nSamp,
			     vector<vector<SamplerNux>> samples,
			     bool bagging) :
  sampler(make_unique<Sampler>(yTrain, move(samples), nSamp, bagging)) {
}


unique_ptr<SamplerBridge> SamplerBridge::crescCtg(const vector<unsigned int>& yTrain,
						  unsigned int nSamp,
						  unsigned int treeChunk,
						  unsigned int nCtg,
						  const vector<double>& classWeight) {
  SamplerNux::setMasks(yTrain.size());
  return make_unique<SamplerBridge>(yTrain, nSamp, treeChunk, nCtg, classWeight);
}


SamplerBridge::SamplerBridge(const vector<unsigned int>& yTrain,
			     unsigned int nSamp,
			     unsigned int treeChunk,
			     unsigned int nCtg,
			     const vector<double>& classWeight) :
  sampler(make_unique<Sampler>(yTrain, nSamp, treeChunk, nCtg, classWeight)) {
}
  


unique_ptr<SamplerBridge> SamplerBridge::readCtg(const vector<unsigned int>& yTrain,
						 unsigned int nCtg,
						 unsigned int nSamp,
						 unsigned int nTree,
						 const double samples[],
						 bool bagging) {
  SamplerNux::setMasks(yTrain.size());
  return make_unique<SamplerBridge>(yTrain, nSamp, move(SamplerRW::unpack(samples, nTree, nSamp)), nCtg, bagging);
}


SamplerBridge::SamplerBridge(const vector<unsigned int>& yTrain,
			     unsigned int nSamp,
			     vector<vector<SamplerNux>> samples,
			     unsigned int nCtg,
			     bool bagging) :
  sampler(make_unique<Sampler>(yTrain, move(samples), nSamp, nCtg, bagging)) {
}


SamplerBridge::~SamplerBridge() {
  SamplerNux::unsetMasks();
}


Sampler* SamplerBridge::getSampler() const {
  return sampler.get();
}


size_t SamplerBridge::getNuxCount() const {
  return sampler->crescBagCount();
}


void SamplerBridge::dumpNux(double nuxOut[]) const {
  sampler->dumpNux(nuxOut);
}
