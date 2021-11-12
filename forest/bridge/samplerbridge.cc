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


SamplerBridge::SamplerBridge(const vector<double>& yTrain,
			     IndexT nSamp,
			     unsigned int treeChunk,
			     bool nux) :
  sampler(make_unique<Sampler>(yTrain, nux, nSamp, treeChunk)) {
}


SamplerBridge::SamplerBridge(const vector<double>& yTrain,
			     IndexT nSamp,
			     unsigned int nTree,
			     bool nux,
			     unsigned char* samples,
			     bool bagging) :
  sampler(make_unique<Sampler>(yTrain, nux, samples, nSamp, nTree, bagging)) {
}


SamplerBridge::~SamplerBridge() {}


Sampler* SamplerBridge::getSampler() const {
  return sampler.get();
}


SamplerBridge::SamplerBridge(const vector<PredictorT>& yTrain,
			     IndexT nSamp,
			     unsigned int treeChunk,
			     bool nux,
			     PredictorT nCtg,
			     const vector<double>& classWeight) :
  sampler(make_unique<Sampler>(yTrain, nux, nSamp, treeChunk, nCtg, classWeight)) {
}
  


SamplerBridge::SamplerBridge(const vector<unsigned int>& yTrain,
			     unsigned int nCtg,
			     IndexT nSamp,
			     unsigned int nTree,
			     bool nux,
			     unsigned char* samples,
			     bool bagging) :
  sampler(make_unique<Sampler>(yTrain, nux, samples, nSamp, nTree, nCtg, bagging)) {
}


size_t SamplerBridge::getBlockBytes() const {
  return sampler->getBlockBytes();
}


void SamplerBridge::dumpRaw(unsigned char blOut[]) const {
  sampler->dumpRaw(blOut);
}
