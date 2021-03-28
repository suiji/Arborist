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
			     unsigned int nTree,
			     const unsigned char* samplerNux) :
  sampler(make_unique<Sampler>(yTrain, (const SamplerNux*) samplerNux, nTree)) {
}


SamplerBridge::~SamplerBridge() {}


const Sampler* SamplerBridge::getSampler() const {
  return sampler.get();
}


SamplerBridge::SamplerBridge(const vector<unsigned int>& yTrain,
			     unsigned int nCtg,
			     unsigned int nTree,
			     const unsigned char* samplerNux) :
  sampler(make_unique<Sampler>(yTrain, (const SamplerNux*) samplerNux, nTree, nCtg)) {
}
