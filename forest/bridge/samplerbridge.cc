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
  sampler(Sampler::trainReg(yTrain, nux, nSamp, treeChunk)) {
}


SamplerBridge::SamplerBridge(const vector<double>& yTrain,
			     IndexT nSamp,
			     unsigned int nTree,
			     bool nux,
			     unsigned char* samples,
			     const double extent[],
			     const double index[],
			     bool bagging) :
  sampler(Sampler::predictReg(yTrain, nux, samples, nSamp, nTree, extent, index, bagging)) {
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
  sampler(Sampler::trainCtg(yTrain, nux, nSamp, treeChunk, nCtg, classWeight)) {
}
  


SamplerBridge::SamplerBridge(const vector<unsigned int>& yTrain,
			     unsigned int nCtg,
			     IndexT nSamp,
			     unsigned int nTree,
			     bool nux,
			     unsigned char* samples,
			     const double extent[],
			     const double index[],
			     bool bagging) :
  sampler(Sampler::predictCtg(yTrain, nux, samples, nSamp, nTree, extent, index, nCtg, bagging)) {
}


size_t SamplerBridge::getBlockBytes() const {
  return sampler->crescBlockBytes();
}


size_t SamplerBridge::getExtentSize() const {
  return sampler->crescExtentSize();
}


size_t SamplerBridge::getIndexSize() const {
  return sampler->crescIndexSize();
}


void SamplerBridge::dumpRaw(unsigned char blOut[]) const {
  sampler->dumpRaw(blOut);
}


void SamplerBridge::dumpExtent(double extentOut[]) const {
  sampler->dumpExtent(extentOut);
}


void SamplerBridge::dumpIndex(double indexOut[]) const {
  sampler->dumpIndex(indexOut);
}
