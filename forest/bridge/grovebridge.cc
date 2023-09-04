// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file grovebridge.cc

   @brief Exportable classes and methods training a grove of trees.

   @author Mark Seligman
*/

#include "grovebridge.h"
#include "forestbridge.h"
#include "trainbridge.h"
#include "samplerbridge.h"
#include "leafbridge.h"
#include "grove.h"

// Type completion only:
#include "forest.h"
#include "nodescorer.h"


unique_ptr<GroveBridge> GroveBridge::train(const TrainBridge& trainBridge,
					   const SamplerBridge& samplerBridge,
					   unsigned int treeOff,
					   unsigned int treeChunk,
					   const LeafBridge& leafBridge) {
  unique_ptr<Grove> grove = make_unique<Grove>(trainBridge.getFrame(), IndexRange(treeOff, treeChunk));
  grove->train(trainBridge.getFrame(), samplerBridge.getSampler(), leafBridge.getLeaf());

  return make_unique<GroveBridge>(std::move(grove));
}


GroveBridge::GroveBridge(unique_ptr<Grove> grove_) : grove(std::move(grove_)) {
}


GroveBridge::~GroveBridge() = default;


const vector<double>& GroveBridge::getPredInfo() const {
  return grove->getPredInfo();
}


const vector<size_t>& GroveBridge::getNodeExtents() const {
  return grove->getNodeExtents();
}


size_t GroveBridge::getNodeCount() const {
  return grove->getNodeCount();
}


void GroveBridge::dumpTree(complex<double> treeOut[]) const {
  grove->cacheNode(treeOut);
}


void GroveBridge::dumpScore(double scoreOut[]) const {
  grove->cacheScore(scoreOut);
}


const vector<size_t>& GroveBridge::getFacExtents() const {
  return grove->getFacExtents();
}


size_t GroveBridge::getFactorBytes() const {
  return grove->getFactorBytes();
}


void GroveBridge::dumpFactorRaw(unsigned char facOut[]) const {
  grove->cacheFacRaw(facOut);
}


void GroveBridge::dumpFactorObserved(unsigned char obsOut[]) const {
  grove->cacheObservedRaw(obsOut);
}

