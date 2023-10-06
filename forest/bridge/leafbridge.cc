
// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leafbridge.cc

   @brief Front-end wrapper for core-level Leaf objects.

   @author Mark Seligman
 */


#include "leaf.h"
#include "leafbridge.h"
#include "samplerbridge.h"


using namespace std;


LeafBridge::LeafBridge(const SamplerBridge& sb) :
  leaf(Leaf::train(sb.getNObs())) {
}


LeafBridge::~LeafBridge() = default;


Leaf* LeafBridge::getLeaf() const {
  return leaf.get();
}


size_t LeafBridge::getExtentSize() const {
  return leaf->getExtentCresc().size();
}


size_t LeafBridge::getIndexSize() const {
  return leaf->getIndexCresc().size();
}


void LeafBridge::dumpExtent(double extentOut[]) const {
  auto extent = leaf->getExtentCresc();
  for (size_t i = 0; i < extent.size(); i++) {
    extentOut[i] = extent[i];
  }
}


void LeafBridge::dumpIndex(double indexOut[]) const {
  auto index = leaf->getIndexCresc();
  for (size_t i = 0; i < index.size(); i++) {
    indexOut[i] = index[i];
  }
}
