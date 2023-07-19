
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
#include "sampler.h"

#include <memory>
using namespace std;


LeafBridge::LeafBridge(const SamplerBridge& sb,
		       bool thin) :
  leaf(Leaf::train(sb.getNObs(), thin)) {
}


LeafBridge::LeafBridge(const SamplerBridge& samplerBridge,
		       bool thin,
		       const double extent_[],
		       const double index_[]) {
  vector<vector<size_t>> extent = unpackExtent(samplerBridge, thin, extent_);
  vector<vector<vector<size_t>>> index = unpackIndex(samplerBridge, thin, extent, index_);
  leaf = Leaf::predict(samplerBridge.getSampler(), thin,
		       std::move(extent), std::move(index));
}


LeafBridge::LeafBridge(const SamplerBridge& samplerBridge,
		       bool thin,
		       vector<vector<size_t>> extent,
		       vector<vector<vector<size_t>>> index) :
  leaf(Leaf::predict(samplerBridge.getSampler(), thin,
		     std::move(extent), std::move(index))) {
}


LeafBridge::LeafBridge(LeafBridge&& lb) :
  leaf(std::exchange(lb.leaf, nullptr)) {
}



LeafBridge::~LeafBridge() = default;


vector<vector<size_t>> LeafBridge::unpackExtent(const SamplerBridge& samplerBridge,
						bool thin,
						const double extentNum[]) {
  if (thin) {
    return vector<vector<size_t>>(0);
  }

  Sampler* sampler = samplerBridge.getSampler();
  unsigned int nTree = sampler->getNRep();
  vector<vector<size_t>> unpacked(nTree);
  size_t idx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    size_t extentTree = 0;
    while (extentTree < sampler->getBagCount(tIdx)) {
      size_t extentLeaf = extentNum[idx++];
      unpacked[tIdx].push_back(extentLeaf);
      extentTree += extentLeaf;
    }
  }
  return unpacked;
}


vector<vector<vector<size_t>>> LeafBridge::unpackIndex(const SamplerBridge& samplerBridge,
						       bool thin,
						       const vector<vector<size_t>>& extent,
						       const double numVal[]) {
  const Sampler* sampler = samplerBridge.getSampler();
  unsigned int nTree = sampler->getNRep();
  if (thin)
    return vector<vector<vector<size_t>>>(0);

  vector<vector<vector<size_t>>> unpacked(nTree);

  size_t idx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    unpacked[tIdx] = vector<vector<size_t>>(extent[tIdx].size());
    for (size_t leafIdx = 0; leafIdx < unpacked[tIdx].size(); leafIdx++) {
      vector<size_t> unpackedLeaf(extent[tIdx][leafIdx]);
      for (size_t slot = 0; slot < unpackedLeaf.size(); slot++) {
	unpackedLeaf[slot] = numVal[idx];
	idx++;
      }
      unpacked[tIdx][leafIdx] = unpackedLeaf;
    }
  }
  return unpacked;
}


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
