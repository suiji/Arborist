// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forestrw.cc

   @brief Core-specific packing/unpacking of external Forest representations.

   @author Mark Seligman
 */


#include "forestrw.h"
#include "forest.h"
#include "samplerbridge.h"
#include "sampler.h"
#include "leaf.h"
#include "bv.h"

using namespace std;


vector<DecTree> ForestRW::unpackDecTree(unsigned int nTree,
					const double nodeExtent[],
					const complex<double> nodes[],
					const double score[],
					const double facExtent[],
					const unsigned char facSplit[],
					const unsigned char facObserved[]) {
  vector<DecTree> decTree;
  size_t facIdx = 0;
  size_t nodeIdx = 0;
  vector<size_t> ndExtent(nodeExtent, nodeExtent + nTree);
  vector<size_t> fcExtent(facExtent, facExtent + nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    decTree.emplace_back(unpackNodes(nodes + nodeIdx, ndExtent[tIdx]),
			 unpackBits(facSplit + facIdx, fcExtent[tIdx]),
			 unpackBits(facObserved + facIdx, fcExtent[tIdx]),
			 unpackDoubles(score + nodeIdx, ndExtent[tIdx]));
    facIdx += fcExtent[tIdx] * sizeof(BVSlotT);
    nodeIdx += ndExtent[tIdx];
  }

  return decTree;
}


vector<double> ForestRW::unpackDoubles(const double val[],
					const size_t extent) {
  vector<double> valVec;
  copy(val, val + extent, back_inserter(valVec));

  return valVec;
}


BV ForestRW::unpackBits(const unsigned char raw[],
			 size_t extent) {
  return BV(raw, extent);
}


vector<DecNode> ForestRW::unpackNodes(const complex<double> nodes[],
				       size_t extent) {
  vector<DecNode> decNode;
  copy(nodes, nodes + extent, back_inserter(decNode));

  return decNode;
}


void ForestRW::dump(const Forest* forest,
		    vector<vector<unsigned int> >& predTree,
		    vector<vector<double> >& splitTree,
		    vector<vector<size_t> >& lhDelTree,
		    vector<vector<unsigned char> >& facSplitTree,
		    vector<vector<double>>& scoreTree) {
  IndexT fsDummy;
  forest->dump(predTree, splitTree, lhDelTree, scoreTree, fsDummy);
}


Leaf ForestRW::unpackLeaf(const SamplerBridge& samplerBridge,
			   const double extent_[],
			   const double index_[]) {
  vector<vector<size_t>> extent = unpackExtent(samplerBridge, extent_);
  vector<vector<vector<size_t>>> index = unpackIndex(samplerBridge, extent, index_);
  return Leaf(samplerBridge.getSampler(), std::move(extent), std::move(index));
}


vector<vector<size_t>> ForestRW::unpackExtent(const SamplerBridge& samplerBridge,
					      const double extentNum[]) {
  if (extentNum == nullptr) {
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


vector<vector<vector<size_t>>> ForestRW::unpackIndex(const SamplerBridge& samplerBridge,
						     const vector<vector<size_t>>& extent,
						     const double numVal[]) {
  if (extent.empty() || numVal == nullptr)
    return vector<vector<vector<size_t>>>(0);

  const Sampler* sampler = samplerBridge.getSampler();
  unsigned int nTree = sampler->getNRep();
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


