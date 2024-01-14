// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file dectree.cc

   @brief Methods for building and walking a decision tree.

   @author Mark Seligman
 */


#include "dectree.h"

#include "quant.h" // Inclusion only.

DecTree::DecTree(const vector<DecNode>& decNode_,
		 const BV& facSplit_,
		 const BV& facObserved_,
		 const vector<double>& nodeScore_) :
  decNode(decNode_),
  facSplit(facSplit_),
  facObserved(facObserved_),
  nodeScore(nodeScore_) {
}


vector<DecTree> DecTree::unpack(unsigned int nTree,
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


vector<double> DecTree::unpackDoubles(const double val[],
				      const size_t extent) {
  vector<double> valVec;
  copy(val, val + extent, back_inserter(valVec));

  return valVec;
}


BV DecTree::unpackBits(const unsigned char raw[],
		       size_t extent) {
  return BV(raw, extent);
}


vector<DecNode> DecTree::unpackNodes(const complex<double> nodes[],
				     size_t extent) {
  vector<DecNode> decNode;
  copy(nodes, nodes + extent, back_inserter(decNode));

  return decNode;
}


DecTree::~DecTree() = default;
