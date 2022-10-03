// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file decnoderw.cc

   @brief Core-specific packing/unpacking of external DecNode representations.

   @author Mark Seligman
 */

#include "decnoderw.h"
#include "typeparam.h"
#include "bv.h"


using namespace std;

vector<vector<DecNode>> DecNodeRW::unpackNodes(const complex<double> nodes[],
						const double nodeExtent[],
						unsigned int nTree) {
  vector<vector<DecNode>> treeNodes(nTree);
  size_t feIdx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (size_t nodeIdx = 0; nodeIdx < nodeExtent[tIdx]; nodeIdx++) {
      treeNodes[tIdx].emplace_back(DecNode(nodes[feIdx]));
      feIdx++;
    }
  }
  return treeNodes;
}


vector<vector<double>> DecNodeRW::unpackScores(const double scores[],
						const double nodeExtent[],
						unsigned int nTree) {
  vector<vector<double>> treeScore(nTree);
  IndexT nodeIdx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (IndexT scoreIdx = 0; scoreIdx != nodeExtent[tIdx]; scoreIdx++)  {
      treeScore[tIdx].push_back(scores[nodeIdx++]);
    }
  }
  return treeScore;
}


vector<unique_ptr<BV>> DecNodeRW::unpackBits(const unsigned char raw[],
					     const double extent[],
					     unsigned int nTree) {
  vector<unique_ptr<BV>> bits;
  size_t rawIdx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    bits.emplace_back(make_unique<BV>(raw + rawIdx, extent[tIdx]));
    rawIdx += extent[tIdx] * sizeof(BVSlotT);
  }

  return bits;
}
