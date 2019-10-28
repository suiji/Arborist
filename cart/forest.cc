// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.cc

   @brief Methods for building and walking the decision forest.

   @author Mark Seligman
 */


#include "bv.h"
#include "forest.h"
#include "cartnode.h"


Forest::Forest(const unsigned int height_[],
	       unsigned int nTree_,
               const CartNode treeNode_[],
	       unsigned int facVec_[],
               const unsigned int facHeight_[]) :
  nodeHeight(height_),
  nTree(nTree_),
  treeNode(treeNode_),
  facSplit(make_unique<BVJagged>(facVec_, facHeight_, nTree)) {
}


Forest::~Forest() {
}


vector<size_t> Forest::cacheOrigin() const {
  vector<size_t> origin(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    origin[tIdx] = tIdx == 0 ? 0 : nodeHeight[tIdx-1];
  }
  return origin;
}


void Forest::dump(vector<vector<unsigned int> > &predTree,
                  vector<vector<double> > &splitTree,
                  vector<vector<unsigned int> > &lhDelTree,
                  vector<vector<unsigned int> > &facSplitTree) const {
  dump(predTree, splitTree, lhDelTree);
  facSplit->dump(facSplitTree);
}


void Forest::dump(vector<vector<unsigned int> > &pred,
                  vector<vector<double> > &split,
                  vector<vector<unsigned int> > &lhDel) const {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (unsigned int nodeIdx = 0; nodeIdx < getNodeHeight(tIdx); nodeIdx++) {
      pred[tIdx].push_back(treeNode[nodeIdx].getPredIdx());
      lhDel[tIdx].push_back(treeNode[nodeIdx].getLHDel());

      // Not quite:  must distinguish numeric from bit-packed:
      split[tIdx].push_back(treeNode[nodeIdx].getSplitNum());
    }
  }
}
