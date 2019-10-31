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


Forest::Forest(const IndexT height_[],
	       unsigned int nTree_,
               const DecNode treeNode_[],
	       PredictorT facVec_[],
               const IndexT facHeight_[]) :
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


void Forest::dump(vector<vector<PredictorT> > &predTree,
                  vector<vector<double> > &splitTree,
                  vector<vector<IndexT> > &lhDelTree,
                  vector<vector<IndexT> > &facSplitTree) const {
  dump(predTree, splitTree, lhDelTree);
  facSplit->dump(facSplitTree);
}


void Forest::dump(vector<vector<PredictorT> > &pred,
                  vector<vector<double> > &split,
                  vector<vector<IndexT> > &lhDel) const {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (IndexT nodeIdx = 0; nodeIdx < getNodeHeight(tIdx); nodeIdx++) {
      pred[tIdx].push_back(treeNode[nodeIdx].getPredIdx());
      lhDel[tIdx].push_back(treeNode[nodeIdx].getLHDel());

      // Not quite:  must distinguish numeric from bit-packed:
      split[tIdx].push_back(treeNode[nodeIdx].getSplitNum());
    }
  }
}
