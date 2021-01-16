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


Forest::Forest(const vector<size_t>& nodeHeight_,
               const DecNode treeNode_[],
	       unsigned int facVec[],
               const vector<size_t>& facHeight) :
  nodeHeight(move(nodeHeight_)),
  treeNode(treeNode_),
  facSplit(make_unique<BVJaggedV>(facVec, facHeight)) {
}


Forest::~Forest() {
}


vector<size_t> Forest::cacheOrigin() const {
  vector<size_t> origin(getNTree());//nTree);
  for (unsigned int tIdx = 0; tIdx < origin.size(); tIdx++) {
    origin[tIdx] = tIdx == 0 ? 0 : nodeHeight[tIdx-1];
  }
  return origin;
}


void Forest::dump(vector<vector<PredictorT> >& predTree,
                  vector<vector<double> >& splitTree,
                  vector<vector<IndexT> >& delIdxTree,
                  vector<vector<IndexT> >& facSplitTree) const {
  dump(predTree, splitTree, delIdxTree);
  facSplitTree = facSplit->dump();
}


void Forest::dump(vector<vector<PredictorT> >& pred,
                  vector<vector<double> >& split,
                  vector<vector<IndexT> >& delIdx) const {
  for (unsigned int tIdx = 0; tIdx < getNTree(); tIdx++) {
    for (IndexT nodeIdx = 0; nodeIdx < nodeHeight[tIdx]; nodeIdx++) {
      pred[tIdx].push_back(treeNode[nodeIdx].getPredIdx());
      delIdx[tIdx].push_back(treeNode[nodeIdx].getDelIdx());

      // N.B.:  split field must fit within a double.
      split[tIdx].push_back(treeNode[nodeIdx].getSplitNum());
    }
  }
}
