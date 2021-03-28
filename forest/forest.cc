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


Forest::Forest(size_t forestHeight,
               const DecNode treeNode_[],
	       unsigned int facVec[],
               const vector<size_t>& facHeight) :
  treeNode(treeNode_),
  leafNode(leafNodes(forestHeight, facHeight.size())),
  facSplit(make_unique<BVJaggedV>(facVec, facHeight)) {
}


Forest::~Forest() {
}


vector<vector<size_t>> Forest::leafNodes(size_t nNode,
					 unsigned int nTree) const {
  vector<vector<size_t>> leaves(nTree);
  IndexT lastLeaf = 0;
  IndexT misorder = 0;
  vector<bool> reached(nNode);
  size_t nodeIdx = 0;
  unsigned int tIdx = 0;
  reached[nodeIdx] = true;
  for (; nodeIdx < nNode; nodeIdx++) { // Unreached nodes mark tree roots.
    if (!reached[nodeIdx]) {
      tIdx++;
    }
    IndexT leafIdx;
    if (treeNode[nodeIdx].getLeafIdx(leafIdx)) {
      if (leafIdx != 0 && leafIdx != lastLeaf + 1)
	misorder++; // Diagnostic only.
      leaves[tIdx].push_back(nodeIdx);
      lastLeaf = leafIdx;
    }
    else {
      IndexT delIdx = treeNode[nodeIdx].getDelIdx();
      reached[nodeIdx + delIdx] = true;
      reached[nodeIdx + delIdx + 1] = true;
    }
  }

  return leaves;
}


vector<vector<double>> Forest::getScores() const {
  vector<vector<double>> treeScore(leafNode.size());
  for (unsigned int tIdx = 0; tIdx < leafNode.size(); tIdx++) {
    for (auto leafIdx : leafNode[tIdx]) {
      treeScore[tIdx].push_back(treeNode[leafIdx].getScore());
    }
  }
  return treeScore;
}


vector<size_t> Forest::treeHeights() const {
  vector<size_t> treeHeight(leafNode.size());
  for (unsigned int tIdx = 0; tIdx < treeHeight.size(); tIdx++) {
    treeHeight[tIdx] = leafNode[tIdx].back() + 1;
  }

  return treeHeight;
}


vector<size_t> Forest::treeOrigins() const {
  vector<size_t> origin(leafNode.size());
  for (unsigned int tIdx = 0; tIdx < origin.size(); tIdx++) {
    origin[tIdx] = tIdx == 0 ? 0 : leafNode[tIdx-1].back() + 1;
  }
  return origin;
}


void Forest::dump(vector<vector<PredictorT> >& predTree,
                  vector<vector<double> >& splitTree,
                  vector<vector<IndexT> >& delIdxTree,
                  vector<vector<IndexT> >& facSplitTree) const {
  // TODO:  Dump leaf scores as well.
  dump(predTree, splitTree, delIdxTree);
  facSplitTree = facSplit->dump();
}


void Forest::dump(vector<vector<PredictorT> >& pred,
                  vector<vector<double> >& split,
                  vector<vector<IndexT> >& delIdx) const {
  vector<size_t> nodeHeight(treeHeights());
  for (unsigned int tIdx = 0; tIdx < nodeHeight.size(); tIdx++) {
    for (IndexT nodeIdx = 0; nodeIdx < nodeHeight[tIdx]; nodeIdx++) {
      pred[tIdx].push_back(treeNode[nodeIdx].getPredIdx());
      delIdx[tIdx].push_back(treeNode[nodeIdx].getDelIdx());

      // N.B.:  split field must fit within a double.
      split[tIdx].push_back(treeNode[nodeIdx].getSplitNum());
    }
  }
}
