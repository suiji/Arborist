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


Forest::Forest(unsigned int nTree_,
	       const DecNode treeNode_[],
	       unsigned int facVec[]) :
  nTree(nTree_),
  treeNode(treeNode_),
  leafNode(leafForest()), // Contains pseudo-leaves.
  facSplit(splitFactors(facVec)), // Removes pseudo-leaves.
  treeHeight(treeHeights())
{
}


size_t FBCresc::appendBits(const class BV& splitBits,
			 size_t bitEnd) {
  return splitBits.appendSlots(fac, bitEnd);
}


vector<vector<size_t>> Forest::leafForest() const {
  vector<vector<size_t>> leaves(nTree);
  IndexT lastLeaf = 0;
  IndexT misorder = 0;
  size_t nodeIdx = 0;
  unsigned int tIdx = 0;
  IndexT unreached = 1; // Root node of first tree.
  size_t facTop = 0;
  while (tIdx < nTree) {
    IndexT leafIdx;
    unreached--; // Current target reached.
    if (treeNode[nodeIdx].getLeafIdx(leafIdx)) {
      if (leafIdx != 0 && leafIdx != lastLeaf + 1)
	misorder++; // Diagnostic only.
      leaves[tIdx].push_back(nodeIdx);
      lastLeaf = leafIdx;
    }
    else {
      unreached += 2;
    }
    if (unreached == 0) { // All live unreached consumed:  end of tree.
      // Incorporates marker node as pseudo-leaf of current tree.
      if (treeNode[++nodeIdx].getLeafIdx(leafIdx)) {
	facTop += leafIdx; // == facHeight[tIdx]
	leaves[tIdx].push_back(facTop);
      }
      else {
	misorder++;
      }

      unreached = 1; // Root node of next tree, if any.
      tIdx++;
    }
    nodeIdx++;
  }

  return leaves;
}


unique_ptr<BVJagged> Forest::splitFactors(unsigned int facVec[]) {
  vector<size_t> facHeight(leafNode.size());
  for (unsigned int tIdx = 0; tIdx < leafNode.size(); tIdx++) {
    facHeight[tIdx] = leafNode[tIdx].back();
    leafNode[tIdx].pop_back(); // Removes temporary pseudo-leaf.
  }

  return make_unique<BVJagged>(facVec, move(facHeight));
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
    // Because of marker nodes, final leaf references the penultimate
    // node position in a tree.
    treeHeight[tIdx] = leafNode[tIdx].back() + 2;
  }

  return treeHeight;
}


vector<size_t> Forest::treeOrigins() const {
  vector<size_t> origin(leafNode.size());
  for (unsigned int tIdx = 0; tIdx < origin.size(); tIdx++) {
    origin[tIdx] = tIdx == 0 ? 0 : treeHeight[tIdx-1];
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
  for (unsigned int tIdx = 0; tIdx < treeHeight.size(); tIdx++) {
    for (IndexT nodeIdx = 0; nodeIdx < treeHeight[tIdx]; nodeIdx++) {
      pred[tIdx].push_back(treeNode[nodeIdx].getPredIdx());
      delIdx[tIdx].push_back(treeNode[nodeIdx].getDelIdx());

      // N.B.:  split field must fit within a double.
      split[tIdx].push_back(treeNode[nodeIdx].getSplitNum());
    }
  }
}
