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
	       const double nodeExtent_[],
	       const DecNode treeNode_[],
	       const double scores_[],
	       const double facExtent[],
	       unsigned int facVec[]) :
  nTree(nTree_),
  nodeExtent(produceExtent(nodeExtent_)),
  treeNode(treeNode_),
  scores(scores_),
  facSplit(make_unique<BVJagged>(facVec, move(produceHeight(facExtent)))) {
}


vector<size_t> Forest::produceExtent(const double extent_[]) const {
  vector<size_t> extent(nTree);
  for (unsigned int i = 0; i < nTree; i++) {
    extent[i] = extent_[i];
  }
  return extent;
}


void FBCresc::appendBits(const class BV& splitBits,
			   size_t bitEnd) {
  size_t nSlot = splitBits.appendSlots(fac, bitEnd);
  extents.push_back(nSlot);
}


vector<size_t> Forest::produceHeight(const double extents[]) const {
  vector<size_t> heights(nTree);
  size_t heightAccum = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    heightAccum += extents[tIdx];
    heights[tIdx] = heightAccum;
  }
  return heights;
}


vector<vector<double>> Forest::produceScores() const {
  vector<vector<double>> treeScore(nTree);
  IndexT nodeIdx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (IndexT scoreIdx = 0; scoreIdx != nodeExtent[tIdx]; scoreIdx++)  {
      treeScore[tIdx].push_back(scores[nodeIdx++]);
    }
  }
  return treeScore;
}


size_t Forest::maxTreeHeight() const {
  return *max_element(nodeExtent.begin(), nodeExtent.end());
}


vector<size_t> Forest::treeOrigins() const {
  vector<size_t> origin(nTree);
  size_t origAccum = 0;
  for (unsigned int tIdx = 0; tIdx < origin.size(); tIdx++) {
    origin[tIdx] = exchange(origAccum, origAccum + nodeExtent[tIdx]);
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
  for (unsigned int tIdx = 0; tIdx < nodeExtent.size(); tIdx++) {
    for (IndexT nodeIdx = 0; nodeIdx < nodeExtent[tIdx]; nodeIdx++) {
      pred[tIdx].push_back(treeNode[nodeIdx].getPredIdx());
      delIdx[tIdx].push_back(treeNode[nodeIdx].getDelIdx());

      // N.B.:  split field must fit within a double.
      split[tIdx].push_back(treeNode[nodeIdx].getSplitNum());
    }
  }
}


vector<IndexRange> Forest::leafDominators(const DecNode tree[],
					  IndexT height) {
  // Gives each node the offset of its predecessor.
  vector<IndexT> delPred(height);
  for (IndexT i = 0; i < height; i++) {
    IndexT delIdx = tree[i].getDelIdx();
    if (delIdx != 0) {
      delPred[i + delIdx] = delIdx;
      delPred[i + delIdx + 1] = delIdx + 1;
    }
  }

  // Pushes dominated leaf count up the tree.
  vector<IndexT> leavesBelow(height);
  for (IndexT i = height - 1; i > 0; i--) {
    leavesBelow[i] += (tree[i].isNonterminal() ? 0: 1);
    leavesBelow[i - delPred[i]] += leavesBelow[i];
  }

  // Pushes index ranges down the tree.
  vector<IndexRange> leafDom(height);
  leafDom[0] = IndexRange(0, leavesBelow[0]); // Root dominates all leaves.
  for (IndexT i = 0; i < height; i++) {
    IndexT delIdx = tree[i].getDelIdx();
    if (delIdx != 0) {
      IndexRange leafRange = leafDom[i];
      IndexT idxTrue = i + delIdx;
      IndexT trueStart = leafRange.getStart();
      leafDom[idxTrue] = IndexRange(trueStart, leavesBelow[idxTrue]);
      IndexT idxFalse = idxTrue + 1;
      IndexT falseStart = leafDom[idxTrue].getEnd();
      leafDom[idxFalse] = IndexRange(falseStart, leavesBelow[idxFalse]);
    }
  }

  return leafDom;
}
