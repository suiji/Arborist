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
#include "ompthread.h"


Forest::Forest(vector<vector<DecNode>> decNode_,
	       vector<vector<double>> scores_,
	       vector<unique_ptr<BV>> factorBits_,
	       vector<unique_ptr<BV>> bitsObserved_) :
  nTree(decNode_.size()),
  decNode(std::move(decNode_)),
  scores(std::move(scores_)),
  factorBits(std::move(factorBits_)),
  bitsObserved(std::move(bitsObserved_)) {
}


void FBCresc::appendBits(const BV& splitBits_,
			 const BV& observedBits_,
			 size_t bitEnd) {
  size_t nSlot = splitBits_.appendSlots(splitBits, bitEnd);
  (void) observedBits_.appendSlots(observedBits, bitEnd);
  extents.push_back(nSlot);
}


vector<size_t> Forest::produceHeight(const vector<size_t>& extents) const {
  vector<size_t> heights(nTree);
  size_t heightAccum = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    heightAccum += extents[tIdx];
    heights[tIdx] = heightAccum;
  }
  return heights;
}


size_t Forest::noNode() const {
  size_t maxHeight = 0;
  for (unsigned int tIdx = 0; tIdx < decNode.size(); tIdx++) {
    maxHeight = max(maxHeight, decNode[tIdx].size());
  }
  return maxHeight;
}


void Forest::dump(vector<vector<PredictorT> >& predTree,
                  vector<vector<double> >& splitTree,
                  vector<vector<size_t> >& delIdxTree,
		  IndexT& dummy) const {
  dump(predTree, splitTree, delIdxTree);
}


void Forest::dump(vector<vector<PredictorT> >& pred,
                  vector<vector<double> >& split,
                  vector<vector<size_t> >& delIdx) const {
  for (unsigned int tIdx = 0; tIdx < decNode.size(); tIdx++) {
    for (IndexT nodeIdx = 0; nodeIdx < decNode[tIdx].size(); nodeIdx++) {
      pred[tIdx].push_back(decNode[tIdx][nodeIdx].getPredIdx());
      delIdx[tIdx].push_back(decNode[tIdx][nodeIdx].getDelIdx());

      // N.B.:  split field must fit within a double.
      split[tIdx].push_back(decNode[tIdx][nodeIdx].getSplitNum());
    }
  }
}


vector<IndexT> Forest::getLeafNodes(unsigned int tIdx,
				    IndexT extent) const {
  vector<IndexT> leafIndices(extent);
  IndexT nodeIdx = 0;
  for (auto node : decNode[tIdx]) {
    IndexT leafIdx;
    if (node.getLeafIdx(leafIdx)) {
      leafIndices[leafIdx] = nodeIdx;
    }
    nodeIdx++;
  }

  return leafIndices;
}


vector<vector<IndexRange>> Forest::leafDominators() const {
  vector<vector<IndexRange>> leafDom(nTree);
  
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    leafDom[tIdx] = leafDominators(decNode[tIdx]);
  }
  }
  return leafDom;
}
  

vector<IndexRange> Forest::leafDominators(const vector<DecNode>& tree) {
  IndexT height = tree.size();
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
