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


#include "dectree.h"
#include "forest.h"
#include "predictframe.h"
#include "sampler.h"
#include "ompthread.h"

// Inclusion only:
#include "quant.h"

Forest::Forest(vector<DecTree>&& decTree_,
	       const tuple<double, double, string>& scoreDesc_,
	       Leaf&& leaf_) :
  decTree(decTree_),
  scoreDesc(ScoreDesc(scoreDesc_)),
  leaf(leaf_),
  noNode(maxHeight(decTree)),
  nTree(decTree.size()) {
}


size_t Forest::maxHeight(const vector<DecTree>& decTree) {
  size_t height = 0;
  for (const DecTree& tree : decTree) {
    height = max(height, tree.nodeCount());
  }
  return height;
}


unique_ptr<ForestPredictionReg> Forest::makePredictionReg(const Sampler* sampler,
							  const class Predict* predict,
							  bool reportAuxiliary) {
  return scoreDesc.makePredictionReg(predict, sampler, reportAuxiliary);
}
						   

unique_ptr<ForestPredictionCtg> Forest::makePredictionCtg(const Sampler* sampler,
							  const class Predict* predict,
							  bool reportAuxiliary) {
  return scoreDesc.makePredictionCtg(predict, sampler, reportAuxiliary);
}
						   

void Forest::dump(vector<vector<PredictorT> >& predTree,
                  vector<vector<double> >& splitTree,
                  vector<vector<size_t> >& delIdxTree,
		  vector<vector<unsigned char>>& facSplitTree,
		  vector<vector<double>>& scoreTree) const {
  dump(predTree, splitTree, delIdxTree, scoreTree);
}


void Forest::dump(vector<vector<PredictorT> >& pred,
                  vector<vector<double> >& split,
                  vector<vector<size_t> >& delIdx,
		  vector<vector<double>>& score) const {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    const DecTree& tree = decTree[tIdx];
    for (IndexT nodeIdx = 0; nodeIdx < tree.nodeCount(); nodeIdx++) {
      pred[tIdx].push_back(tree.getPredIdx(nodeIdx));
      delIdx[tIdx].push_back(tree.getDelIdx(nodeIdx));
      score[tIdx].push_back(tree.getScore(nodeIdx));
      // N.B.:  split field must fit within a double.
      split[tIdx].push_back(tree.getSplitNum(nodeIdx));
    }
  }
}


vector<IndexT> Forest::getLeafNodes(unsigned int tIdx,
				    IndexT extent) const {
  vector<IndexT> leafIndices(extent);
  IndexT nodeIdx = 0;
  for (auto node : decTree[tIdx].getNode()) {
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
    leafDom[tIdx] = leafDominators(decTree[tIdx].getNode());
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
