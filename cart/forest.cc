// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.cc

   @brief Methods for building and walking the dec`ision tree.

   @author Mark Seligman
 */


#include "bv.h"
#include "forest.h"
#include "summaryframe.h"
#include "rankedframe.h"
#include "predict.h"


vector<double> TreeNode::splitQuant;


ForestTrain::ForestTrain(unsigned int treeChunk) :
  nbCresc(make_unique<NBCresc>(treeChunk)),
  fbCresc(make_unique<FBCresc>(treeChunk)) {
}


ForestTrain::~ForestTrain() {
}


Forest::Forest(const unsigned int height_[],
	       unsigned int nTree_,
               const TreeNode treeNode_[],
	       unsigned int facVec_[],
               const unsigned int facHeight_[]) :
  nodeHeight(height_),
  nTree(nTree_),
  treeNode(treeNode_),
  facSplit(make_unique<BVJagged>(facVec_, facHeight_, nTree)) {
}


Forest::~Forest() {
}


unsigned int TreeNode::advance(const BVJagged *facSplit,
                               const unsigned int rowT[],
                               unsigned int tIdx,
                               unsigned int &leafIdx) const {
  auto predIdx = getPredIdx();
  if (lhDel == 0) {
    leafIdx = predIdx;
    return 0;
  }
  else {
    IndexType bitOff = getSplitBit() + rowT[predIdx];
    return facSplit->testBit(tIdx, bitOff) ? lhDel : lhDel + 1;
  }
}


unsigned int TreeNode::advance(const PredictFrame* blockFrame,
                               const BVJagged *facSplit,
                               const unsigned int *rowFT,
                               const double *rowNT,
                               unsigned int tIdx,
                               unsigned int &leafIdx) const {
  auto predIdx = getPredIdx();
  if (lhDel == 0) {
    leafIdx = predIdx;
    return 0;
  }
  else {
    bool isFactor;
    unsigned int blockIdx = blockFrame->getIdx(predIdx, isFactor);
    return isFactor ?
      (facSplit->testBit(tIdx, getSplitBit() + rowFT[blockIdx]) ?
       lhDel : lhDel + 1) : (rowNT[blockIdx] <= getSplitNum() ?
                             lhDel : lhDel + 1);
  }
}


void ForestTrain::treeInit(unsigned int tIdx, unsigned int nodeCount) {
  nbCresc->treeInit(tIdx, nodeCount);
}


NBCresc::NBCresc(unsigned int treeChunk) :
  treeNode(vector<TreeNode>(0)),
  height(vector<size_t>(treeChunk)) {
}


FBCresc::FBCresc(unsigned int treeChunk) :
  fac(vector<unsigned int>(0)),
  height(vector<size_t>(treeChunk)) {
}


void NBCresc::treeInit(unsigned int tIdx, unsigned int nodeCount) {
  treeFloor = treeNode.size();
  height[tIdx] = treeFloor + nodeCount;
  TreeNode tn;
  treeNode.insert(treeNode.end(), nodeCount, tn);
}


void FBCresc::treeCap(unsigned int tIdx) {
  height[tIdx] = fac.size();
}


void NBCresc::dumpRaw(unsigned char nodeRaw[]) const {
  for (size_t i = 0; i < treeNode.size() * sizeof(TreeNode); i++) {
    nodeRaw[i] = ((unsigned char*) &treeNode[0])[i];
  }
}


void FBCresc::dumpRaw(unsigned char facRaw[]) const {
  for (size_t i = 0; i < fac.size() * sizeof(unsigned int); i++) {
    facRaw[i] = ((unsigned char*) &fac[0])[i];
  }
}


void ForestTrain::appendBits(const BV *splitBits,
                             unsigned int bitEnd,
                             unsigned int tIdx) {
  fbCresc->appendBits(splitBits, bitEnd, tIdx);
}


void FBCresc::appendBits(const BV* splitBits,
                         unsigned int bitEnd,
                         unsigned int tIdx) {
  splitBits->consume(fac, bitEnd);
  treeCap(tIdx);
}


void ForestTrain::nonTerminal(IndexType nodeIdx,
                              IndexType lhDel,
                              const SplitCrit& crit) {
  nbCresc->branchProduce(nodeIdx, lhDel, crit);
}


void ForestTrain::terminal(unsigned int nodeIdx,
                            unsigned int leafIdx) {
  nbCresc->leafProduce(nodeIdx, leafIdx);
}


void ForestTrain::splitUpdate(const SummaryFrame *sf) {
  nbCresc->splitUpdate(sf);
}


void NBCresc::splitUpdate(const SummaryFrame* sf) {
  for (auto & tn : treeNode) {
    tn.setQuantRank(sf);
  }
}


void TreeNode::setQuantRank(const SummaryFrame* sf) {
  auto predIdx = getPredIdx();
  if (!Nonterminal() || sf->isFactor(predIdx))
    return;

  double rankNum = criterion.imputeRank(splitQuant[predIdx]);
  IndexType rankFloor = floor(rankNum);
  IndexType rankCeil = ceil(rankNum);
  double valFloor = sf->getNumVal(predIdx, rankFloor);
  double valCeil = sf->getNumVal(predIdx, rankCeil);
  criterion.setNum(valFloor + (rankNum - rankFloor) * (valCeil - valFloor));
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
