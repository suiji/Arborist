// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.cc

   @brief Methods for building and walking the decision tree.

   @author Mark Seligman
 */


#include "bv.h"
#include "forest.h"
#include "block.h"
#include "framemap.h"
#include "rowrank.h"
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
  nodeCount(nodeHeight[nTree-1]),
  facSplit(make_unique<BVJagged>(facVec_, facHeight_, nTree)) {
}


Forest::~Forest() {
}


unsigned int TreeNode::advance(const BVJagged *facSplit,
                               const unsigned int rowT[],
                               unsigned int tIdx,
                               unsigned int &leafIdx) const {
  if (lhDel == 0) {
    leafIdx = predIdx;
    return 0;
  }
  else {
    unsigned int bitOff = splitVal.offset + rowT[predIdx];
    return facSplit->testBit(tIdx, bitOff) ? lhDel : lhDel + 1;
  }
}


unsigned int TreeNode::advance(const FramePredict *framePredict,
                               const BVJagged *facSplit,
                               const unsigned int *rowFT,
                               const double *rowNT,
                               unsigned int tIdx,
                               unsigned int &leafIdx) const {
  if (lhDel == 0) {
    leafIdx = predIdx;
    return 0;
  }
  else {
    bool isFactor;
    unsigned int blockIdx = framePredict->FacIdx(predIdx, isFactor);
    return isFactor ?
      (facSplit->testBit(tIdx, splitVal.offset + rowFT[blockIdx]) ?
       lhDel : lhDel + 1) : (rowNT[blockIdx] <= splitVal.num ?
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
  tn.init();
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


void ForestTrain::nonTerminal(const FrameTrain *frameTrain,
                              unsigned int nodeIdx,
                              const DecNode *decNode) {
  nbCresc->branchProduce(nodeIdx, decNode, frameTrain->isFactor(decNode->predIdx));
}


void ForestTrain::terminal(unsigned int nodeIdx,
                            unsigned int leafIdx) {
  nbCresc->leafProduce(nodeIdx, leafIdx);
}


void ForestTrain::splitUpdate(const FrameTrain *frameTrain,
                              const BlockRanked *numRanked) {
  nbCresc->splitUpdate(frameTrain, numRanked);
}


void NBCresc::splitUpdate(const FrameTrain* frameTrain,
                          const BlockRanked* numRanked) {
  for (auto & tn : treeNode) {
    tn.splitUpdate(frameTrain, numRanked);
  }
}


void TreeNode::splitUpdate(const FrameTrain *frameTrain,
                           const BlockRanked *numRanked) {
  if (Nonterminal() && !frameTrain->isFactor(predIdx)) {
    splitVal.num = numRanked->QuantRank(predIdx, splitVal.rankRange, splitQuant);
  }
}


vector<size_t> Forest::cacheOrigin() const {
  vector<size_t> origin(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    origin[tIdx] = tIdx == 0 ? 0 : nodeHeight[tIdx-1];
  }
  return move(origin);
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
      pred[tIdx].push_back(treeNode[nodeIdx].getPred());
      lhDel[tIdx].push_back(treeNode[nodeIdx].getLHDel());

      // Not quite:  must distinguish numeric from bit-packed:
      split[tIdx].push_back(treeNode[nodeIdx].getSplitNum());
    }
  }
}
