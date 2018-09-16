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


vector<double> ForestNode::splitQuant;


/**
   @brief Crescent constructor for training.
*/
ForestTrain::ForestTrain(unsigned int treeChunk) :
  forestNode(vector<ForestNode>(0)),
  nodeHeight(vector<size_t>(treeChunk)),
  facHeight(vector<size_t>(treeChunk)),
  facVec(vector<unsigned int>(0)) {
}


ForestTrain::~ForestTrain() {
}


/**
   @brief Constructor for prediction.
*/
Forest::Forest(const unsigned int height_[],
	       unsigned int nTree_,
               const ForestNode forestNode_[],
	       unsigned int facVec_[],
               const unsigned int facHeight_[]) :
  nodeHeight(height_),
  nTree(nTree_),
  forestNode(forestNode_),
  nodeCount(nodeHeight[nTree-1]),
  facSplit(make_unique<BVJagged>(facVec_, facHeight_, nTree)) {
}


/**
 */ 
Forest::~Forest() {
}


unsigned int ForestNode::advance(const BVJagged *facSplit,
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


unsigned int ForestNode::advance(const FramePredict *framePredict,
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
    return isFactor ? (facSplit->testBit(tIdx, splitVal.offset + rowFT[blockIdx]) ? lhDel : lhDel + 1) : (rowNT[blockIdx] <= splitVal.num ? lhDel : lhDel + 1);
  }
}


/**
 */
void ForestTrain::initNode(unsigned int extent) {
  ForestNode fn;
  fn.Init();
  forestNode.insert(forestNode.end(), extent, fn);
}


/**
   @brief Produces new splits for an entire tree.
 */
void ForestTrain::BitProduce(const BV *splitBits,
                             unsigned int bitEnd) {
  splitBits->Consume(facVec, bitEnd);
}


/**
  @brief Reserves space in the relevant vectors for new trees.
 */
void ForestTrain::Reserve(unsigned int blockHeight,
                          unsigned int blockFac,
                          double slop) {
  forestNode.reserve(slop * blockHeight);
  if (blockFac > 0) {
    facVec.reserve(slop * blockFac);
  }
}

void ForestTrain::setHeights(unsigned int tIdx) {
  nodeHeight[tIdx] = getHeight();
  facHeight[tIdx] = getSplitHeight();
}


void ForestTrain::NonTerminal(const FrameTrain *frameTrain,
                              unsigned int tIdx,
                              unsigned int idx,
                              const DecNode *decNode) {
  BranchProduce(tIdx, idx, decNode, frameTrain->isFactor(decNode->predIdx));
}


/**
   @brief Post-pass to update numerical splitting values from ranks.

   @param rowRank holds the presorted predictor values.

   @return void
 */
void ForestTrain::SplitUpdate(const FrameTrain *frameTrain,
                              const BlockRanked *numRanked) {
  for (auto & fn : forestNode) {
    fn.SplitUpdate(frameTrain, numRanked);
  }
}


/**
   @brief Assigns value at quantile rank to numerical split.

   @param rowRank holds the presorted predictor values.

   @return void.
 */
void ForestNode::SplitUpdate(const FrameTrain *frameTrain,
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
  return origin;
}


void Forest::Export(vector<vector<unsigned int> > &predTree,
                    vector<vector<double> > &splitTree,
                    vector<vector<unsigned int> > &lhDelTree,
                    vector<vector<unsigned int> > &facSplitTree) const {
  NodeExport(predTree, splitTree, lhDelTree);
  facSplit->Export(facSplitTree);
}


/**
   @brief Unpacks node fields into vector of per-tree vectors.

   @return void, with output reference vectors.
 */
void Forest::NodeExport(vector<vector<unsigned int> > &pred,
                        vector<vector<double> > &split,
                        vector<vector<unsigned int> > &lhDel) const {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (unsigned int nodeIdx = 0; nodeIdx < getNodeHeight(tIdx); nodeIdx++) {
      pred[tIdx].push_back(forestNode[nodeIdx].Pred());
      lhDel[tIdx].push_back(forestNode[nodeIdx].LHDel());

      // Not quite:  must distinguish numeric from bit-packed:
      split[tIdx].push_back(forestNode[nodeIdx].Split());
    }
  }
}
