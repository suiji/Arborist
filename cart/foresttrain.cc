// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file foresttrain.cc

   @brief Methods for growing the crescent forest.

   @author Mark Seligman
 */

#include "foresttrain.h"
#include "cartnode.h"
#include "bv.h"
#include "summaryframe.h"


ForestTrain::ForestTrain(unsigned int treeChunk) :
  nbCresc(make_unique<NBCresc>(treeChunk)),
  fbCresc(make_unique<FBCresc>(treeChunk)) {
}


ForestTrain::~ForestTrain() {
}


void ForestTrain::treeInit(unsigned int tIdx, unsigned int nodeCount) {
  nbCresc->treeInit(tIdx, nodeCount);
}


NBCresc::NBCresc(unsigned int treeChunk) :
  treeNode(vector<CartNode>(0)),
  height(vector<size_t>(treeChunk)) {
}


FBCresc::FBCresc(unsigned int treeChunk) :
  fac(vector<unsigned int>(0)),
  height(vector<size_t>(treeChunk)) {
}


void NBCresc::treeInit(unsigned int tIdx, unsigned int nodeCount) {
  treeFloor = treeNode.size();
  height[tIdx] = treeFloor + nodeCount;
  CartNode tn;
  treeNode.insert(treeNode.end(), nodeCount, tn);
}


void FBCresc::treeCap(unsigned int tIdx) {
  height[tIdx] = fac.size();
}


void NBCresc::dumpRaw(unsigned char nodeRaw[]) const {
  for (size_t i = 0; i < treeNode.size() * sizeof(CartNode); i++) {
    nodeRaw[i] = ((unsigned char*) &treeNode[0])[i];
  }
}


void FBCresc::dumpRaw(unsigned char facRaw[]) const {
  for (size_t i = 0; i < fac.size() * sizeof(unsigned int); i++) {
    facRaw[i] = ((unsigned char*) &fac[0])[i];
  }
}


void ForestTrain::appendBits(const BV *splitBits,
                             size_t bitEnd,
                             unsigned int tIdx) {
  fbCresc->appendBits(splitBits, bitEnd, tIdx);
}


void FBCresc::appendBits(const BV* splitBits,
                         size_t bitEnd,
                         unsigned int tIdx) {
  splitBits->consume(fac, bitEnd);
  treeCap(tIdx);
}


void ForestTrain::nonTerminal(IndexT nodeIdx,
                              IndexT lhDel,
                              const Crit& crit) {
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


void NBCresc::branchProduce(unsigned int nodeIdx,
                            IndexT lhDel,
                            const struct Crit& crit) {
  treeNode[treeFloor + nodeIdx].setBranch(lhDel, crit);
}


void NBCresc::leafProduce(unsigned int nodeIdx,
                          unsigned int leafIdx) {
  treeNode[treeFloor + nodeIdx].setLeaf(leafIdx);
}



