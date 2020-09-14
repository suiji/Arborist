// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leafpredict.cc

   @brief Methods for validation and prediction.

   @author Mark Seligman
 */

#include "leafpredict.h"
#include "leaf.h"
#include "predict.h"
#include "bag.h"


LeafBlock::LeafBlock(unsigned int nTree,
                     const unsigned int* height,
                     const Leaf* leaf) :
  raw(make_unique<JaggedArray<const Leaf*, const unsigned int*> >(nTree, height, leaf)) {
}


size_t LeafBlock::size() const {
  return raw->size();
}


unsigned int LeafBlock::nTree() const {
  return raw->getNMajor();
}
  

vector<size_t> LeafBlock::setOffsets() const {
  vector<size_t> offset(raw->size());
  unsigned int countAccum = 0;
  unsigned int idx = 0;
  for (auto & off : offset) {
    off = countAccum;
    countAccum += getExtent(idx++);
  }

  return offset;
  // Post-condition:  countAccum == total bag size.
}


/**
     @brief Index-parametrized score getter.

     @param idx is the absolute index of a leaf.

     @return score at leaf.
   */

double LeafBlock::getScore(unsigned int idx) const {
  return raw->items[idx].getScore();
}


size_t LeafBlock::absOffset(unsigned int tIdx, IndexT leafIdx) const {
  return raw->absOffset(tIdx, leafIdx);
}


unsigned int LeafBlock::treeBase(unsigned int tIdx) const {
  return raw->majorOffset(tIdx);
}


double LeafBlock::getScore(unsigned int tIdx, IndexT idx) const {
  size_t absOff = raw->absOffset(tIdx, idx);
  return raw->items[absOff].getScore();
}


unsigned int LeafBlock::getExtent(size_t leafAbs) const {
  return raw->items[leafAbs].getExtent();
}


void LeafBlock::dump(vector<vector<double> >& score,
                     vector<vector<unsigned int> >& extent) const {
  size_t idx = 0;
  for (auto tIdx = 0ul; tIdx < raw->getNMajor(); tIdx++) {
    for (; idx < raw->height[tIdx]; idx++) {
      score[tIdx].push_back(getScore(idx));
      extent[tIdx].push_back(getExtent(idx));
    }
  }
}


BLBlock::BLBlock(unsigned int nTree,
                 const unsigned int* height,
                 const BagSample* bagSample) :
  raw(make_unique<JaggedArray<const BagSample*, const unsigned int*> >(nTree, height, bagSample)) {
}
                     

size_t BLBlock::size() const {
  return raw->size();
}


const IndexT BLBlock::getSCount(size_t absOff) const {
  return raw->items[absOff].getSCount();
}


const IndexT BLBlock::getLeafIdx(size_t absOff) const {
  return raw->items[absOff].getLeafIdx();
}


void BLBlock::dump(const Bag* bag,
                   vector<vector<size_t> >& rowTree,
                   vector<vector<IndexT> >& sCountTree) const {
  size_t bagIdx = 0;
  const BitMatrix* baggedRows(bag->getBitMatrix());
  for (auto tIdx = 0ul; tIdx < raw->getNMajor(); tIdx++) {
    for (auto row = 0ul; row < baggedRows->getStride(); row++) {
      if (baggedRows->testBit(tIdx, row)) {
        rowTree[tIdx].emplace_back(row);
        sCountTree[tIdx].emplace_back(getSCount(bagIdx++));
      }
    }
  }
}


LeafPredict::LeafPredict(const unsigned int* leafHeight,
			 unsigned int nTree,
			 const Leaf* leaf,
			 const unsigned int* bagHeight,
			 const BagSample *bagSample) :
  leafBlock(make_unique<LeafBlock>(nTree, leafHeight, leaf)),
  blBlock(make_unique<BLBlock>(nTree, bagHeight, bagSample)),
  offset(leafBlock->setOffsets()) {
}


LeafPredict::~LeafPredict() {
}


void LeafPredict::dump(const Bag* bag,
		       vector< vector<size_t> >& rowTree,
		       vector< vector<IndexT> >& sCountTree,
		       vector<vector<double> >& scoreTree,
		       vector<vector<unsigned int> >& extentTree) const {
  if (bag != nullptr) {
    blBlock->dump(bag, rowTree, sCountTree);
  }
  leafBlock->dump(scoreTree, extentTree);
  //  ctgProb->dump(probTree); TODO:  move elsewhere.
}


  /**
     @brief Accessor for #samples at an absolute bag index.
   */
IndexT LeafPredict::getSCount(IndexT bagIdx) const {
  return blBlock->getSCount(bagIdx);
}


  /**
     @param absSIdx is an absolute bagged sample index.

     @return tree-relative leaf index of bagged sample.
   */

auto LeafPredict::getLeafLoc(unsigned int absSIdx) const {
  return blBlock->getLeafIdx(absSIdx);
}

  /**
     @brief Accessor for forest-relative leaf index .

     @param tIdx is the tree index.

     @param absSIdx is a forest-relative sample index.

     @return forest-relative leaf index.
   */

size_t LeafPredict:: getLeafAbs(unsigned int tIdx,
				unsigned int absSIdx) const {
  return leafBlock->absOffset(tIdx, getLeafLoc(absSIdx));
}


  /**
     @brief Determines inattainable leaf index value from leaf
     vector.

     @return inattainable leaf index value.
   */
size_t LeafPredict::getNoLeaf() const {
  return leafBlock->size();
}


  /**
     @brief computes total number of leaves in forest.

     @return size of leaf vector.
   */

size_t LeafPredict::leafCount() const {
  return leafBlock->size();
}


LeafBlock* LeafPredict::getLeafBlock() const {
  return leafBlock.get();
}


void LeafPredict::bagBounds(unsigned int tIdx,
                        IndexT leafIdx,
                        size_t& start,
                        size_t& end) const {
  size_t leafAbs = leafBlock->absOffset(tIdx, leafIdx);
  start = offset[leafAbs];
  end = start + leafBlock->getExtent(leafAbs);
}


vector<RankCount> LeafPredict::setRankCount(const BitMatrix* baggedRows,
					    const vector<IndexT>& row2Rank) const {
  vector<RankCount> rankCount(blBlock->size());
  if (baggedRows->isEmpty())
    return rankCount; // Short circuits with empty vector.

  vector<unsigned int> leafSeen(leafCount());
  unsigned int bagIdx = 0;  // Absolute sample index.
  for (unsigned int tIdx = 0; tIdx < baggedRows->getNRow(); tIdx++) {
    for (IndexT row = 0; row < row2Rank.size(); row++) {
      if (baggedRows->testBit(tIdx, row)) {
        size_t leafAbs = getLeafAbs(tIdx, bagIdx);
        size_t sIdx = offset[leafAbs] + leafSeen[leafAbs]++;
        rankCount[sIdx].init(row2Rank[row], getSCount(bagIdx));
        bagIdx++;
      }
    }
  }

  return rankCount;
}
