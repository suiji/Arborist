// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leafpredict.h

   @brief Class definitions for terminal manipulation during prediction.

   @author Mark Seligman
 */

#ifndef TREE_LEAFPREDICT_H
#define TREE_LEAFPREDICT_H

#include "jagged.h"
#include "typeparam.h"

#include <vector>

class LeafBlock {
  const unique_ptr<JaggedArrayV<const class Leaf*, size_t>> raw;

public:
  LeafBlock(const vector<size_t>& height_,
            const Leaf* leaf_);

  /**
   @brief Gets the size of the jagged array.

   @return total number of leaves in the block.
  */
  size_t size() const {
    return raw->size();
  }


  unsigned int nTree() const {
    return raw->getNMajor();
  }

  
  /**
     @brief Accessor for height vector.
   */
  size_t getHeight(unsigned int tIdx) const {
    return raw->getHeight(tIdx);
  }

  
  /**
     @brief Accumulates individual leaf extents across the forest.

     @return forest-wide offset vector.
   */
  vector<size_t> setOffsets() const;


  /**
     @brief Scores a categorical row across all trees.

     @param ctgDefault is the default category if all trees in-bag.

     @param[out] yCtg[] outputs per-category scores over a block of rows.
   */
  void scoreCtg(const class RowBlock* rowBlock,
		size_t row,
		unsigned int ctgDefault,
		double yCtg[]) const;

  /**
     @brief Index-parametrized score getter.

     @param idx is the absolute index of a leaf.

     @return score at leaf.
   */
  double getScore(unsigned int idx) const;


  /**
     @brief Derives forest-relative offset of tree/leaf coordinate.

     @param tIdx is the tree index.

     @param leafIdx is a tree-local leaf index.

     @return absolute offset of leaf.
   */
  size_t absOffset(unsigned int tIdx, IndexT leafIdx) const;


  /**
     @return beginning leaf offset for tree.
   */
  unsigned int treeBase(unsigned int tIdx) const;


  /**
     @brief Coordinate-parametrized score getter.

     @param tIdx is the tree index.

     @param idx is the tree-relative index of a leaf.

     @return score at leaf.
   */
  double getScore(unsigned int tIdx, IndexT idx) const;


  /**
     @brief Derives count of samples assigned to a leaf.

     @param leafIdx is the absolute leaf index.

     @return extent value.
   */
  IndexT getExtent(size_t leafAbs) const;


  /**
     @brief Dumps leaf members into separate per-tree vectors.

     @param[out] score ouputs per-tree vectors of leaf scores.

     @param[out] extent outputs per-tree vectors of leaf extents.
   */
  void dump(vector<vector<double> >& score,
            vector<vector<unsigned int> >& extent) const;
};


/**
   @brief Jagged vector of bagging summaries.
 */
class BLBlock {
  const unique_ptr<JaggedArrayV<const class BagSample*, size_t> > raw;

public:
  BLBlock(const vector<size_t>& height_,
          const BagSample* bagSample_);

  /**
     @brief Derives size of raw contents.
   */
  size_t size() const;


  void dump(const class Bag* bag,
            vector<vector<size_t> >& rowTree,
            vector<vector<IndexT> >& sCountTree) const;


  /**
     @brief Index-parametrized sample-count getter.
   */
  const IndexT getSCount(size_t absOff) const;


  /**
     @brief Index-parametrized leaf-index getter.

     @param absOff is the forest-relative bag offset.

     @return associated tree-relative leaf index.
   */
  const IndexT getLeafIdx(size_t absOff) const;
};


/**
   @brief Rank and sample-counts associated with bagged rows.

   Client:  quantile inference.
 */
struct RankCount {
  IndexT rank; // Training rank of row.
  IndexT sCount; // # times row sampled.

  void init(IndexT rank,
            IndexT sCount) {
    this->rank = rank;
    this->sCount = sCount;
  }
};


/**
   @brief Encapsulates trained leaves for prediction.
 */
class LeafPredict {

protected:
  unique_ptr<class LeafBlock> leafBlock; // Leaves.
  unique_ptr<class BLBlock> blBlock; // Bag-sample summaries.
  vector<size_t> offset; // Accumulated offsets

public:
  LeafPredict(const vector<size_t>& height,
	      const class Leaf* leaf_,
	      const vector<size_t>& bagHeight_,
	      const class BagSample* bagSample_);


  virtual ~LeafPredict();

  /**
     @brief Accessor for height vector.
   */
  size_t getHeight(unsigned int tIdx) const {
    return leafBlock->getHeight(tIdx);
  }

  

  const unsigned int getNTree() const {
    return leafBlock->nTree();
  }
  
  
  /**
     @brief Accessor for #samples at an absolute bag index.
   */
  IndexT getSCount(IndexT bagIdx) const;

  /**
     @param absSIdx is an absolute bagged sample index.

     @return tree-relative leaf index of bagged sample.
   */
  auto getLeafLoc(unsigned int absSIdx) const;


  /**
     @brief Accessor for forest-relative leaf index .

     @param tIdx is the tree index.

     @param absSIdx is a forest-relative sample index.

     @return forest-relative leaf index.
   */
   size_t getLeafAbs(unsigned int tIdx,
		     unsigned int absSIdx) const;


  /**
     @brief Determines inattainable leaf index value from leaf
     vector.

     @return inattainable leaf index value.
   */
  size_t getNoLeaf() const;


  /**
     @brief computes total number of leaves in forest.

     @return size of leaf vector.
   */
  size_t leafCount() const;


  LeafBlock* getLeafBlock() const;
  

  /**
     @brief Builds row-ordered mapping of leaves to rank/count pairs.

     @param baggedRows encodes the forest-wide tree bagging.

     @param row2Rank is the ranked training outcome.

     @return per-leaf vector expressing mapping.
   */
  vector<RankCount> setRankCount(const class Bag* bag,
                                 const vector<IndexT>& row2Rank) const;


  /**
     @brief Dumps block components into separate tree-based vectors.
   */
  void dump(const class Bag* bag,
            vector< vector<size_t> >& rowTree,
            vector< vector<IndexT> >& sCountTree,
            vector<vector<double> >& scoreTree,
            vector<vector<unsigned int> >& extentTree) const;


  /**
     @brief Computes bag index bounds in forest setting (Quant only).

     @param tIdx is the tree index.

     @param leafIdx is the tree-relative leaf index.

     @param[out] start outputs the staring sample offset.

     @param[out] end outputs the final sample offset. 
  */
  void bagBounds(unsigned int tIdx,
		 IndexT leafIdx,
		 size_t& start,
		 size_t& end) const;
};

#endif
