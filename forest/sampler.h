// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sampler.h

   @brief Forest-wide packed representation of sampled observations.

   @author Mark Seligman
 */

#ifndef FOREST_SAMPLER_H
#define FOREST_SAMPLER_H

#include "jagged.h"
#include "bv.h"
#include "typeparam.h"
#include "leaf.h"

#include "samplernux.h"

/**
   @brief Rank and sample-counts associated with sampled rows.

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
   @brief Wraps jagged vector of sampler summaries.
 */
class SamplerBlock {
  const unique_ptr<JaggedArrayV<const SamplerNux*, size_t> > raw;
  
public:
  vector<IndexT> sampleExtent; // # samples subsumed per leaf.
  vector<size_t> sampleOffset; // Per-leaf sample offset:  extent patial sums.

  
  SamplerBlock(const SamplerNux* sampls,
	       const vector<size_t>& height,
	       const vector<size_t>& forestIdx);

  
  /**
     @brief Derives size of raw contents.
   */
  size_t size() const {
    return raw->size();
  }

  
  IndexT getHeight(unsigned int tIdx) const {
    return raw->getHeight(tIdx);
  }
  

  /**
     @brief Index-parametrized sample-count getter.
   */
  IndexT getSCount(size_t absOff) const {
    return raw->items[absOff].getSCount();
  }


  /**
     @brief Index-parametrized sample-count getter.
   */
  IndexT getDelRow(size_t absOff) const {
    return raw->items[absOff].getDelRow();
  }


  bool isSampled(size_t absOff,
		 IndexT& sCount) {
    sCount = raw->items[absOff].getSCount();
    return sCount > 0;
  }
  

  /**
     @brief Index-parametrized leaf-index getter.

     @param absOff is the forest-relative bag offset.

     @return associated tree-relative leaf index.
   */
  IndexT getLeafIdx(size_t absOff) const {
    return raw->items[absOff].getLeafIdx();
  }

  vector<IndexT> ctgSamples(const class Predict* predict,
			    const LeafCtg* leaf) const;

    
  vector<RankCount> countLeafRanks(const class Predict* predict,
				   const vector<IndexT>& row2Rank) const;


  /**
     @brief Derives sample boundary coordinates of a leaf.

     @param tIdx is the tree index.

     @param leafIdx is the tree-relative leaf index.

     @param[out] start outputs the staring sample offset.

     @param[out] end outputs the final sample offset. 
  */
  void getSampleBounds(size_t forestIdx,
		       size_t& start,
		       size_t& end) const {
    start = sampleOffset[forestIdx];
    end = start + sampleExtent[forestIdx];
  }

  
  void dump(const class Sampler* sampler,
            vector<vector<size_t> >& rowTree,
            vector<vector<IndexT> >& sCountTree) const;
};


struct Sampler {
  const unsigned int nTree;
  const size_t nObs; // # training observations
  const size_t nSamp;  // # samples requested per tree.
  const PredictorT nCtg; // Cardinality of training response.
  const unique_ptr<class Leaf> leaf;
  unique_ptr<class BitMatrix> bitMatrix;
  const unique_ptr<SamplerBlock> samplerBlock;

  
  /**
     @brief Classification constructor.
   */
  Sampler(const vector<PredictorT>& yTrain,
	  const SamplerNux* samples,
	  unsigned int nTree_,
	  PredictorT nCtg_);


  /**
     @brief Regression constructor.
   */
  Sampler(const vector<double>& yTrain,
	  const SamplerNux* samples,
	  unsigned int nTree_);

  
  /**
     @brief Constructor for empty bag.
   */
  Sampler();

  
  const Leaf* getLeaf() const {
    return leaf.get();
  }
  

  auto getNObs() const {
    return nObs;
  }


  auto getNTree() const {
    return nTree;
  }


  auto getNCtg() const {
    return nCtg;
  }

  
  /**
     @brief Determines whether a given forest coordinate is bagged.

     @param tIdx is the tree index.

     @param row is the row index.

     @return true iff matrix is nonempty and the coordinate bit is set.
   */
  inline bool isBagged(unsigned int tIdx, size_t row) const {
    return nTree != 0 && bitMatrix->testBit(tIdx, row);
  }

  
  bool isEmpty() const {
    return samplerBlock->size() == 0;
  }

  
  /**
     @brief Surveys leaf contents.  Side-effects bit-matrix.

     @return SamplerBlock constructed from internal survey.
   */
  unique_ptr<SamplerBlock> setExtents(const SamplerNux* samples);

  
  class BitMatrix* getBitMatrix() const;


  /**
     @brief Counts samples at each leaf, by category.

     @return per-leaf vector enumerating samples at each category.
   */
  vector<IndexT> ctgSamples(const class Predict* predict,
			    const LeafCtg* leaf) const {
    return samplerBlock->ctgSamples(predict, leaf);
  }
  
  
  /**
     @brief Count samples at each rank, per leaf.

     Meant for regression.

     @param row2Rank is the ranked training outcome.

     @return per-leaf vector expressing mapping.
   */
  vector<RankCount> countLeafRanks(const class Predict* predict,
				   const vector<IndexT>& row2Rank) const {
    return samplerBlock->countLeafRanks(predict, row2Rank);
  }


  /**
     @brief Wrapper for call on samplerBlock.
   */
  void getSampleBounds(size_t forestIdx,
		       size_t& start,
		       size_t& end) const {
    samplerBlock->getSampleBounds(forestIdx, start, end);
  }
};


#endif
