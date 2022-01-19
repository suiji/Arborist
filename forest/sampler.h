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

#include "util.h"
#include "jagged.h"
#include "bv.h"
#include "typeparam.h"
#include "leaf.h"
#include "sample.h"

#include <memory>
#include <vector>

using namespace std;

/**
   @brief Rank and sample-counts associated with sampled rows.

   Client:  quantile inference.
 */
class RankCount {
  // When sampling is not weighted, the sample-count value typically
  // requires four bits or fewer.  Packing therefore accomodates rank
  // values well over 32 bits.
  PackedT packed; // Packed representation of rank and sample count.

  static unsigned int rightBits; // # bits occupied by rank value.
  static PackedT rankMask; // Mask unpacking the rank value.

public:

  /**
     @brief Invoked at Sampler construction, as needed.
   */
  static void setMasks(IndexT nObs) {
    rightBits = Util::packedWidth(nObs);
    rankMask = (1 << rightBits) - 1;
  }


  /**
     @brief Invoked at Sampler destruction.
   */
  static void unsetMasks() {
    rightBits = 0;
    rankMask = 0;
  }
  

  /**
     @brief Packs statistics associated with a response.

     @param rank is the rank of the response value.

     @param sCount is the number of times the observation was sampled.
   */
  void init(IndexT rank,
            IndexT sCount) {
    packed = rank | (sCount << rightBits);
  }

  IndexT getRank() const {
    return packed & rankMask;
  }


  IndexT getSCount() const {
    return packed >> rightBits;
  }
};


class SamplerNux {
  // As with RankCount, unweighted sampling typically incurs very
  // small sample counts and row deltas, leaving well over 32
  // bits for leaf indices.
  PackedT packed;

public:
  static IndexT delWidth;
  static PackedT delMask;
  static PackedT leafMask;
  static IndexT rightBits;


  static void unsetMasks() {
    delMask = 0;
    delWidth = 0;
    leafMask = 0;
    rightBits = 0;
  }
  
  SamplerNux(IndexT delRow,
	     IndexT leafIdx,
	     IndexT sCount) :
    packed(delRow | (static_cast<PackedT>(leafIdx) << delWidth) | (static_cast<PackedT>(sCount) << rightBits)) {
  }

  
  static void setMasks(IndexT nObs,
		       IndexT nSamp) {
    delWidth = Util::packedWidth(nObs);
    delMask = (1ull << delWidth) - 1;
    unsigned int leafWidth = Util::packedWidth(nSamp);
    leafMask = (1ull << leafWidth) - 1;
    rightBits = delWidth + leafWidth;
  }


  /**
     @return difference in adjacent row numbers.  Always < nObs.
   */
  inline auto getDelRow() const {
    return packed & delMask;
  }
  

  /**
     @return leaf index within tree.
   */
  inline auto getLeafIdx() const {
    return (packed >> delWidth) & leafMask;
  }

  /**
     @return sample count
   */  
  inline auto getSCount() const {
    return packed >> rightBits;
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

  
  SamplerBlock(const SamplerNux* samples,
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

  
  /**
     @brief Enumerates the number of samples at each leaf's category.

     'probSample' is the only client.

     @return forest-wide vector of category counts, by leaf.
   */
  vector<IndexT> countLeafCtg(const class Sampler* sampler,
			      const LeafCtg* leaf) const;

    
  vector<RankCount> countLeafRanks(const class Sampler* sample,
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


class Sampler {
  const unsigned int nTree;
  const size_t nObs; // # training observations
  const size_t nSamp;  // # samples requested per tree.
  const PredictorT nCtg; // Cardinality of training response.
  const bool bagging; // Whether bagging required.
  const bool nuxSamples;  // Whether SamplerNux are emited/read:  training/prediction.
  
  const unique_ptr<class Leaf> leaf;
  unique_ptr<class BitMatrix> bagMatrix; // Empty if samplerBlock empty.
  unique_ptr<SamplerBlock> samplerBlock;

  // Crescent only:
  vector<IndexT> leafExtent; // Per-tree leaf count over block.
  vector<SamplerNux> sbCresc; // Crescent block.
  unique_ptr<class Sample> sample; // Reset at each tree.
  unsigned int tIdx; // Block-relative index of current tree.


  /**
     @brief Constructs bag according to encoding.
   */
  static unique_ptr<BitMatrix> bagRaw(unsigned char* raWSamples,
				      bool nuxSamples,
				      bool bagging,
				      unsigned int nTree,
				      IndexT nObs);

  /**
     @brief Surveys leaf contents.

     @return SamplerBlock constructed from internal survey.
   */
  unique_ptr<SamplerBlock> readRaw(unsigned char* samplesRaw);


public:
  size_t leafCount;
  vector<size_t> leafBase; // Per-tree offset of leaf.


  bool isBagging() const {
    return bagging;
  }

  
  class Sample* getSample() const;

  
  void rootSample(const class TrainFrame* frame);


  /**
     @brief Copies samples to the block, if 'thin' not specified.
   */
  void blockSamples(const vector<IndexT>& leafMap);


  /**
     @brief Computes # bytes subsumed by samples.
   */
  size_t getBlockBytes() const;
  
  
  /**
     @brief Records multiplicity and leaf index for bagged samples
     within a tree.  Accessed by bag vector, so sample indices must
     reference consecutive bagged rows.
     @param leafMap maps sample indices to leaves.
  */
  void bagLeaves(const class Sample *sample,
                 const vector<IndexT> &leafMap,
		 unsigned int tIdx);


  /**
     @brief Generic entry for serialization.
   */
  void dumpRaw(unsigned char snRaw[]) const; 



  /**
     @brief Serializes sampler block.
   */
  void dumpNuxRaw(unsigned char bagRaw[]) const; 

  
  /**
     Classification constructor:  training.
   */
  Sampler(const vector<PredictorT>& yTrain,
	  bool nuxSamples_,
	  IndexT nSamp_,
	  unsigned int treeChunk,
	  PredictorT nCtg_,
	  const vector<double>& classWeight_,
	  bool bagging_ = true);


  /**
     @brief Classification constructor:  post training.
   */
  Sampler(const vector<PredictorT>& yTrain,
	  bool nux,
	  unsigned char* samples,
	  IndexT nSamp_,
	  unsigned int nTree_,
	  PredictorT nCtg_,
	  bool bagging_);


  /**
     @brief Regression constructor: training.
   */
  Sampler(const vector<double>& yTrain,
	  bool nuxSamples_,
	  IndexT nSamp_,
	  unsigned int treeChunk,
	  bool bagging_ = true);

  
  /**
     @brief Regression constructor:  post-training.
   */
  Sampler(const vector<double>& yTrain,
	  bool nuxSamples_,
	  unsigned char* samples,
	  IndexT nSamp_,
	  unsigned int nTree_,
	  bool bagging_);

  ~Sampler() {
    RankCount::unsetMasks();
    SamplerNux::unsetMasks();
  }

  
  /**
     @brief Derives the forest index of a leaf.
   */
  inline size_t absLeafIdx(unsigned int tIdx,
			   IndexT leafIdx) const {
    return leafBase[tIdx] + leafIdx;
  }
  
  
  const Leaf* getLeaf() const {
    return leaf.get();
  }


  auto getNSamp() const {
    return nSamp;
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

     @return true iff bagging and the coordinate bit is set.
   */
  inline bool isBagged(unsigned int tIdx, size_t row) const {
    return bagging && bagMatrix->testBit(tIdx, row);
  }

  
  bool hasSamples() const {
    return samplerBlock != nullptr && samplerBlock->size() != 0;
  }

  
  /**
     @brief Counts samples at each leaf, by category.

     @return per-leaf vector enumerating samples at each category.
   */
  vector<IndexT> countLeafCtg(//const class Predict* predict,
			      const LeafCtg* leaf) const {
    return hasSamples() ? samplerBlock->countLeafCtg(this, /*predict,*/ leaf) : vector<IndexT>(0);
  }
  
  
  /**
     @brief Count samples at each rank, per leaf.

     Meant for regression.

     @param row2Rank is the ranked training outcome.

     @return per-leaf vector expressing mapping.
   */
  vector<RankCount> countLeafRanks(//const class Predict* predict,
				   const vector<IndexT>& row2Rank) const {
    return samplerBlock->countLeafRanks(this, /*predict,*/ row2Rank);
  }


  /**
     @brief Wrapper for call on samplerBlock.
   */
  void getSampleBounds(unsigned int tIdx,
		       IndexT leafIdx,
		       size_t& start,
		       size_t& end) const {
    samplerBlock->getSampleBounds(absLeafIdx(tIdx, leafIdx), start, end);
  }
};


#endif
