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
  // small sample counts and row deltas.
  PackedT packed;

public:
  static PackedT delMask;
  static IndexT rightBits;


  static void unsetMasks() {
    delMask = 0;
    rightBits = 0;
  }
  
  SamplerNux(IndexT delRow,
	     IndexT sCount) :
    packed(delRow | (static_cast<PackedT>(sCount) << rightBits)) {
  }

  
  static void setMasks(IndexT nObs) {
    rightBits = Util::packedWidth(nObs);
    delMask = (1ull << rightBits) - 1;
   }


  /**
     @return difference in adjacent row numbers.  Always < nObs.
   */
  inline auto getDelRow() const {
    return packed & delMask;
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

  
  SamplerBlock(const Sampler* sampler,
	       const SamplerNux* samples,
	       const vector<size_t>& height);


  /**
     @bool nuxSamples is true iff matrix is to be rebagged.
   */
  void bagRows(class BitMatrix* bagMatrix,
	       bool nuxSamples);
  
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
     @brief Enumerates the number of samples at each leaf's category.

     'probSample' is the only client.

     @return 3-d vector category counts, indexed by tree/leaf/ctg.
   */
  vector<vector<vector<size_t>>> countLeafCtg(const class Sampler* sampler,
					      const LeafCtg* leaf) const;


  /**
     @return 3-d vector of rank counts, indexed by tree/leaf/offset.
   */
  vector<vector<vector<RankCount>>> alignRanks(const class Sampler* sample,
					       const vector<IndexT>& row2Rank) const;


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
  const vector<size_t> bagCount; // nonempty only at prediction.

  // extent, index only nonempty at prediction.  Move to Leaf.
  const vector<vector<size_t>> extent; // # sample index entries per leaf, per tree.
  const vector<vector<vector<size_t>>> index; // sample indices per leaf, per tree.
  const unique_ptr<class Leaf> leaf;

  const unique_ptr<class BitMatrix> bagMatrix; // Empty if samplerBlock empty.
  const unique_ptr<SamplerBlock> samplerBlock;

  // Crescent only:
  vector<IndexT> indexCresc; // Sample indices within leaves.
  vector<IndexT> extentCresc; // Index extent, per leaf.
  vector<SamplerNux> sbCresc; // Crescent block.


  /**
     @return bag count of each tree.
   */
  vector<size_t> countSamples(const unsigned char rawSamples[]) const;

  
  vector<vector<size_t>> unpackExtent(const double extentNum[]) const;

  
  vector<vector<vector<size_t>>> unpackIndex(const double indexNum[]) const;

  /**
     @brief Constructs bag according to encoding.
   */
  static unique_ptr<BitMatrix> bagRaw(const unsigned char raWSamples[],
				      bool nuxSamples,
				      bool bagging,
				      unsigned int nTree,
				      IndexT nObs);

  /**
     @brief Surveys leaf contents.

     @return SamplerBlock constructed from internal survey.
   */
  unique_ptr<SamplerBlock> readRaw(const unsigned char samplesRaw[]);


public:

  ~Sampler() {
    RankCount::unsetMasks();
    SamplerNux::unsetMasks();
  }

  
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


  static unique_ptr<Sampler> trainCtg(const vector<PredictorT>& yTrain,
				      bool nuxSamples,
				      IndexT nSamp,
				      unsigned int treeChunk,
				      PredictorT nCtg,
				      const vector<double>& classWeight);


  /**
     @brief Regression constructor: training.
   */
  Sampler(const vector<double>& yTrain,
	  bool nuxSamples_,
	  IndexT nSamp_,
	  unsigned int treeChunk,
	  bool bagging_ = true);

  
  static unique_ptr<Sampler> trainReg(const vector<double>& yTrain,
				      bool nuxSamples,
				      IndexT nSamp,
				      unsigned int treeChunk);


  /**
     @brief Classification constructor:  post training.
   */
  Sampler(const vector<PredictorT>& yTrain,
	  bool nux,
	  const unsigned char samples[],
	  IndexT nSamp_,
	  unsigned int nTree_,
	  const double extentNum[],
	  const double indexNum[],
	  PredictorT nCtg_,
	  bool bagging_);


  static unique_ptr<Sampler> predictCtg(const vector<PredictorT>& yTrain,
					bool nux,
					const unsigned char samples[],
					IndexT nSamp,
					unsigned int nTree,
					const double extentNum[],
					const double indexNum[],
					PredictorT nCtg,
					bool bagging);


  /**
     @brief Regression constructor:  post-training.
   */
  Sampler(const vector<double>& yTrain,
	  bool nuxSamples_,
	  const unsigned char samples[],
	  IndexT nSamp_,
	  unsigned int nTree_,
	  const double extentNum[],
	  const double indexNum[],
	  bool bagging_);

  
  /**
     @brief Static entry from bridge; initializes masks.
   */
  static unique_ptr<Sampler> predictReg(const vector<double>& yTrain,
			    bool nuxSamples,
			    const unsigned char samples[],
			    IndexT nSamp,
			    unsigned int nTree,
			    const double extentNum[],
			    const double indexNum[],
			    bool bagging);


  size_t getLeafCount(unsigned int tIdx) const {
    return extent[tIdx].size();
  }

  
  size_t getBagCount(unsigned int tIdx) const {
    return bagCount[tIdx];
  }
  

  const vector<size_t>& getExtents(unsigned int tIdx) const {
    return extent[tIdx];
  }


  const vector<vector<size_t>>& getIndices(unsigned int tIdx) const {
    return index[tIdx];
  }
  
  
  bool isBagging() const {
    return bagging;
  }


  unique_ptr<class Sample> rootSample(const class TrainFrame* frame,
				unsigned int tIdx);


  /**
     @brief Copies samples to the block, if 'noLeaf' not specified.
   */
  void consumeSamples(const class PreTree* pretree,
		      const class SampleMap& smTerminal);


  /**
     @brief Computes # bytes subsumed by samples.
   */
  size_t crescBlockBytes() const;

  size_t crescExtentSize() const {
    return extentCresc.size();
  }
  

  size_t crescIndexSize() const{
    return indexCresc.size();
  }


  /**
     @brief Generic entry for serialization.
   */
  void dumpRaw(unsigned char snRaw[]) const; 



  /**
     @brief Serializes sampler block.
   */
  void dumpNuxRaw(unsigned char bagRaw[]) const; 


  void dumpIndex(double indexOut[]) const;

  
  void dumpExtent(double extentOut[]) const;
  
  
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
     @brief Counts samples at each leaf in the forest, by category.

     @return per-leaf vector enumerating samples at each category.
   */
  vector<vector<vector<size_t>>> countLeafCtg(const LeafCtg* leaf) const {
    return hasSamples() ? samplerBlock->countLeafCtg(this, leaf) : vector<vector<vector<size_t>>>(0);
  }
  
  
  /**
     @brief Count samples at each rank, per leaf.

     Meant for regression.

     @param row2Rank is the ranked training outcome.

     @return per-leaf vector expressing mapping.
   */
  vector<vector<vector<RankCount>>> alignRanks(const vector<IndexT>& row2Rank) const {
    return hasSamples() ? samplerBlock->alignRanks(this, row2Rank) : vector<vector<vector<RankCount>>>(0);
  }
};


#endif
