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
#include "sample.h"

#include <memory>
#include <vector>

using namespace std;

class SamplerNux {
  // As with RankCount, unweighted sampling typically incurs very
  // small sample counts and row deltas.
  PackedT packed;

public:
  static PackedT delMask;
  static IndexT rightBits;


  static void setMasks(IndexT nObs) {
    rightBits = Util::packedWidth(nObs);
    delMask = (1ull << rightBits) - 1;
  }
  

  static void unsetMasks() {
    delMask = 0;
    rightBits = 0;
  }


  /**
     @brief Constructor for external packed value.
   */
  SamplerNux(PackedT packed_) :
    packed(packed_) {
  }

  
  SamplerNux(IndexT delRow,
	     IndexT sCount) :
    packed(delRow | (static_cast<PackedT>(sCount) << rightBits)) {
  }

  
  /**
     @return difference in adjacent row numbers.  Always < nObs.
   */
  inline IndexT getDelRow() const {
    return packed & delMask;
  }
  

  /**
     @return sample count
   */  
  inline IndexT getSCount() const {
    return packed >> rightBits;
  }

  
  /**
     @brief Obtains sample count for external packed value.
   */
  static IndexT getSCount(PackedT packed) {
    return packed >> rightBits;
  }


  PackedT getPacked() const {
    return packed;
  }
};


class Sampler {
  // Experimental coarse-grained control of locality:  Not quite
  // coding-to-cache, but almost.
  static constexpr unsigned int locExp = 18;  // Log of locality threshold.

  const unsigned int nTree;
  const size_t nObs; // # training observations
  const size_t nSamp;  // # samples requested per tree.
  const bool bagging; // Whether bagging required.

  const unique_ptr<class Response> response;
  
  const vector<vector<SamplerNux>> samples;
  const unique_ptr<class BitMatrix> bagMatrix; // null iff training

  // Crescent only:
  vector<IndexT> sCountRow; // Temporary proxy for sbCresc.
  vector<SamplerNux> sbCresc; // Crescent block.

  
  /**
     @brief Constructs bag according to encoding.
   */
  unique_ptr<BitMatrix> bagRows();


  /**
     @brief Maps an index into its bin.

     @param idx is the index in question.

     @return bin index.
   */
  static constexpr unsigned int binIdx(IndexT idx) {
    return idx >> locExp;
  }
  

  /**
     @brief Bins a vector of indices for coarse locality.  Equivalent to
     the first pass of a radix sort.

     @param idx is an unordered vector of indices.

     @return binned version of index vector passed.
   */
  static vector<unsigned int> binIndices(const vector<IndexT>& idx);


  /**
     @brief Tabulates a collection of indices by occurrence.

     @param sampleCount tabulates the occurrence count of each index.

     @return vector of sample counts.
   */
  static vector<IndexT> countSamples(IndexT nRow,
				     IndexT nSamp);

public:

  /**
     Classification constructor:  training.
   */
  Sampler(const vector<PredictorT>& yTrain,
	  IndexT nSamp_,
	  unsigned int treeChunk,
	  PredictorT nCtg,
	  const vector<double>& classWeight_,
	  bool bagging_ = true);

  
  ~Sampler();

  
  /**
     @brief Regression constructor: training.
   */
  Sampler(const vector<double>& yTrain,
	  IndexT nSamp_,
	  unsigned int treeChunk,
	  bool bagging_ = true);


  /**
     @brief Classification constructor:  post training.
   */
  Sampler(const vector<PredictorT>& yTrain,
	  vector<vector<SamplerNux>> samples_,
	  IndexT nSamp_,
	  PredictorT nCtg,
	  bool bagging_);


  /**
     @brief Regression constructor:  post-training.
   */
  Sampler(const vector<double>& yTrain,
	  vector<vector<SamplerNux>> samples_,
	  IndexT nSamp_,
	  bool bagging_);


  const vector<IndexT>& getSampledRows() const {
    return sCountRow;
  }


  /**
     @brief Two-coordinat lookup of sample count.
   */
  IndexT getSCount(unsigned int tIdx,
		   IndexT sIdx) const {
    return samples[tIdx][sIdx].getSCount();
  }


  /**
     @brief As above, but row delta.
   */
  size_t getDelRow(unsigned int tIdx,
		   IndexT sIdx) const {
    return samples[tIdx][sIdx].getDelRow();
  }


  size_t getBagCount(unsigned int tIdx) const {
    return samples[tIdx].size();
  }
  

  bool isBagging() const {
    return bagging;
  }


  unique_ptr<class Sample> rootSample(unsigned int tIdx);


  /**
     @brief Computes # records subsumed by sampling this chunk.
   */
  size_t crescBagCount() const {
    return sbCresc.size();
  }


  void dumpNux(double sampleOut[]) const {
    for (size_t i = 0; i < sbCresc.size(); i++) {
      sampleOut[i] = sbCresc[i].getPacked();
    }
  }

  
  const class Response* getResponse() const {
    return response.get();
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


  /**
     @brief Determines whether a given forest coordinate is bagged.

     @param tIdx is the tree index.

     @param row is the row index.

     @return true iff bagging and the coordinate bit is set.
   */
  inline bool isBagged(unsigned int tIdx, size_t row) const {
    return bagging && bagMatrix->testBit(tIdx, row);
  }

  
  /**
     @brief Indicates whether block can be used for enumeration.

     @return true iff block is nonempty.
   */  
  bool hasSamples() const {
    return !samples.empty();
  }

};

#endif
