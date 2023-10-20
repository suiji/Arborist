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

#include "idcount.h"
#include "typeparam.h"
#include "sampledobs.h"
#include "sample.h"

#include <memory>
#include <vector>

using namespace std;


class Sampler {
  // Experimental coarse-grained control of locality:  Not quite
  // coding-to-cache, but almost.
  static constexpr unsigned int locExp = 18;  // Log of locality threshold.

  const unsigned int nRep;
  const size_t nObs; ///< # training observations
  const size_t nSamp;  ///< # samples requested per tree.

  const unique_ptr<struct Response> response;
  
  const vector<vector<class SamplerNux>> samples;
  unique_ptr<class Predict> predict; // Training, prediction only.


  // Presampling only:
  bool trivial; ///< Shortcut.
  vector<SamplerNux> sbCresc; ///< Crescent block.
  unique_ptr<Sample::Walker<size_t>> walker; ///< Walker table.
  vector<double> weightNoReplace; ///< Non-replacement weights.
  vector<size_t> coeffNoReplace; ///< Uniform non-replacement coefficients.


  /**
     @brief Maps an index into its bin.

     @param idx is the index in question.

     @return bin index.
   */
  static constexpr unsigned int binIdx(IndexT idx) {
    return idx >> locExp;
  }
  

  /**
     @brief Bins a vector of indices for coarse locality.

     Equivalent to the first pass of a radix sort.

     @param nObs specfies the maximum sampled index.

     @param idx is an unordered vector of indices.

     @return binned version of index vector passed.
   */
  static vector<size_t> binIndices(size_t nObs,
				   const vector<size_t>& idx);


  /**
     @brief Tabulates a collection of indices by occurrence.

     @param sampleCount tabulates the occurrence count of each index.

     @return vector of sample counts.
   */
  vector<IndexT> countSamples(const vector<size_t>& idx);
  

public:

  /**
     @brief Sampling constructor.
   */
  Sampler(IndexT nSamp_,
	  IndexT nObs_,
	  unsigned int nRep_,
	  bool replace_,
	  const double weight[]);


  /**
     @brief Generic constructor, no response.
   */
  Sampler(IndexT nObs_,
	  IndexT nSamp_,
	  const vector<vector<SamplerNux>>& samples_);


  /**
     Classification constructor:  training.
   */
  Sampler(const vector<PredictorT>& yTrain,
	  IndexT nSamp_,
	  vector<vector<SamplerNux>> nux,
	  PredictorT nCtg,
	  const vector<double>& classWeight_);

  
  ~Sampler();

  
  /**
     @brief Regression constructor: training.
   */
  Sampler(const vector<double>& yTrain,
	  IndexT nSamp_,
	  vector<vector<SamplerNux>> nux);


  /**
     @brief Classification constructor:  post training.
   */
  Sampler(const vector<PredictorT>& yTrain,
	  vector<vector<SamplerNux>> samples_,
	  IndexT nSamp_,
	  PredictorT nCtg,
	  unique_ptr<struct RLEFrame> rleFrame);


  /**
     @brief Regression constructor:  post-training.
   */
  Sampler(const vector<double>& yTrain,
	  vector<vector<SamplerNux>> samples_,
	  IndexT nSamp_,
	  unique_ptr<struct RLEFrame> rleFrame);


  /**
     @brief Samples response for a single tree.
   */
  void appendSamples(const vector<size_t>& idx);


  const vector<SamplerNux>& getSamples(unsigned int tIdx) const {
    return samples[tIdx];
  }


  /**
     @brief Expands SamplerNux vector for a single tree.

     @param tIdx is the tree index.
     
     @return vector of unpacked SamplerNux.
   */
  vector<IdCount> unpack(unsigned int tIdx) const {
    vector<IdCount> idCount;
    IndexT obsIdx = 0;
    for (SamplerNux nux : samples[tIdx]) {
      obsIdx += nux.getDelRow();
      idCount.emplace_back(obsIdx, nux.getSCount());
    }

    return idCount;
  }


  /**
     @brief Constructs bag according to encoding.
   */
  unique_ptr<class BitMatrix> bagRows(bool bagging) const;


  IndexT getExtent(unsigned int tIdx) const {
    return samples[tIdx].size();
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


  /**
     @brief Empty vector iff trivial:  nObs == nSamp.
     
     @return # unique samples at rep index.
   */
  size_t getBagCount(unsigned int repIdx) const {
    return samples[repIdx].empty() ? nSamp : samples[repIdx].size();
  }


  /**
     @brief Passes through to Response method.
   */
  unique_ptr<class SampledObs> getObs(unsigned int tIdx) const;

  
  /**
     @brief Computes # records subsumed by sampling this block.

     @return sum of each tree's bag count.
   */
  size_t crescCount() const {
    return sbCresc.size();
  }


  void dumpNux(double sampleOut[]) const {
    for (size_t i = 0; i < sbCresc.size(); i++) {
      sampleOut[i] = sbCresc[i].getPacked();
    }
  }

  
  const struct Response* getResponse() const {
    return response.get();
  }


  /**
     @brief Passes through to response.
   */
  CtgT getNCtg() const;


  auto getNSamp() const {
    return nSamp;
  }
  

  auto getNObs() const {
    return nObs;
  }


  auto getNRep() const {
    return nRep;
  }

  
  /**
     @brief Initializes coefficients specialized for sampling type.
   */
  void setCoefficients(const double weight[],
		       bool replace);

  
  /**
     @brief Samples a single tree's worth of observations.
   */
  void sample();


  /**
     @brief Indicates whether block can be used for enumeration.

     @return true iff block is nonempty.
   */  
  bool hasSamples() const {
    return !samples.empty();
  }


  /**
     @brief Decompresses a tree's worth of samples into observations.
     
     @param tIdx is the absolute tree index.
     
     @return vector of observation indices, counts.
   */
  vector<IdCount> obsExpand(const vector<SampleNux>& nuxen) const;


  /**
     @brief Pass-through to Predict member functions of the same name.
   */
  unique_ptr<struct SummaryReg> predictReg(class Forest* forest,
					   const vector<double>& yTest) const;


  unique_ptr<struct SummaryCtg> predictCtg(class Forest* forest,
					   const vector<unsigned int>& yTest) const;
};

#endif
