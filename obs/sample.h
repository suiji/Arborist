// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sample.h

   @brief Class definitions for sample-oriented aspects of training.

   @author Mark Seligman
 */

#ifndef OBS_SAMPLE_H
#define OBS_SAMPLE_H

#include "sumcount.h"
#include "typeparam.h"

#include <vector>

#include "samplenux.h" // For now.


/**
 @brief Run of instances of a given row obtained from sampling for an individual tree.
*/
class Sample {
  // Experimental coarse-grained control of locality:  Not quite
  // coding-to-cache, but almost.
  static constexpr unsigned int locExp = 18;  // Log of locality threshold.

  /**
     @brief Maps an index into its bin.

     @param idx is the index in question.

     @return bin index.
   */
  static constexpr unsigned int binIdx(IndexT idx) {
    return idx >> locExp;
  }

  
 protected:
  const IndexT nSamp; // Number of row samples requested.
  const bool bagging; // Whether bagging required.
  vector<SampledNux> sampledNux; // Per-sample summary, with row-delta.
  vector<SumCount> ctgRoot; // Root census of categorical response.
  vector<IndexT> row2Sample; // Maps row index to sample index.
  IndexT bagCount;
  double bagSum; // Sum of bagged responses.

  
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

     @return count of uniquely-sampled elements.
   */
  static IndexT countSamples(vector<IndexT>& sampleCount,
			     IndexT nSamp);

  
  /**
     @brief Samples rows and counts resulting occurrences.

     @param y is the proxy / response:  classification / summary.

     @param yCtg is true response / zero:  classification / regression.
  */
  void bagSamples(const vector<double>& y,
		  const vector<PredictorT>& yCtg);

  
  /**
     @brief As above, but bypasses slow trivial sampling.
   */
  void bagTrivial(const vector<double>& y,
		  const vector<PredictorT>& yCtg);

  
  /**
     @brief Appends summary node to crescent vector.

     @param val is the sum of sampled responses.

     @param sCount is the number of times sampled.

     @param ctg is the category index:  unused if not categorical.
   */
  // Can be specified via pointer-to-member-function i/o virtual.
  virtual double addNode(IndexT delRow,
			 double val,
			 IndexT sCount,
			 PredictorT ctg) = 0;

 public:

  /**
     @brief Static entry for discrete response (classification).

     @param y is a real-valued proxy for the training response.

     @param frame summarizes the ranked observations.

     @param yCtg is the training response.

     @return new SampleCtg instance.
   */
  static unique_ptr<class SampleCtg> factoryCtg(const class Sampler* sampler,
						const vector<double>&  y,
                                                const class TrainFrame *frame,
                                                const vector<PredictorT>& yCtg);

  
  /**
     @brief Static entry for continuous response (regression).

     @param y is the training response.

     @param frame summarizes the ranked observations.

     @return new SampleReg instance.
   */
  static unique_ptr<class SampleReg>factoryReg(const class Sampler* sampler,
					       const vector<double>& y,
                                               const class TrainFrame *frame);


  /**
     @brief Constructor.

     @param frame summarizes predictor ranks by row.
   */
  Sample(const class TrainFrame* frame,
	 const class Sampler* sampler);

  
  /**
     @brief Getter for root category census vector.
   */
  inline const vector<SumCount> getCtgRoot() const {
    return ctgRoot;
  }


  inline auto getNCtg() const {
    return ctgRoot.size();
  }
  
  
  /**
     @brief Getter for user-specified sample count.
   */
  inline IndexT getNSamp() const {
    return nSamp;
  }


  /**
     @brief Getter for bag count:  # uniquely-sampled rows.
   */
  inline IndexT getBagCount() const {
    return bagCount;
  }


  /**
     @brief Getter for sum of bagged responses.
   */
  inline double  getBagSum() const {
    return bagSum;
  }


  /**
     @brief Looks up sample index for sampled row.

     @param[out] sampleIdx is the associated sample index, if sampled.

     @return true iff row is sampled.
   */
  bool isSampled(IndexT row,
		 IndexT& sampleIdx) const {
    sampleIdx = row2Sample[row];
    return sampleIdx < bagCount;
  }
  
  
  /**
     @brief Appends a rank entry for row, if sampled.

     @param row is the row number in question.

     @param[in, out] sIdx addresses the sample index, if any, associated with row.
     @param[in, out] spn is the current sampled rank entry, if any, for row.
  */
  inline void joinRank(IndexT row,
		       IndexT*& sIdx,
		       SampleRank*& spn,
		       IndexT rank) const {
    IndexT smpIdx;
    if (isSampled(row, smpIdx)) {
      *sIdx++ = smpIdx;
      spn++->join(sampledNux[smpIdx], rank);
    }
  }


  /**
     @brief Getter for sample count.

     @param sIdx is the sample index.
   */
  inline IndexT getSCount(IndexT sIdx) const {
    return sampledNux[sIdx].getSCount();
  }


  /**
     @brief Getter for row delta.

     @param sIdx is the sample index.
   */
  inline IndexT getDelRow(IndexT sIdx) const {
    return sampledNux[sIdx].getDelRow();
  }


  /**
     @brief Getter for the sampled response sum.

     @param sIdx is the sample index.
   */
  inline FltVal getSum(IndexT sIdx) const {
    return sampledNux[sIdx].getSum();
  }

  
  /**
     @return response category at index passed.
   */
  inline PredictorT getCtg(IndexT sIdx) const {
    return sampledNux[sIdx].getCtg();
  }
};


/**
   @brief Regression-specific methods and members.
*/
struct SampleReg : public Sample {

  SampleReg(const class TrainFrame* frame,
	    const class Sampler* sampler);


  /**
     @brief Appends regression-style sampling record.

     Parameters as described at virtual declaration.

     @param ctg unused, as response is not categorical.
   */
  inline double addNode(IndexT delRow,
			double yVal,
                        IndexT sCount,
                        PredictorT ctg) {
    sampledNux.emplace_back(delRow, yVal, sCount);
    return sampledNux.back().getSum();
  }
  

  /**
     @brief Inverts the randomly-sampled vector of rows.

     @param y is the response vector.
  */
  void bagSamples(const vector<double>& y);
};


/**
 @brief Classification-specific sampling.
*/
struct SampleCtg : public Sample {
  
  SampleCtg(const class TrainFrame* frame,
	    const class Sampler* sampler);

  
  /**
     @brief Appends a sample summary record.

     Parameters as described in virtual declaration.

     @return sum of sampled response values.
   */
  inline double addNode(IndexT delRow,
			double yVal,
			IndexT sCount,
			PredictorT ctg) {
    sampledNux.emplace_back(delRow, yVal, sCount, ctg);
    double ySum = sampledNux.back().getSum();
    ctgRoot[ctg] += SumCount(ySum, sCount);

    return ySum;
  }
  
  
  /**
     @brief Samples the response, sets in-bag bits.

     @param yCtg is the response vector.

     @param y is the proxy response vector.
  */
  void bagSamples(const vector<PredictorT>& yCtg,
		  const vector<double>& y);
};


#endif
