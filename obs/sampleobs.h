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

#ifndef OBS_SAMPLEOBS_H
#define OBS_SAMPLEOBS_H

#include "sumcount.h"
#include "typeparam.h"
#include "samplenux.h"
#include "obscell.h"

#include <vector>


/**
 @brief Run of instances of a given row obtained from sampling for an individual tree.
*/
class SampleObs {
 protected:
  const IndexT nSamp; // Number of row samples requested.
  double (SampleObs::* adder)(double, const class SamplerNux&, PredictorT);
  vector<SampleNux> sampleNux; // Per-sample summary, with row-delta.
  vector<SumCount> ctgRoot; // Root census of categorical response.
  vector<IndexT> row2Sample; // Maps row index to sample index.
  IndexT bagCount;
  double bagSum; // Sum of bagged responses.

  
  /**
     @brief Samples rows and counts resulting occurrences.

     @param y is the proxy / response:  classification / summary.

     @param yCtg is true response / zero:  classification / regression.
  */
  void bagSamples(const class Sampler* sampler,
		  const vector<double>& y,
		  const vector<PredictorT>& yCtg,
		  unsigned int tIdx);

  
  /**
     @brief As above, but bypasses slow trivial sampling.
   */
  void bagTrivial(const vector<double>& y,
		  const vector<PredictorT>& yCtg);

  
 public:

  /**
     @brief Static entry for categorical response (classification).

     @param y is a real-valued proxy for the training response.

     @param yCtg is the training response.

     @return new SampleCtg instance.
   */
  static unique_ptr<class SampleCtg> factoryCtg(const class Sampler* sampler,
						const class Response* response,
						const vector<double>&  y,
                                                const vector<PredictorT>& yCtg,
						unsigned int tIdx);


  /**
     @brief Static entry for continuous response (regression).

     @param y is the training response.

     @return new SampleReg instance.
   */
  static unique_ptr<class SampleReg>factoryReg(const class Sampler* sampler,
					       const class Response* response,
					       const vector<double>& y,
					       unsigned int tIdx);


  /**
     @brief Constructor.

     @param frame summarizes predictor ranks by row.
   */
  SampleObs(const class Sampler* sampler,
	 const class Response* response,
	 double (SampleObs::* adder_)(double, const class SamplerNux&, PredictorT) = nullptr);

  
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
		       class ObsCell*& spn,
		       IndexT rank) const {
    IndexT smpIdx;
    if (isSampled(row, smpIdx)) {
      *sIdx++ = smpIdx;
      spn++->join(sampleNux[smpIdx], rank);
    }
  }


  /**
     @brief Getter for sample count.

     @param sIdx is the sample index.
   */
  inline IndexT getSCount(IndexT sIdx) const {
    return sampleNux[sIdx].getSCount();
  }


  /**
     @brief Getter for row delta.

     @param sIdx is the sample index.
   */
  inline IndexT getDelRow(IndexT sIdx) const {
    return sampleNux[sIdx].getDelRow();
  }


  /**
     @brief Getter for the sampled response sum.

     @param sIdx is the sample index.
   */
  inline double getSum(IndexT sIdx) const {
    return sampleNux[sIdx].getYSum();
  }

  
  /**
     @return response category at index passed.
   */
  inline PredictorT getCtg(IndexT sIdx) const {
    return sampleNux[sIdx].getCtg();
  }
};


/**
   @brief Regression-specific methods and members.
*/
struct SampleReg : public SampleObs {

  SampleReg(const class Sampler* sampler,
	    const class Response* respone);


  /**
     @brief Appends regression-style sampling record.

     @delRow is the distance to the previous added node.

     @param val is the sum of sampled responses.

     @param sCount is the number of times sampled.

     @param ctg unused, as response is not categorical.
   */
  inline double addNode(double yVal,
			const class SamplerNux& nux,
                        PredictorT ctg) {
    sampleNux.emplace_back(yVal, nux);
    return sampleNux.back().getYSum();
  }
  

  /**
     @brief Inverts the randomly-sampled vector of rows.

     @param y is the response vector.
  */
  void bagSamples(const class Sampler* sampler,
		  const vector<double>& y,
		  unsigned int tIdx);
};


/**
 @brief Classification-specific sampling.
*/
struct SampleCtg : public SampleObs {
  
  SampleCtg(const class Sampler* sampler,
	    const class Response* response);

  
  /**
     @brief Appends a sample summary record.

     Parameters as described above.

     @return sum of sampled response values.
   */
  inline double addNode(double yVal,
			const class SamplerNux& nux,
			PredictorT ctg) {
    sampleNux.emplace_back(yVal, nux, ctg);
    double ySum = sampleNux.back().getYSum();
    ctgRoot[ctg] += SumCount(ySum, sampleNux.back().getSCount());
    return ySum;
  }
  
  
  /**
     @brief Samples the response, sets in-bag bits.

     @param yCtg is the response vector.

     @param y is the proxy response vector.
  */
  void bagSamples(const class Sampler* sampler,
		  const vector<PredictorT>& yCtg,
		  const vector<double>& y,
		  unsigned int tIdx);
};


#endif
