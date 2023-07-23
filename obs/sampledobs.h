// This file is part of ArboristBase.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sampledobs.h

   @brief Compact representations of sampled observations.

   @author Mark Seligman
 */

#ifndef OBS_SAMPLEDOBS_H
#define OBS_SAMPLEDOBS_H

#include "sumcount.h"
#include "typeparam.h"
#include "samplenux.h"
#include "obs.h"

#include <vector>


/**
 @brief Run of instances of a given row obtained from sampling for an individual tree.
*/
class SampledObs {
 protected:
  const IndexT nSamp; ///< Number of observation samples requested.
  const vector<class SamplerNux>& nux; ///
  const IndexT bagCount; ///< # distinct sampled observations.
  const double learningRate; ///< Learning rate.

  double (SampledObs::* adder)(double, const class SamplerNux&, PredictorT);

  double bagSum; ///< Sum of bagged responses.
  vector<IndexT> obs2Sample; ///< Maps observation index to sample index.
  vector<SumCount> ctgRoot; ///< Root census of categorical response.
  vector<SampleNux> sampleNux; ///< Per-sample summary, with row-delta.

  // Reset at staging:
  vector<vector<IndexT>> sample2Rank; ///< Splitting rank map.
  vector<IndexT> runCount; ///< Staging initialization.

  // Only used for sequential training:
  unsigned int seqIdx; ///< Accumulated # visits.
  vector<double> preScore; ///< Accumulated estimand.

  
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
     @return map from sample index to predictor rank.
   */
  vector<IndexT> sampleRanks(const class PredictorFrame* layout,
			     PredictorT predIdx);

public:

  /**
     @brief Static entry for categorical response (classification).

     @param response summarizes the training response.

     @return new SampledCtg instance.
   */
  static unique_ptr<struct SampledCtg> factoryCtg(const class Sampler* sampler,
						  const class Train* train,
						  const class ResponseCtg* response,
						  unsigned int tIdx);


  /**
     @brief Static entry for continuous response (regression).

     @param y is the training response.

     @return new SampledReg instance.
   */
  static unique_ptr<struct SampledReg>factoryReg(const class Sampler* sampler,
						 const class Train* train,
						 const class ResponseReg* response,
						 unsigned int tIdx);

  bool sequential() {
    return learningRate > 0.0;
  }

  
  /**
     @brief Constructor.

     @param frame summarizes predictor ranks by row.
   */
  SampledObs(const class Sampler* sampler,
	     const class Train* train,
	     const struct Response* response,
	     unsigned int tIdx,
	     double (SampledObs::* adder_)(double, const class SamplerNux&, PredictorT) = nullptr);


  virtual ~SampledObs();

  
  /**
     @brief Records per-sample scores from trained tree.
   */
  void scoreSamples(const class PreTree* preTree,
		    const struct SampleMap& sampleMap);


  virtual void sampleRoot(const class Frontier* frontier,
			  const class PredictorFrame* frame) = 0;

  
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
  inline bool isSampled(IndexT row,
			IndexT& sampleIdx,
			SampleNux& nux) const {
    sampleIdx = obs2Sample[row];
    if (sampleIdx < bagCount) {
      nux = sampleNux[sampleIdx];
      return true;
    }
    else {
      return false;
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


  inline IndexT getRank(PredictorT predIdx,
			IndexT sIdx) const {
    return sample2Rank[predIdx][sIdx];
  }


  inline IndexT getRunCount(PredictorT predIdx) const {
    return runCount[predIdx];
  }

  
  void setRanks(const class PredictorFrame* layout);
};


/**
   @brief Regression-specific methods and members.
*/
struct SampledReg : public SampledObs {
  const class ResponseReg* response;


  SampledReg(const class Sampler* sampler,
	     const class Train* train,
	     const class ResponseReg* response,
	     unsigned int tId);


  ~SampledReg();

  
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


  void sampleRoot(const class Frontier* frontier,
		  const class PredictorFrame* frame);


  /**
     @brief Inverts the randomly-sampled vector of rows.

     @param y is the response vector.


  */
  void bagSamples(const vector<double>& y,
		  const class PredictorFrame* frame,
		  const class Frontier* frontier);
};


/**
 @brief Classification-specific sampling.
*/
struct SampledCtg : public SampledObs {
  const class ResponseCtg* response;

  
  SampledCtg(const class Sampler* sampler,
	     const class Train* train,
	     const class ResponseCtg* response_,
	     unsigned int tIdx);


  ~SampledCtg();

  
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
  
  
  void sampleRoot(const class Frontier* frontier,
		  const class PredictorFrame* frame);


  /**
     @brief Samples the response, sets in-bag bits.

     @param yCtg is the response vector.

     @param y is the proxy response vector.
  */
  void bagSamples(const vector<PredictorT>& yCtg,
		  const vector<double>& y);
};


#endif
