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
  static unique_ptr<SampledObs> sampledObs; ///< Persists iff boosting.

  const IndexT nSamp; ///< Number of observation samples requested.
  const vector<class SamplerNux>& nux; ///< Sampler nodes.
  const IndexT bagCount; ///< # distinct bagged samples.
  const bool boosting; ///< Static property of Booster class.

  double (SampledObs::* adder)(double, const class SamplerNux&, PredictorT);

  double bagSum; ///< Sum of bagged responses.  Updated iff booosting.
  vector<IndexT> obs2Sample; ///< Maps observation index to sample index.
  vector<SumCount> ctgRoot; ///< Root census of categorical response.
  vector<SampleNux> sampleNux; ///< Per-sample summary, with row-delta.

  // Reset at staging:
  vector<vector<IndexT>> sample2Rank; ///< Splitting rank map.
  vector<IndexT> runCount; ///< Staging initialization.

  bool treeZero; ///< Independent trees or first boosting.


  virtual void bagSamples(const class PredictorFrame* frame) = 0;


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

  /**
     @brief Boosts score of next tree.

     @param scoreBase is true iff base score to be assigned.
   */
  void boostPrescore(bool scoreBase);


public:

  static void deInit();
  

  const vector<SampleNux>& getSamples() const {
    return sampleNux;
  }

  
  void setSamples(vector<SampleNux> sampleNux) {
    this->sampleNux = std::move(sampleNux);
  }
  

  /**
     @brief Static entry for categorical response (classification).

     @param response summarizes the training response.

     @return new SampledCtg instance.
   */
  static SampledObs* getCtg(const class Sampler* sampler,
			    const class ResponseCtg* response,
			    unsigned int tIdx);


  /**
     @brief Static entry for continuous response (regression).

     @param y is the training response.

     @return new SampledReg instance.
   */
  static SampledObs* getReg(const class Sampler* sampler,
			    const class ResponseReg* response,
			    unsigned int tIdx);

  
  /**
     @brief Constructor.

     @param frame summarizes predictor ranks by row.
   */
  SampledObs(const class Sampler* sampler,
	     unsigned int tIdx,
	     double (SampledObs::* adder_)(double, const class SamplerNux&, PredictorT) = nullptr);


  virtual ~SampledObs();

  
  void sampleRoot(const class PredictorFrame* frame,
		  class FrontierScorer* scorer);

  
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

  
  void bagSamples(const class PredictorFrame* frame);


  /**
     @brief Inverts the randomly-sampled vector of rows.

     @param y is the response vector.


  */
  void bagSamples(const class PredictorFrame* frame,
		  const vector<double>& y);
};


/**
 @brief Classification-specific sampling.
*/
struct SampledCtg : public SampledObs {
  const class ResponseCtg* response;

  
  SampledCtg(const class Sampler* sampler,
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
  
  
  void bagSamples(const class PredictorFrame* frame);


  /**
     @brief Samples the response, sets in-bag bits.

     @param yCtg is the response vector.

     @param y is the proxy response vector.
  */
  void bagSamples(const class PredictorFrame* frame,
		  const vector<PredictorT>& yCtg,
		  const vector<double>& y);
};


#endif
