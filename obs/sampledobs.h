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

struct NodeScorer;

/**
 @brief Run of instances of a given row obtained from sampling for an individual tree.
*/

class SamplerNux;
class PredictorFrame;
class ResponseReg;
class ResponseCtg;

class SampledObs {
 protected:

  static vector<double> obsWeight;
  
  const IndexT nSamp; ///< Number of observation samples requested.
  const vector<SamplerNux>& nux; ///< Sampler nodes.
  const IndexT bagCount; ///< # distinct bagged samples.

  double (SampledObs::* adder)(double, const SamplerNux&, PredictorT);

  double bagSum; ///< Sum of bagged responses.  Updated iff booosting.
  vector<IndexT> obs2Sample; ///< Maps observation index to sample index.
  vector<SumCount> ctgRoot; ///< Root census of categorical response.
  vector<SampleNux> sampleNux; ///< Per-sample summary, with row-delta.

  // Reset at staging:
  vector<vector<IndexT>> sample2Rank; ///< Splitting rank map.
  vector<IndexT> runCount; ///< Staging initialization.


  virtual void sampleObservations(NodeScorer*) = 0;


  /**
     @brief Samples rows and counts resulting occurrences.

     @param y is the proxy / response:  classification / summary.

     @param yCtg is true response / zero:  classification / regression.
  */
  void sampleObservations(const vector<double>& y,
			  const vector<PredictorT>& yCtg);


  /**
     @brief As above, but bypasses slow trivial sampling.
   */
  void bagTrivial(const vector<double>& y,
		  const vector<PredictorT>& yCtg);


  /**
     @return map from sample index to predictor rank.
   */
  vector<IndexT> sampleRanks(const PredictorFrame* layout,
			     PredictorT predIdx);


public:


  static void init(vector<double> obsWeight_);

  
  static void deInit();

  
  vector<SampleNux>& getSamples() {
    return sampleNux;
  }


  const vector<IndexT>& getObs2Sample() const {
    return obs2Sample;
  }


  /**
     @brief Constructor.

     @param frame summarizes predictor ranks by row.
   */
  SampledObs(const class Sampler* sampler,
	     unsigned int tIdx,
	     double (SampledObs::* adder_)(double, const SamplerNux&, PredictorT) = nullptr);


  virtual ~SampledObs();

  
  void sampleRoot(const PredictorFrame* frame,
		  NodeScorer* scorer);

  
  /**
     @brief Getter for root category census vector.
   */
  const vector<SumCount> getCtgRoot() const {
    return ctgRoot;
  }


  auto getNCtg() const {
    return ctgRoot.size();
  }
  
  
  /**
     @brief Getter for user-specified sample count.
  */ 
  IndexT getNSamp() const {
    return nSamp;
  }

  
  /**
     @brief Getter for bag count:  # uniquely-sampled rows.
  */
  IndexT getBagCount() const {
    return bagCount;
  }


  /**
     @brief Getter for sum of bagged responses.
   */
  double  getBagSum() const {
    return bagSum;
  }


  /**
     @brief Determines whether observation is sampled.

     @param[out] sampledIdx is the sample index, iff sampled.
   */
  bool isSampled(IndexT obsIdx,
			IndexT& sampleIdx) const {
    sampleIdx = obs2Sample[obsIdx];
    if (sampleIdx < bagCount) {
      return true;
    }
    else {
      return false;
    }
  }


  /**
     @brief Looks up sample index for sampled row.

     @param[out] sampleIdx is the associated sample index, if sampled.

     @return true iff row is sampled.
   */
  bool isSampled(IndexT obsIdx,
			IndexT& sampleIdx,
			SampleNux*& nux) {
    sampleIdx = obs2Sample[obsIdx];
    if (sampleIdx < bagCount) {
      nux = &sampleNux[sampleIdx];
      return true;
    }
    else {
      return false;
    }
  }


  /**
     @brief As above, no access to members.
   */
  bool isSampled(IndexT obsIdx,
			IndexT& sampleIdx,
			SampleNux& nux) const {
    sampleIdx = obs2Sample[obsIdx];
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
  IndexT getSCount(IndexT sIdx) const {
    return sampleNux[sIdx].getSCount();
  }


  /**
     @brief Getter for row delta.

     @param sIdx is the sample index.
   */
  IndexT getDelRow(IndexT sIdx) const {
    return sampleNux[sIdx].getDelRow();
  }


  /**
     @brief Getter for the sampled response sum.

     @param sIdx is the sample index.
   */
  double getSum(IndexT sIdx) const {
    return sampleNux[sIdx].getYSum();
  }

  
  /**
     @return response category at index passed.
   */
  PredictorT getCtg(IndexT sIdx) const {
    return sampleNux[sIdx].getCtg();
  }


  IndexT getRank(PredictorT predIdx,
			IndexT sIdx) const {
    return sample2Rank[predIdx][sIdx];
  }


  IndexT getRunCount(PredictorT predIdx) const {
    return runCount[predIdx];
  }

  
  void setRanks(const PredictorFrame* layout);
};


/**
   @brief Regression-specific methods and members.
*/
struct SampledReg : public SampledObs {
  const ResponseReg* response;


  SampledReg(const Sampler* sampler,
	     const ResponseReg* response,
	     unsigned int tId);


  ~SampledReg();

  
  /**
     @brief Appends regression-style sampling record.

     @delRow is the distance to the previous added node.

     @param val is the sum of sampled responses.

     @param sCount is the number of times sampled.

     @param ctg unused, as response is not categorical.
   */
  double addNode(double yVal,
			const SamplerNux& nux,
                        PredictorT ctg) {
    sampleNux.emplace_back(yVal, nux);
    return sampleNux.back().getYSum();
  }

  
  void sampleObservations(NodeScorer* scorer);


  /**
     @brief Inverts the randomly-sampled vector of rows.

     @param y is the response vector.


  */
  void sampleObservations(NodeScorer* scorer,
			  const vector<double>& y);
};


/**
 @brief Classification-specific sampling.
*/
struct SampledCtg : public SampledObs {
  const ResponseCtg* response;

  static vector<double> classWeight;


  static void init(vector<double> classWeight_);

  
  SampledCtg(const Sampler* sampler,
	     const ResponseCtg* response_,
	     unsigned int tIdx);


  ~SampledCtg();

  
  /**
     @brief Appends a sample summary record.

     Parameters as described above.

     @return sum of sampled response values.
   */
  double addNode(double yVal,
			const SamplerNux& nux,
			PredictorT ctg) {
    sampleNux.emplace_back(yVal, nux, ctg);
    double ySum = sampleNux.back().getYSum();
    ctgRoot[ctg] += SumCount(ySum, sampleNux.back().getSCount());
    return ySum;
  }
  
  
  void sampleObservations(NodeScorer* scorer);


  /**
     @brief Samples the response, sets in-bag bits.

     @param yCtg is the response vector.

     @param y is the proxy response vector.
  */
  void sampleObservations(NodeScorer* scorer,
			  const vector<PredictorT>& yCtg);
};


#endif
