// Copyright (C)  2012-2024   Mark Seligman
//
// This file is part of RboristBase.
//
// RboristBase is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RboristBase is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RboristBase.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file samplerR.h

   @brief C++ interface to R entry for sampled obsservations.

   @author Mark Seligman
 */

#ifndef SAMPLER_R_H
#define SAMPLER_R_H

#include <Rcpp.h>
using namespace Rcpp;

#include <vector>
using namespace std;

RcppExport SEXP rootSample(const SEXP sY,
			   const SEXP sRowWeight,
			   const SEXP sNSamp,
			   const SEXP sNTree,
			   const SEXP sWithRepl,
			   const SEXP sNHoldout,
			   const SEXP sUndefined);


/**
   @brief Summary of bagged rows, by tree.

   Recast as namespace?
 */
struct SamplerR {
  static const string strYTrain;
  static const string strNSamp;
  static const string strNTree; // EXIT
  static const string strNRep;
  static const string strSamples; ///< Output field name of sample.
  static const string strHash; ///< Post-sampling hash.

  static List rootSample(const SEXP sY,
			 const SEXP sNSamp,
			 const SEXP sNTree,
			 const SEXP sWithRepl,
			 const vector<double>& weight,
			 const SEXP sNHoldout,
			 const vector<size_t>& undefined);


  /**
     @brief sY is the response vector.
     
     @return number of observations.
   */
  static size_t getNObs(const SEXP& sY);


  static unsigned int getNRep(const List& lSampler);
  

  /**
     @brief As above, but with sampler parameter.
   */
  static size_t countObservations(const List& lSampler);

  
  /**
     @brief Invokes bridge sampler per rep.
   */
  static void sampleRepeatedly(struct SamplerBridge& bridge);


  /**
    @brief Call-back to internal sampling implementation.


    @param nObs is the size of the set to be sampled.

    @param nSamp is the number of samples to draw.

    @param replace is true iff sampling with replacement.

    @param weight is either a buffer of nObs-many weights or empty.

    @return vector of sampled indices with length 'nSamp'.
  */
  static vector<size_t> sampleObs(size_t nSamp,
				  bool replace,
				  NumericVector& weight);


  static IntegerVector sampleReplace(NumericVector& weight,
				     size_t nSamp);


  static IntegerVector sampleNoReplace(NumericVector& weight,
				       size_t nSamp);

  /**
     @brief Bundles trained bag into format suitable for R.

     The wrap functions are called at TrainR::summary, following which
     'this' is deleted.  There is therefore no need to reinitialize
     the block state.

     @return list containing raw data and summary information.
   */
  static List wrap(const struct SamplerBridge& bridge,
		   const SEXP& sY);


  static List wrap(const struct SamplerBridge& bridge,
		   const IntegerVector& yTrain);
  
  
  static List wrap(const struct SamplerBridge& bridge,
		   const NumericVector& yTrain);
  
  
  /**
     @brief Consumes a block of samples following training.
   */
  static NumericVector bridgeConsume(const struct SamplerBridge& bridge);


  /**
     @brief Checks that bag and prediction data set have conforming rows.

     @param lSampler summarizes the sampled response.

     @param lDeframe summarizes the predictors.
   */
  static void checkOOB(const List& lSampler,
		       const List& lDeframe);
  

  /**
     @brief Reads bundled sampler in R form.

     @param lSampler contains the R sampler summary.

     @param return instantiation suitable for training.
   */
  static struct SamplerBridge unwrapTrain(const List& lSampler);


  /**
     @brief Reads bundled bag information in front-end format.

     @param lSampler contains the R sampler summary.

     @param lDeframe contains the deframed observations.

     @param bagging true iff bagging specified.  EXIT.

     @return instantiation suitable for prediction.
   */
  static struct SamplerBridge unwrapPredict(const List& lSampler,
					    const List& lDeframe,
					    bool bagging);


  /**
     @return core-ready vector of zero-based factor codes.
   */
  static vector<unsigned int> coreCtg(const IntegerVector& yTrain);


  /**
     @return minimal SamplerBridge.
   */
  static struct SamplerBridge unwrapGeneric(const List& lSampler);


  static struct SamplerBridge makeBridgeTrain(const List& lSampler,
					      const IntegerVector& yTrain);


  static struct SamplerBridge makeBridgeTrain(const List& lSampler,
				       const NumericVector& yTrain);


  static struct SamplerBridge makeBridgeCtg(const List& lSampler,
					    const List& lDeframe,
					    bool generic = false);


  static struct SamplerBridge makeBridgeNum(const List& lSampler,
					    const List& lDeframe,
					    bool generic = false);
};


/**
   @brief Representation caching export values.
 */
struct SamplerExpand {
  unsigned int nTree;
  size_t nObs;

  SamplerExpand(unsigned int nTree_,
		size_t nObs_) :
    nTree(nTree_),
    nObs(nObs_) {
  }


  /**
     @return wrapped export summary.
   */
  static struct SamplerExpand unwrap(const List& lSampler);
};


#endif
