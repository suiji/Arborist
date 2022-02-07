// Copyright (C)  2012-2022   Mark Seligman
//
// This file is part of rf.
//
// rf is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rf is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file samplerR.h

   @brief C++ interface to R entry for sampled obsservations.

   @author Mark Seligman
 */

#ifndef FOREST_SAMPLER_R_H
#define FOREST_SAMPLER_R_H

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>
using namespace std;

/**
   @brief Summary of bagged rows, by tree.
 */
struct SamplerR {
  static const string strYTrain;
  static const string strNSamp;
  static const string strNTree;
  static const string strSamples; // Output field name of sample.
  
  const unsigned int nSamp; // # samples specified.
  const unsigned int nTree; // # trees trained.

  NumericVector blockNum;
  size_t nuxTop; // First available index in bag buffer.

  SamplerR(unsigned int nSamp_,
	   unsigned int nTree_);


  /**
     @brief Getter for tree count.
   */
  const auto getNTree() const {
    return nTree;
  }

  
  /**
     @brief Bundles trained bag into format suitable for R.

     The wrap functions are called at TrainR::summary, following which
     'this' is deleted.  There is therefore no need to reinitialize
     the block state.

     @return list containing raw data and summary information.
   */
  List wrap(const IntegerVector& yTrain);
  
  
  List wrap(const NumericVector& yTrain);
  
  
  /**
     @brief Consumes a block of samples following training.

     @param scale is a fudge-factor for resizing.
   */
  void bridgeConsume(const struct SamplerBridge* sb,
		     double scale);
  
  /**
     @brief Checks that bag and prediction data set have conforming rows.

     @param lBag is the training bag.
   */
  static SEXP checkOOB(const List& lBag,
                       const size_t nRow);
  

  /**
     @brief Reads bundled bag information in front-end format.

     @param lTrain contains the training summary.

     @param lDeframe contains the deframed observations.

     @param bagging indicates whether a non-null bag is requested.

     @return instantiation containing bag raw data.
   */
  static unique_ptr<struct SamplerBridge> unwrap(const List& lTrain,
						 const List& lDeframe,
						 bool bagging);

  
  static unique_ptr<struct SamplerBridge> unwrap(const List& lSampler,
						 bool bagging = false);


  static unique_ptr<struct SamplerBridge> unwrapNum(const List& lSampler,
						    bool bagging);


  static unique_ptr<struct SamplerBridge> unwrapFac(const List& lSampler,
						    bool bagging);
};

#endif
