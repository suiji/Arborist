// Copyright (C)  2012-2021   Mark Seligman
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
  const unsigned int nTree; // # trees trained.

  static bool trainThin; // Whether to record full sample contents.

  SamplerR(unsigned int nTree_);


  ~SamplerR();

  static void init(bool thin) {
    trainThin = thin;
  };

  
  static void deInit() {
    trainThin = false;
  };

  
  RawVector resizeRaw(const unsigned char raw[],
                      size_t nodeOff,
                      size_t nodeBytes,
                      double scale);

  vector<size_t> samplerBlockHeight; // Accumulated per-tree extent of BagSample vector.
  RawVector samplerBlockRaw; // Packed bag/sample structures as raw data.


  /**
     @brief Getter for tree count.
   */
  const auto getNTree() const {
    return nTree;
  }


  virtual List wrap() = 0;

  /**
     @brief Consumes a chunk of tree bags following training.

     @param train is the trained object.

     @param chunkOff is the offset of the current chunk.
   */
  virtual void consume(const struct TrainChunk* train,
		       unsigned int tIdx,
		       double scale);
  
  /**
     @brief Checks that bag and prediction data set have conforming rows.

     @param lBag is the training bag.
   */
  static SEXP checkOOB(const List& lBag,
                       const size_t nRow);
  
};


struct SamplerRegR : public SamplerR {
  const NumericVector yTrain; // Training response.

  SamplerRegR(const NumericVector& yTrain_, unsigned int nTree_);


  ~SamplerRegR();


  /**
     @brief Bundles trained bag into format suitable for front end.

     @return list containing raw data and summary information.
   */
  List wrap();

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

  static unique_ptr<struct SamplerBridge> unwrap(const List& lTrain);
};


struct SamplerCtgR : public SamplerR {
  const IntegerVector yTrain; // Zero-based training response.
  

  SamplerCtgR(const IntegerVector& yTrain_,
	      unsigned int nTree_);


  ~SamplerCtgR();

  
  /**
     @brief Specialized for writing probability vector. EXIT?
   */
  void consume(const struct TrainChunk* train,
	       unsigned int tIdx,
	       double scale);
  
  /**
     @brief Bundles trained bag into format suitable for front end.

     @return list containing raw data and summary information.
   */
  List wrap();

  
  /**
     @return cardinality of training respone.
   */
  static unsigned int getNCtg(const List& lSampler);
  
  /**
     @brief As above, but categorical.
   */  
  static unique_ptr<struct SamplerBridge> unwrap(const List& lTrain,
						    const List& lDeframe,
						    bool bagging);

  
  static unique_ptr<struct SamplerBridge> unwrap(const List& lTrain);
};

#endif
