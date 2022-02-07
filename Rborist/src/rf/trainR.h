// Copyright (C)  2012-2022   Mark Seligman
//
// This file is part of rf
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
   @file trainR.h

   @brief C++ interface to R entry for training.

   @author Mark Seligman
 */

#ifndef RF_TRAIN_R_H
#define RF_TRAIN_R_H

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>
#include <string>
#include <vector>
using namespace std;

RcppExport SEXP TrainRF(const SEXP sRLEFrame,
			const SEXP sArgList);


struct TrainRf {

  // Training granularity.  Values guesstimated to minimize footprint of
  // Core-to-Bridge copies while also not over-allocating:
  static constexpr unsigned int treeChunk = 20;
  static constexpr double allocSlop = 1.2;

  static bool verbose; // Whether to report progress while training.

  const unsigned int nSamp; // # samples per tree.
  const unsigned int nTree; // # trees under training.
  unique_ptr<class SamplerR> sampler; // Summarizes row bagging, by tree.
  unique_ptr<class LeafR> leaf; // Summarizes sample-to-leaf mapping.
  unique_ptr<struct FBTrain> forest; // Pointer to core forest.
  NumericVector predInfo; // Forest-wide sum of predictors' split information.


  /**
     @brief Cconstructor.

     @param nSamp_ is the number of samples per tree.

     @param nTree_ is the number of trees in the block.
   */
  TrainRf(unsigned int nSamp_,
	  unsigned int nTree_);


  /**
     @brief Trains classification forest.

     @param summaryFrame summarizes the predictor frame.

     @return R-style list of trained summaries.
  */
  static List classification(const List& argList,
			     const struct TrainBridge* trainBridge,
			     vector<string>& diag);

  
  /**
     @brief Trains regression forest.

     @param summaryFrame summarizes the predictor frame.

     @return R-style list of trained summaries.
  */
  static List regression(const List& argList,
			 const struct TrainBridge* trainBridge,
			 vector<string>& diag);

  
  /**
      @brief Class weighting.

      Class weighting constructs a proxy response based on category
      frequency.  In the absence of class weighting, proxy values are
      identical for all clasess.  The technique of class weighting is
      not without controversy.

      @param y is the (zero-based) categorical response vector.

      @param classWeight are user-suppled weightings of the categories.

      @return vector of scaled class weights.
  */
  static NumericVector ctgWeight(const IntegerVector &y,
				 const NumericVector &classWeight);



  /**
     @brief Scales the per-predictor information quantity by # trees.

     @return remapped vector of scaled information values.
   */
  NumericVector scaleInfo(const TrainBridge* trainBridge);

  
  /**
     @return implicit R_NilValue.
   */
  static SEXP initFromArgs(const List &argList,
			   struct TrainBridge* trainBridge);


  /**
     @brief Unsets static initializations.

     @param trainBridge is a persistent training handle.

     @return implicit R_NilValue.
   */
  static SEXP deInit(struct TrainBridge* trainBridge);
  

  /**
     @brief Pins frame vectors locally and passes through to TrainRf method.

     @param argList is the front-end argument list.

     @return list of trained forest objects.
   */
  static List train(const List& lRLEFrame,
		    const List& argList);


  /**
     @brief Static entry into training.

     @param argList is the user-supplied argument list.

     @return R-style list of trained summaries.
   */
  static List train(const List& argList,
                    const struct RLEFrame* rleFrame);

  
  /**
     @brief Consumes core representation of a trained tree for writing.

     @unsigned int tIdx is the absolute tree index.

     @param scale guesstimates a reallocation size.
   */
  void consume(const struct ForestBridge& fb,
	       const struct SamplerBridge* sb,
	       const struct LeafBridge* lb,
               unsigned int tIdx,
               unsigned int chunkSize) const;


  /**
     @brief As above, but consumes information vector.
   */
  void consumeInfo(const struct TrainedChunk* train);

  
  /**
     @brief Whole-forest summary of trained chunks.

     @param trainBridge contains trained summary from core.

     @return the summary.
   */
  List summarize(const TrainBridge* trainBridge,
		 const IntegerVector& yTrain,
		 const vector<string>& diag);

  
  List summarize(const TrainBridge* trainBridge,
		 const NumericVector& yTrain,
		 const vector<string>& diag);

  
private:
  

  /**
     @brief Estimates scale factor for full-forest reallocation.

     @param treesTot is the total number of trees trained so far.

     @return scale estimation sufficient to accommodate entire forest.
   */
  inline double safeScale(unsigned int treesTot) const {
    return (treesTot == nTree ? 1 : allocSlop) * double(nTree) / treesTot;
  }
};
#endif
