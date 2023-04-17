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

#include "leafR.h"
#include "forestR.h"
#include "samplerbridge.h"

/**
   @brief Main training entry from front end.
 */
RcppExport SEXP rfTrain(const SEXP sRLEFrame,
			const SEXP sSampler,
			const SEXP sArgList);


struct TrainR {

  // Training granularity.  Values guesstimated to minimize footprint of
  // Core-to-Bridge copies while also not over-allocating:
  static constexpr unsigned int treeChunk = 20;
  static constexpr double allocSlop = 1.2;

  static bool verbose; ///< Whether to report progress while training.

  const SamplerBridge samplerBridge;
  const unsigned int nTree; ///< # trees under training.
  LeafR leaf; ///< Summarizes sample-to-leaf mapping.
  FBTrain forest; ///< Pointer to core forest.
  NumericVector predInfo; ///< Forest-wide sum of predictors' split information.


  TrainR(const List& lSampler,
	 const List& argList);


  void trainChunks(const struct TrainBridge& tb,
		   bool thinLeaves);


  /**
     @brief Scales the per-predictor information quantity by # trees.

     @return remapped vector of scaled information values.
   */
  NumericVector scaleInfo(const TrainBridge& trainBridge) const;

  
  /**
     @return implicit R_NilValue.
   */
  static SEXP initFromArgs(const List &argList,
			   struct TrainBridge& trainBridge);


  /**
     @brief Unsets static initializations.
   */
  static void deInit();


  /**
     @brief Static entry into training.

     @param argList is the user-supplied argument list.

     @return R-style list of trained summaries.
   */
  static List train(const List& lRLEFrame,
		    const List& lSampler,
		    const List& argList);


  /**
     @brief Consumes core representation of a trained tree for writing.

     @unsigned int tIdx is the absolute tree index.

     @param scale guesstimates a reallocation size.
   */
  void consume(const struct ForestBridge& fb,
	       const struct LeafBridge& lb,
               unsigned int tIdx,
               unsigned int chunkSize);


  /**
     @brief As above, but consumes information vector.
   */
  void consumeInfo(const struct TrainedChunk* train);

  
  /**
     @brief Whole-forest summary of trained chunks.

     @param trainBridge contains trained summary from core.

     @return the summary.
   */
  List summarize(const TrainBridge& trainBridge,
		 const vector<string>& diag);// const;

  
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
