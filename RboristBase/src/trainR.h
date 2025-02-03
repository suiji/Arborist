// Copyright (C)  2012-2025   Mark Seligman
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
   @file trainR.h

   @brief C++ interface to R entry for training.

   @author Mark Seligman
 */

#ifndef RBORISTBASE_TRAIN_R_H
#define RBORISTBASE_TRAIN_R_H

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
   @brief Expands trained forest into summary vectors.

   @param sTrain is the trained forest.

   @return expanded forest as list of vectors.
*/
RcppExport SEXP expandTrainRcpp(SEXP sTrain);


struct TrainR {

  // Training granularity.  Values guesstimated to minimize footprint of
  // Core-to-Bridge copies while also not over-allocating:
  static constexpr unsigned int groveSize = 20;
  static constexpr double allocSlop = 1.2;

  static const string strY; 
  static const string strVersion;
  static const string strSignature;
  static const string strSamplerHash;
  static const string strPredInfo;
  static const string strForest;
  static const string strLeaf;
  static const string strDiagnostic;
  static const string strClassName;
  static const string strAutoCompress;
  static const string strEnableCoproc;
  static const string strVerbose;
  static const string strProbVec;
  static const string strPredFixed;
  static const string strSplitQuant;
  static const string strMinNode;
  static const string strNLevel;
  static const string strMinInfo;
  static const string strLoss;
  static const string strForestScore;
  static const string strNodeScore;
  static const string strMaxLeaf;
  static const string strObsWeight;
  static const string strThinLeaves;
  static const string strTreeBlock;
  static const string strNThread;
  static const string strRegMono;
  static const string strClassWeight;

  static bool verbose; ///< Whether to report progress while training.

  const SamplerBridge samplerBridge; ///< handle to core Sampler image.
  const unsigned int nTree; ///< # trees under training.
  LeafR leaf; ///< Summarizes sample-to-leaf mapping.
  FBTrain forest; ///< Pointer to core forest.
  NumericVector predInfo; ///< Forest-wide sum of predictors' split information.
  double nu; ///< Learning rate, passed up from training.
  double baseScore; ///< Base score, " ".

  /**
     @brief Tree count dictated by sampler.
   */
  TrainR(const List& lSampler);


  void trainGrove(const struct TrainBridge& tb);


  static IntegerVector predMap(const List& lTrain);

  
  /**
     @return number of predictors involved in training.
   */
  static unsigned int nPred(const List& lTrain);

  
  /**
     @brief Scales the per-predictor information quantity by # trees.

     @return remapped vector of scaled information values.
   */
  NumericVector scaleInfo(const List& lDeframe) const;


  /**
     @brief Per-invocation initialization of core static values.

     Algorithm-specific implementation included by configuration
     script.
   */
  static void initPerInvocation(const List& lDeframe,
				const List& argList,
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
      @brief Class weighting.

      Class weighting constructs a proxy response based on category
      frequency.  In the absence of class weighting, proxy values are
      identical for all clasess.  The technique of class weighting is
      not without controversy.

      @param classWeight are user-suppled weightings of the categories.

      @return core-ready vector of unnormalized class weights.
   */
  static vector<double> ctgWeight(const IntegerVector& yTrain,
				  const NumericVector& classWeight);


  /**
     @brief Consumes core representation of a trained tree for writing.

     @unsigned int tIdx is the absolute tree index.

     @param scale guesstimates a reallocation size.
   */
  void consume(const struct GroveBridge* grove,
	       const struct LeafBridge& lb,
               unsigned int tIdx,
               unsigned int chunkSize);


  /**
     @brief Whole-forest summary of trained chunks.

     @param trainBridge contains trained summary from core.

     @param lDeframe is the R deframed training data.

     @param lSampler is the R sampler.

     @return the summary.
   */
  List summarize(const TrainBridge& trainBridge,
		 const List& lDeframe,
		 const List& lSampler,
		 const List& argList,
		 const vector<string>& diag);


  /**
     @brief Expands contents as vectors interpretable by the front end.
   */
  static List expand(const List& lTrain);
  
private:
  

  /**
     @brief Estimates scale factor for full-forest reallocation.

     @param treesTot is the total number of trees trained so far.

     @return scale estimation sufficient to accommodate entire forest.
   */
  double safeScale(unsigned int treesTot) const {
    return (treesTot == nTree ? 1 : allocSlop) * double(nTree) / treesTot;
  }
};
#endif
