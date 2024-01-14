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
   @file predictR.h

   @brief C++ interface to R entry for prediction.

   @author Mark Seligman
 */


#ifndef PREDICT_R_H
#define PREDICT_R_H

#include <Rcpp.h>
using namespace Rcpp;


#include<memory>
using namespace std;

struct SamplerBridge;
class Predict;

/**
   @brief Prediction with separate test vector.

   @param sFrame contains the blocked observations.

   @param sTrain contains the trained object.

   @param sYTest is the vector of test values, possbily NULL.

   @param sOOB indicates whether testing is out-of-bag.

   @return wrapped predict object.
 */
RcppExport SEXP predictRcpp(const SEXP sFrame,
		 const SEXP sTrain,
		 const SEXP sSampler,
		 const SEXP sYTest,
		 const SEXP sArgs);


/**
   @brief Prediction with training response as test vector.

   Paramaters as with predictRcpp.
 */
RcppExport SEXP validateRcpp(const SEXP sFrame,
		  const SEXP sTrain,
		  const SEXP sSampler,
		  const SEXP sArgs);


/**
   @bridge Prediction through unwrapped PredictBridge object.
 */
struct PredictR {
  static const string strQuantVec;
  static const string strImpPermute;
  static const string strIndexing;
  static const string strBagging;
  static const string strTrapUnobserved;
  static const string strNThread;
  static const string strCtgProb;


  /**
     @brief Drives prediction according to response type.
   */
  static List predict(const List& lDeframe,
		      const List& lTrain,
		      const List& lSampler,
		      const List& lArgs,
		      const SEXP sYTest);


  /**
     @brief Instantiates core classification object and summarizes.

     @return wrapped prediction.
   */
  static List predictCtg(const List& lDeframe,
			 const List& lSampler,
			 const SamplerBridge& samplerBridge,
			 struct ForestBridge& forestBridge,
			 const SEXP sYTest);


  /**
     @brief Instantiates core regression object and summarizes.

     @return wrapped prediction.
   */
  static List predictReg(const List& lDeframe,
			 const SamplerBridge& samplerBridge,
			 struct ForestBridge& forestBridge,
			 const SEXP sYTest);


  /**
     @brief Per-invocation initialization of core static values.

     Algorithm-specific implementation included by configuration
     script.

     @retun implicit R_NilValue.
   */
  static void initPerInvocation(const List& lArgs);


  /**
     @brief Instantiates core prediction object and predicts quantiles.

     @return wrapped predictions.
   */
  List predict(SEXP sYTest,
               const vector<double>& quantile) const;


  static List summary(const List& lDeframe,
		      SEXP sYTest,
                      const struct PredictRegBridge* pBridge);


  /**
     @brief Builds a NumericMatrix representation of the quantile predictions.
     
     @param pBridge is the prediction handle.

     @return transposed core matrix if quantiles requested, else empty matrix.
  */
  static NumericMatrix getQPred(const struct PredictRegBridge* pBridge);


  static List getPrediction(const PredictRegBridge* pBridge);


  static NumericMatrix getIndices(const struct PredictRegBridge* pBridge);

  
  /**
     @param varTest is the variance of the test vector.
   */  
  static List getValidation(const PredictRegBridge* pBridge,
			    const NumericVector& yTestFE);
  

  static List getImportance(const struct PredictRegBridge* pBridge,
			    const NumericVector& yTestFE,
			    const CharacterVector& predNames);

private:
  /**
     @brief Instantiates core prediction object and predicts means.

     @return wrapped predictions.
   */
  static List predictReg(SEXP sYTest);


  /**
     @return regression test vector suitable for core.
   */
  static vector<double> regTest(const SEXP sYTest);


  /**
     @return quantile vector suitable for core.
   */
  static vector<double> quantVec(const List& lArgs);
  

  static vector<unsigned int> ctgTest(const List& lSampler,
				      const SEXP sYTest);
};


/**
   @brief Rf specialization of Core PredictReg, q.v.
 */
struct LeafRegRf {

  static List predict(const List &list,
                      SEXP sYTest,
                      Predict *predict);
};

  
/**
   @brief Rf specialization of Core PredictCtg, q.v.
 */
struct LeafCtgRf {
  static List predict(const List &list,
		      SEXP sYTest,
		      const List &signature,
		      Predict *predict,
		      bool doProb);
  /**
     @param sYTest is the one-based test vector, possibly null.

     @param rowNames are the row names of the test data.

     @return list of summary entries.   
  */
  static List summary(const List& lDeframe,
		      const List& lSampler,
                      const struct PredictCtgBridge* pBridge,
                      SEXP sYTest);


  static NumericMatrix getIndices(const struct PredictCtgBridge* pBridge);

  
  /**
     @brief Produces census summary, which is common to all categorical
     prediction.

     @param rowNames is the user-supplied specification of row names.

     @return matrix of predicted categorical responses, by row.
  */
  static IntegerMatrix getCensus(const PredictCtgBridge* pBridge,
                                 const CharacterVector& levelsTrain,
                                 const CharacterVector& rowNames);

  
  /**
     @param rowNames is the user-supplied collection of row names.

     @return probability matrix if requested, otherwise empty matrix.
  */
  static NumericMatrix getProb(const PredictCtgBridge* pBridge,
                               const CharacterVector& levelsTrain,
                               const CharacterVector &rowNames);

  
  static List getPrediction(const PredictCtgBridge* pBridge,
			    const CharacterVector& levelsTrain,
			    const CharacterVector& ctgNames);
};


/**
   @brief Internal back end-style vectors cache annotations for
   per-tree access.
 */
struct TestCtgR {
  const CharacterVector levelsTrain;
  const CharacterVector levels;
  const IntegerVector test2Merged;
  const vector<unsigned int> yTestZero;
  const unsigned int ctgMerged;

  TestCtgR(const IntegerVector& yTest,
          const CharacterVector& levelsTrain_);

  
  /**
     @brief Determines summary array dimensions by reconciling cardinalities
     of training and test reponses.

     @return reconciled test vector.
  */
  static vector<unsigned int> reconcile(const IntegerVector& test2Train,
					const IntegerVector& yTestOne);
  

  /**
     @brief Reconciles factor encodings of training and test responses.
   */
  IntegerVector mergeLevels(const CharacterVector& levelsTest);


  List getValidation(const PredictCtgBridge* pBridge);


  List getImportance(const PredictCtgBridge* pBridge,
		     const CharacterVector& predNames);

  
  /**
     @brief Fills in misprediction vector.

     @param pBridge is the bridge handle.
  */
  NumericVector getMisprediction(const struct PredictCtgBridge* pBridge) const;
  

/**
   @brief Produces summary information specific to testing:  mispredction
   vector and confusion matrix.

   @param pBridge is the bridge handle.

   @param levelsTrain are the levels encountered during training.

   @return numeric matrix to accommodate wide count values.
 */
  NumericMatrix getConfusion(const PredictCtgBridge* pBridge,
			     const CharacterVector& levelsTrain) const;



  List mispredPermuted(const PredictCtgBridge* pBridge,
		       const CharacterVector& predNames) const;



  NumericMatrix oobErrPermuted(const PredictCtgBridge* pBridge,
			       const CharacterVector& predNames) const;
};

#endif
