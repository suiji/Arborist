// Copyright (C)  2012-2019   Mark Seligman
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
   @file predictRf.h

   @brief C++ interface to R entry for prediction.

   @author Mark Seligman
 */


#ifndef RF_PREDICT_RF_H
#define RF_PREDICT_RF_H

#include <Rcpp.h>
using namespace Rcpp;


RcppExport SEXP ValidateReg(const SEXP sFrame,
                            const SEXP sTrain,
                            SEXP sYTest,
			    SEXP sPermute,
                            SEXP sNThread);


RcppExport SEXP TestReg(const SEXP sFrame,
                        const SEXP sTrain,
                        SEXP sYTest,
                        SEXP sOOB,
                        SEXP sNThread);


RcppExport SEXP ValidateVotes(const SEXP sFrame,
                              const SEXP sTrain,
                              SEXP sYTest,
			      SEXP sPermute,
                              SEXP sNThread);


RcppExport SEXP ValidateProb(const SEXP sFrame,
                             const SEXP sTrain,
                             SEXP sYTest,
			     SEXP sPermute,
                             SEXP sNThread);


RcppExport SEXP ValidateQuant(const SEXP sFrame,
                              const SEXP sTrain,
                              SEXP sYTest,
			      SEXP sPermute,
                              SEXP sQuantVec,
                              SEXP sNThread);


RcppExport SEXP TestQuant(const SEXP sFrame,
                          const SEXP sTrain,
                          SEXP sQuantVec,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread);

/**
   @brief Predicts with class votes.

   @param sFrame contains the blocked observations.

   @param sTrain contains the trained object.

   @param sYTest is the vector of test values.

   @param sOOB indicates whether testing is out-of-bag.

   @return wrapped predict object.
 */
RcppExport SEXP TestProb(const SEXP sFrame,
                         const SEXP sTrain,
                         SEXP sYTest,
                         SEXP sOOB,
                         SEXP sNThread);


/**
   @brief Predicts with class votes.

   @param sFrame contains the blocked observations.

   @param sTrain contains the trained object.

   @param sYTest contains the test vector.

   @param sOOB indicates whether testing is out-of-bag.

   @return wrapped predict object.
 */
RcppExport SEXP TestVotes(const SEXP sFrame,
                          const SEXP sTrain,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread);

/**
   @brief Bridge-variant PredictBridge pins unwrapped front-end structures.
 */
struct PBRf {

  /**
     @brief Exception-throwing guard ensuring valid encapsulation.

     @return wrapped List representing Core-generated PredictCtg.
   */
  static List checkLeafReg(const List &lTrain);

  /**
     @brief Exception-throwing guard ensuring valid encapsulation.

     @return wrapped List representing Core-generated PredictCtg.
   */
  static List checkLeafCtg(const List &lTrain);


  static List predictCtg(const List& lDeframe,
			 const List& lTrain,
			 SEXP sYTest,
			 bool oob,
			 bool doProb,
			 unsigned int permute,
			 unsigned int nThread);

  
  /**
     @brief Prediction for regression.  Parameters as above.
   */
  static List predictReg(const List& lDeframe,
                         const List& lTrain,
                         SEXP sYTest,
                         bool oob,
			 unsigned int permute,
                         unsigned int nThread);


  /**
  @brief Prediction with quantiles.

    @param sFrame contains the blocked observations.

    @param sTrain contains the trained object.

    @param sQuantVec is a vector of quantile training data.
   
    @param sYTest is the test vector.

    @param oob is true iff testing restricted to out-of-bag.

    @param permute is positive iff permutation testing is specified.

    @return wrapped prediction list.
 */
  static List predictQuant(const List& lDeframe,
			   const List& sTrain,
			   SEXP sQuantVec,
			   SEXP sYTest,
			   bool oob,
			   unsigned int permute,
			   unsigned int nThread);

  /**
     @brief Unwraps regression data structurs and moves to box.

     @return unique pointer to bridge-variant PredictBridge. 
   */
  static unique_ptr<struct PredictRegBridge> unwrapReg(const List& lDeframe,
						    const List& lTrain,
						    SEXP sYTest,
						    bool oob,
						    unsigned int permute,
						    unsigned int nThread,
						    vector<double> quantile = vector<double>(0));

  /**
     @brief Instantiates core prediction object and predicts quantiles.

     @return wrapped predictions.
   */
  List predict(SEXP sYTest,
               const vector<double>& quantile) const;

  /**
     @brief Unwraps regression data structurs and moves to box.

     @return unique pointer to bridge-variant PredictBridge. 
   */
  static unique_ptr<struct PredictCtgBridge> unwrapCtg(const List& lDeframe,
						       const List& lTrain,
						       SEXP sYTest,
						       bool oob,
						       bool doProb,
						       unsigned int permute,
						       unsigned int nThread);


  static List summary(const List& lDeframe,
		      SEXP sYTest,
                      const struct PredictRegBridge* pBridge);


  /**
     @brief Builds a NumericMatrix representation of the quantile predictions.
     
     @param leafBridge is the leaf handle.

     @param pBridge is the prediction handle.

     @return transposed core matrix if quantiles requested, else empty matrix.
  */
  static NumericMatrix getQPred(const struct PredictRegBridge* pBridge);


  /**
     @brief Builds a NumericVector representation of the estimand quantiles.
     
     @param pBridge is the prediction handle.

     @return quantile of predictions if quantiles requesed, else empty vector.
   */
  static NumericVector getQEst(const struct PredictRegBridge* pBridge);


  static List getPrediction(const PredictRegBridge* pBridge);


  /**
     @param varTest is the variance of the test vector.
   */  
  static List getValidation(const PredictRegBridge* pBridge,
			    const NumericVector& yTestFE);
  

  static List getImportance(const class PredictRegBridge* pBridge,
			    const NumericVector& yTestFE,
			    const CharacterVector& predNames);



private:
  /**
     @brief Instantiates core prediction object and predicts means.

     @return wrapped predictions.
   */
  static List predictReg(SEXP sYTest);

  /**
     @brief Instantiates core PredictRf object, driving prediction.

     @return wrapped prediction.
   */
  static List predictCtg(SEXP sYTest, const List& lTrain, const List& sFrame);

  
  static vector<double> regTest(SEXP sYTest);

  
  static vector<double> regTrain(const List& lList);

  
  /**
     @param train is a previously-verified RegLeaf list.

     @return mean of training response.
   */
  static double meanTrain(const List& lLeaf);


  static vector<unsigned int> ctgTest(const List& lLeaf,
				      SEXP sYTest);

  
  /**
     @param train is a previously-verified CtgLeaf list.

     @return cardinaltiy of training response.
   */
  static unsigned int ctgTrain(const List& lLeaf);
};


/**
   @brief Rf specialization of Core PredictReg, q.v.
 */
struct LeafRegRf {

  static List predict(const List &list,
                      SEXP sYTest,
                      class Predict *predict);

  /**
     @brief Builds bridge object from wrapped front-end data.

     @param lLeaf references the leaf object.

     @param lDeframe references the deframed observations.
   */
  static unique_ptr<struct LeafBridge> unwrap(const List& lLeaf,
						    const List& lDeframe);
};


struct LeafPredictRf {
  /**
     @brief Instantiates front-end leaf.

     @param lLeaf references the leaf.

     @param lDeframe references the deframed observations.

     @param doProb indicates whether a probability matrix is requested.
   */
  static unique_ptr<struct LeafBridge> unwrap(const List& lLeaf,
						 const List& lDeframe);
};

  
/**
   @brief Rf specialization of Core PredictCtg, q.v.
 */
struct LeafCtgRf {
  static List predict(const List &list,
		      SEXP sYTest,
		      const List &signature,
		      class Predict *predict,
		      bool doProb);
  /**
     @param sYTest is the one-based test vector, possibly null.

     @param rowNames are the row names of the test data.

     @return list of summary entries.   
  */
  static List summary(const List& lDeframe,
                      const List& lTrain,
                      const struct PredictCtgBridge* pBridge,
                      SEXP sYTest);


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
struct TestCtg {
  const CharacterVector levelsTrain;
  const CharacterVector levels;
  const IntegerVector test2Merged;
  const vector<unsigned int> yTestZero;
  const unsigned int ctgMerged;

  TestCtg(const IntegerVector& yTest,
          const CharacterVector &levelsTrain_);

  
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



  NumericMatrix mispredPermute(const PredictCtgBridge* pBridge,
			       const CharacterVector& predNames) const;



  NumericVector oobErrPermute(const PredictCtgBridge* pBridge,
			      const CharacterVector& predNames) const;
};

#endif
