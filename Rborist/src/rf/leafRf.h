// Copyright (C)  2012-2020  Mark Seligman
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
   @file leafRf.h

   @brief C++ class definitions for managing Leaf object.

   @author Mark Seligman

 */

#ifndef RF_LEAF_RF_H
#define RF_LEAF_RF_H

#include <Rcpp.h>
using namespace Rcpp;

#include <vector>
#include <memory>
using namespace std;

/**
   @brief Rf specialization of Core LeafReg, q.v.
 */
struct LeafRegRf {
  /**
     @brief Validates contents of front-end leaf object.

     Exception thrown if contents invalid.

     @return wrapped List representing core-generated leaf.
   */  
  static List checkLeaf(const List &lTrain);

  static List predict(const List &list,
                      SEXP sYTest,
                      class Predict *predict);

  /**
     @brief Builds bridge object from wrapped front-end data.

     @param lLeaf references the leaf object.

     @param lDeframe references the deframed observations.
   */
  static unique_ptr<struct LeafRegBridge> unwrap(const List& lLeaf,
						 const List& lDeframe);

  
  static List summary(SEXP sYTest,
                      const struct PredictBridge* pBridge);


  /**
     @brief Builds a NumericMatrix representation of the quantile predictions.
     
     @param leafBridge is the leaf handle.

     @param pBridge is the prediction handle.

     @return transposed core matrix if quantiles requested, else empty matrix.
  */
  static NumericMatrix getQPred(const struct LeafRegBridge* leafBridge,
                                const struct PredictBridge* pBridge);


  /**
     @brief Builds a NumericVector representation of the estimand quantiles.
     
     @param pBridge is the prediction handle.

     @return quantile of predictions if quantiles requesed, else empty vector.
   */
  static NumericVector getQEst(const struct PredictBridge* pBridge);

  
  /**
     @brief Utility for computing mean-square error of prediction.

     Error is estimated using the prediction and test vectors.  This is somewhat
     different from the approach of the "randomForest" package, which estimates
     a per-tree mean of mean-square oob errors.
   
     @param yPred is the prediction.

     @param yTest is the observed response.

     @param rsq[out] is the r-squared statistic.

     @param mae[out] is the mean absolute error.

     @return mean squared error.
  */
  static double mse(const vector<double>& yPred,
		    const NumericVector& yTest,
		    double& rsq,
		    double& mae);

  
  /**
     @brief Computes predictor importances by permutation.

     Importance is given as the diffence between the permuted and test
     MSE values, computed as above.

     @param yTest is the test calibration.

     @return vector of mse values under permutation, by predictor.
   */
  static NumericVector msePermute(const LeafRegBridge* leaf,
				  const NumericVector& yTest);
};


/**
   @brief Rf specialization of Core LeafCtg, q.v.
 */
struct LeafCtgRf {
  /**
     @brief Exception-throwing guard ensuring valid encapsulation.

     @return wrapped List representing Core-generated LeafCtg.
   */
  static List checkLeaf(const List &lTrain);

  static List predict(const List &list,
                  SEXP sYTest,
                  const List &signature,
                  class Predict *predict,
                  bool doProb);

  /**
     @brief Instantiates front-end leaf.

     @param lLeaf references the leaf.

     @param lDeframe references the deframed observations.

     @param doProb indicates whether a probability matrix is requested.
   */
  static unique_ptr<struct LeafCtgBridge> unwrap(const List& lLeaf,
						 const List& lDeframe,
						 bool doProb);

  
  /**
     @param sYTest is the one-based test vector, possibly null.

     @param rowNames are the row names of the test data.

     @return list of summary entries.   
  */
  static List summary(const List& lDeframe,
                      const List& lTrain,
                      const struct PredictBridge* pBridge,
                      SEXP sYTest);


  /**
     @brief Produces census summary, which is common to all categorical
     prediction.

     @param rowNames is the user-supplied specification of row names.

     @return matrix of predicted categorical responses, by row.
  */
  static IntegerMatrix getCensus(const struct LeafCtgBridge* leaf,
                                 const CharacterVector& levelsTrain,
                                 const CharacterVector& rowNames);

  
  /**
     @param rowNames is the user-supplied collection of row names.

     @return probability matrix if requested, otherwise empty matrix.
  */
  static NumericMatrix getProb(const struct LeafCtgBridge* leaf,
                               const CharacterVector& levelsTrain,
                               const CharacterVector &rowNames);
};


/**
   @brief Internal back end-style vectors cache annotations for
   per-tree access.
 */
class TestCtg {
  const CharacterVector levelsTrain;
  const IntegerVector yTestOne;
  const CharacterVector levels;
  const unsigned int nCtg;
  const IntegerVector test2Merged;
  const IntegerVector yTestZero;
  const unsigned int ctgMerged;

 public:
  TestCtg(SEXP sYTest,
          const CharacterVector &levelsTrain_);

  
  /**
     @brief Determines summary array dimensions by reconciling cardinalities
     of training and test reponses.

     @return reconciled test vector.
  */
  static IntegerVector Reconcile(const IntegerVector &test2Train,
                                 const IntegerVector &yTestOne);
  

  /**
     @brief Reconciles factor encodings of training and test responses.
   */
  static IntegerVector mergeLevels(const CharacterVector &levelsTest,
                                   const CharacterVector &levelsTrain);


  /**
     @brief Fills in misprediction vector.

     @param leaf summarizes the trained leaf frame.
  */
  NumericVector misprediction(const struct LeafCtgBridge* leaf,
			      const vector<unsigned int>& yPred) const;
  

  NumericMatrix mispredPermute(const LeafCtgBridge* leaf) const;


  NumericVector oobErrPermute(const LeafCtgBridge* leaf) const;


  vector<unsigned int> buildConfusion(const LeafCtgBridge* leaf,
				      const vector<unsigned int>& yPred) const;
  
  
/**
   @brief Produces summary information specific to testing:  mispredction
   vector and confusion matrix.

   @param confusion is the internal confusion matrix.

   @param levelsTrain are the levels encountered during training.

   @return output confusion matrix.
 */
  IntegerMatrix getConfusion(const LeafCtgBridge* leaf,
			     const CharacterVector& levelsTrain);


  /**
     @brief Estimates the out-of-bag error.

     @param yPred is the zero-based prediction vector derived by the core.

     @return mean number of mispredictions.
  */
  double oobError(const vector<unsigned int>& yPred) const;
};


/**
   @brief Maintains R-style vectors represting the crescent leaf component
   of the forest during training.
 */
struct LBTrain {
private:
  static bool thin; // User option:  whether to annotate bag state.

  /**
     @brief Consumes core Node recrods and writes as raw data.

     @param leaf is the core representation of a trained leaf.

     @param tIdx is the absolute tree index.

     @param scale estimates a resizing factor.
   */
  void writeNode(const struct TrainChunk* train,
                 unsigned int tIdx,
                 double scale);

  
  /**
   */
  RawVector rawResize(const unsigned char raw[],
                      size_t nodeOff,
                      size_t nodeBytes,
                      double scale);

  /**
     @brief Consumes the BagSample records and writes as raw data.
   */
  void writeBagSample(const struct TrainChunk* train,
                    unsigned int treeOff,
                    double scale);
public:
  IntegerVector nodeHeight;  // Accumulated per-tree extent of Leaf vector.
  RawVector nodeRaw; // Packed node structures as raw data.

  IntegerVector bagHeight; // Accumulated per-tree extent of BagSample vector.
  RawVector blRaw; // Packed bag/sample structures as raw data.

  /**
     @brief Constructor.

     @param nTree is the number of trees over which to train.
   */
  LBTrain(unsigned int nTree);

  virtual ~LBTrain() {}
  
  /**
     @brief Static initialization.

     @param thin_ indicates whether certain annotations may be omitted.
   */
  static void init(bool thin_);

  /**
     @brief Resets static initializations.
   */
  static void deInit();


  /**
     @brief High-level entry for writing contents of a tree's leaves.

     @param leaf is the core represenation of the trained leaves.

     @param tIdx is the absolute index of the tree.

     @param scale estimates a resizing factor.
   */
  virtual void consume(const struct TrainChunk* train,
               unsigned int treeOff,
               double scale);

  /**
     @brief Packages contents for storage by front end.

     @return named list of summary fields.
   */
  virtual List wrap() = 0;
};


struct LBTrainReg : public LBTrain {
  const NumericVector yTrain; // Training response.

  LBTrainReg(const NumericVector& yTrain_,
             unsigned int nTree);

  ~LBTrainReg() {}

  /**
     @brief Description and parameters as with virutal declaration.
   */
  void consume(const struct TrainChunk* train,
               unsigned int treeOff,
               double scale);
  /**
     @brief Description as with virtual declaration.s
   */
  List wrap();
};


/**
   @brief Specialization for categorical leaves, which maintain an
   additional field for weights.
 */
struct LBTrainCtg : public LBTrain {
  NumericVector weight; // Per-category probabilities.
  R_xlen_t weightSize; // Running Size of weight vector.  Not saved.
  const IntegerVector yTrain; // Training response.

  LBTrainCtg(const IntegerVector& yTrain_,
             unsigned int nTree);

  ~LBTrainCtg() {}

  /**
     @brief Description and parameters as with virtual declaration.
   */
  void consume(const struct TrainChunk* train,
               unsigned int treeOff,
               double scale);

  /**
     @brief Description as with virtual declaration.
   */
  List wrap();


    /**
   */
  NumericVector numericResize(const double num[],
                          size_t nodeOff,
                          size_t elts,
                          double scale);

private:
  /**
     @brief Writes leaf weights from core representation.

     Not jagged, so tree index parameter unneeded.

     @param leaf is the core representation of a tree's leaves.

     @double scale estimates a resizing factor.
   */
  void writeWeight(const struct TrainChunk* train,
                   double scale);

};
#endif
