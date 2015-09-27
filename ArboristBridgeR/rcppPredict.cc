// Copyright (C)  2012-2015   Mark Seligman
//
// This file is part of ArboristBridgeR.
//
// ArboristBridgeR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// ArboristBridgeR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ArboristBridgeR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file rcppPredict.cc

   @brief C++ interface to R entry for prediction methods.

   @author Mark Seligman
 */

#include <R.h>
#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

#include "predict.h"
//#include <iostream>

/**
   @brief Forests a previously-generated classification forest for use by prediction.  Trees are stored sequentially in long vectors.
   
   @param sPreds are the predictors for each terminal of each tree.

   @param sSplits are the split values for each nonterminal.

   @param sBump are the increments to the left-hand offspring index.

   @param sOrigins is a vector of tree origin offsets.

   @param sFacOff are offsets into a vector holding bit encodings of the LHS for factor-valued predictors.

   @param sFacSplits are bit encodings for left-hand subset membership decisions.

   @return Wrapped zero.
 */
RcppExport SEXP RcppForestCtg(SEXP sForest, SEXP sLeaf) {
  List forest(sForest);
  List leaf(sLeaf);

  IntegerVector pred((SEXP) forest["pred"]);
  NumericVector split((SEXP) forest["split"]);
  IntegerVector bump((SEXP) forest["bump"]);
  IntegerVector origin((SEXP) forest["origin"]);
  IntegerVector facOrig((SEXP) forest["facOrig"]);
  IntegerVector facSplit((SEXP) forest["facSplit"]);
  CharacterVector yLevels((SEXP) forest["yLevels"]);
  NumericVector leafWeight((SEXP) leaf["weight"]);
  
  Predict::ForestCtg(origin.length(), pred.length(), pred.begin(), split.begin(), bump.begin(), origin.begin(), facOrig.begin(), (unsigned int *) facSplit.begin(), yLevels.length(), leafWeight.begin());

  return wrap(0);
}


/**
   @brief Forests a previously-generated regression forest for use by prediction.  Trees are stored sequentially in long vectors.
   
   @param sPreds are the predictors for each terminal of each tree.

   @param sSplits are the split values for each nonterminal.

   @param sBump are the increments to the left-hand offspring index.

   @param sOrigins is a vector of tree origin offsets.

   @param sFacOff are offsets into a vector holding bit encodings of the LHS for factor-valued predictors.

   @param sFacSplits are bit encodings for left-hand subset membership decisions.

   @param sYRanked is a vector of ranked response values.

   @param sRank is a vector of ranks.

   @param sRankCount is a vector recording the rank counts.

   @return Wrapped zero.
 */
RcppExport SEXP RcppForestReg(SEXP sForest, SEXP sLeaf) {
  List forest(sForest);
  List leaf(sLeaf);

  IntegerVector pred((SEXP) forest["pred"]);
  NumericVector split((SEXP) forest["split"]);
  IntegerVector bump((SEXP) forest["bump"]);
  IntegerVector origin((SEXP) forest["origin"]);
  IntegerVector facOff((SEXP) forest["facOrig"]);
  IntegerVector facSplit((SEXP) forest["facSplit"]);
  NumericVector yRanked((SEXP) leaf["yRanked"]);
  IntegerVector rank((SEXP) leaf["rank"]);
  IntegerVector sCount((SEXP) leaf["sCount"]);

  Predict::ForestReg(origin.length(), pred.length(), pred.begin(), split.begin(), bump.begin(), origin.begin(), facOff.begin(), (unsigned int *) facSplit.begin(), rank.begin(), sCount.begin(), yRanked.begin());

  return wrap(0);
}


/**
   @brief Out-of-box predction for regression.

   @param sPredInfo is a copy-out vector with Info values.

   @param sError holds a copy-out scalar with mean-square error.

   @return Wrapped zero, with copy-out parameters.
 */
RcppExport SEXP RcppValidateReg(SEXP sY, SEXP sBag) {
  NumericVector y(sY);
  IntegerVector bag(sBag);

  int errCode = Predict::PredictReg(y.begin(), (unsigned int *) bag.begin());
  if (errCode != 0)
    stop("Internal error:  class mismatch");

  return wrap(0);
}


/**
   @brief Predicts from a regression forest.

   @param sY is an output vector containing the predicted response values.

   @return Wrapped zero, with output parameter vector.
 */
RcppExport SEXP RcppPredictReg(SEXP sY) {
  NumericVector y(sY);

  int errCode = Predict::PredictReg(y.begin(), 0);
  if (errCode != 0)
    stop("Internal error:  class mismatch");

  return wrap(0);
}


/**
   @brief Out-of-box prediction for classification.

   @param sPredInfo is a copy-out vector reporting Info values.

   @param sConf is a copy-out confusion matrix

   @param sError is a copy-out scalar with mean-square error.

   @return Wrapped zero, with copy-out parameters.
 */
RcppExport SEXP RcppValidateVotes(SEXP sYValid, SEXP sBag, SEXP sYPred, SEXP sConf, SEXP sError, SEXP sCensus) {
  IntegerVector yValid(sYValid);
  IntegerVector bag(sBag);
  IntegerVector yPred(sYPred);
  IntegerVector conf(sConf);
  NumericVector error(sError);
  IntegerVector census(sCensus);

  IntegerVector y = yValid - 1;
  int errCode = Predict::ValidateCtg(y.begin(), (unsigned int *) bag.begin(), yPred.begin(), census.begin(), conf.begin(), error.begin(), 0);
  if (errCode != 0)
    stop("Internal error:  class mismatch");

  yPred = yPred + 1;

  return wrap(0);
}


/**
   @brief Out-of-box prediction for classification.

   @param sPredInfo is a copy-out vector reporting Info values.

   @param sConf is a copy-out confusion matrix

   @param sError is a copy-out scalar with mean-square error.

   @return Wrapped zero, with copy-out parameters.
 */
RcppExport SEXP RcppValidateProb(SEXP sYValid, SEXP sBag, SEXP sYPred, SEXP sConf, SEXP sError, SEXP sCensus, SEXP sProb) {
  IntegerVector yValid(sYValid);
  IntegerVector bag(sBag);
  IntegerVector yPred(sYPred);
  IntegerVector conf(sConf);
  NumericVector error(sError);
  IntegerVector census(sCensus);
  NumericVector prob(sProb);

  IntegerVector y = yValid - 1;
  int errCode = Predict::ValidateCtg(y.begin(), (unsigned int *) bag.begin(), yPred.begin(), census.begin(), conf.begin(), error.begin(), prob.begin());
  if (errCode != 0)
    stop("Internal error:  class mismatch");

  yPred = yPred + 1;
  return wrap(0);
}


/**
   @brief Predicts with class votes.

   @param sY outputs the (1-based) predicted response values.

   @param sVotes outputs the vote predictions.

   @return Wrapped zero, with output pameter vector.
 */
RcppExport SEXP RcppPredictVotes(SEXP sYPred, SEXP sCensus) {
  IntegerVector yPred(sYPred);
  IntegerVector census(sCensus);

  int errCode = Predict::PredictCtg(yPred.begin(), census.begin(), 0);
  if (errCode != 0)
    stop("Internal error:  class mismatch");

  yPred = yPred + 1;
  return wrap(0);
}


/**
   @brief Prediction with class probabilities.

   @param sY outputs the (1-based) predicted response values.

   @param sProb outputs the leaf-weighted response probablities.

   @return Wrapped zero, with output pameter vector.
 */
RcppExport SEXP RcppPredictProb(SEXP sYPred, SEXP sCensus, SEXP sProb) {
  IntegerVector yPred(sYPred);
  IntegerVector census(sCensus);
  NumericVector prob(sProb);

  int errCode = Predict::PredictCtg(yPred.begin(), census.begin(), prob.begin());
  if (errCode != 0)
    stop("Internal error:  class mismatch");

  yPred = yPred + 1;
  return wrap(0);
}


/**
 @brief Validation with quantiles.

 @param sQuantVec is a vector of quantile training data.

 @param sQBin is the bin parameter.

 @param sQPred is an output vector of quantile predictions.

 @param sY is the predicted vector.

 @param sBag is the training bag set.

 @return Wrapped zero, with copy-out vector paramters.
*/
RcppExport SEXP RcppValidateQuant(SEXP sQuantVec, SEXP sQBin, SEXP sQPred, SEXP sY, SEXP sBag) {
  NumericVector quantVec(sQuantVec);
  NumericVector qPred(sQPred);
  NumericVector y(sY);
  IntegerVector bag(sBag);

  int errCode = Predict::PredictQuant(y.begin(), quantVec.begin(), quantVec.length(), as<int>(sQBin), qPred.begin(), (unsigned int *) bag.begin());

  if (errCode != 0)
    stop("Internal error:  class mismatch");
  
  return wrap(0);
}


/**
 @brief Prediction with quantiles.

 @param sQuantVec is a vector of quantile training data.

 @param sQBin is the bin parameter.

 @param sQPred is an output vector of quantile predictions.

 @param sY is the predicted vector.

 @return Wrapped zero, with copy-out vector paramters.
*/
RcppExport SEXP RcppPredictQuant(SEXP sQuantVec, SEXP sQBin, SEXP sQPred, SEXP sY) {
  NumericVector quantVec(sQuantVec);
  NumericVector qPred(sQPred);
  NumericVector y(sY);

  int errCode = Predict::PredictQuant(y.begin(), quantVec.begin(), quantVec.length(), as<int>(sQBin), qPred.begin(), 0);

  if (errCode != 0)
    stop("Internal error:  class mismatch");
  
  return wrap(0);
}


// Move somewhere appropriate:
//
RcppExport SEXP RcppMSE(SEXP sYValid, SEXP sY) {
  NumericVector yValid(sYValid);
  NumericVector y(sY);

  double SSE = 0.0;
  for (int i = 0; i < y.length(); i++) {
    SSE += (yValid[i] - y[i]) * (yValid[i] - y[i]);
  }

  // TODO:  Repair assumption that every row sampled.
  double MSE = SSE / y.length();

  return wrap(MSE);
}


