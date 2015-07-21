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

   @brief C++ interface to R for prediction.

   @author Mark Seligman
 */

#include <Rcpp.h>

//#include <iostream>
using namespace Rcpp;
using namespace std;

#include "predict.h"

/**
 @brief Out-of-box prediction with quantiles.  Individual predictions are not exposed.  Rather, the OOB/test/importance statistics are averaged and returned.

 @param sPredGini is a copy-out vector of per-predictor Gini gain values.

 @param sError is a copy-out scalar hoding mean-square error.

 @param sQuantVec is a vector of quantile training data.

 @param sQPred is an output vector of quantile predictions.

 @return Wrapped zero, with copy-out vector and scalar paramters.

*/
RcppExport SEXP RcppPredictOOBQuant(SEXP sPredGini, SEXP sError, SEXP sQuantVec, SEXP sQBin, SEXP sQPred) {
  NumericVector error(sError);
  NumericVector quantVec(sQuantVec);
  NumericVector qPred(sQPred);
  NumericVector predGini(sPredGini);

  Predict::PredictOOBQuant(error.begin(), quantVec.begin(), quantVec.length(), as<int>(sQBin), qPred.begin(), predGini.begin());

  return wrap(0);
}

/**
   @brief Out-of-box predction for regression.

   @param sPredGini is a copy-out vector with Gini values.

   @param sError holds a copy-out scalar with mean-square error.

   @return Wrapped zero, with copy-out parameters.
 */
RcppExport SEXP RcppPredictOOB(SEXP sPredGini, SEXP sError) {
  NumericVector error(sError);
  NumericVector predGini(sPredGini);

  Predict::PredictOOBReg(error.begin(), predGini.begin());
  return wrap(0);
}

/**
   @brief Out-of-box prediction for classification.

   @param sPredGini is a copy-out vector reporting Gini values.

   @param sConf is a copy-out confusion matrix

   @param sError is a copy-out scalar with mean-square error.

   @return Wrapped zero, with copy-out parameters.
 */
RcppExport SEXP RcppPredictOOBCtg(SEXP sPredGini, SEXP sConf, SEXP sError, SEXP sCensus) {
  IntegerVector conf(sConf);
  NumericVector error(sError);
  NumericVector predGini(sPredGini);

  if (Rf_isNull(sCensus)) {
    Predict::PredictOOBCtg(conf.begin(), error.begin(), predGini.begin(), 0);
  }
  else {
    IntegerVector census(sCensus);
    Predict::PredictOOBCtg(conf.begin(), error.begin(), predGini.begin(), census.begin());
  }
    
  return wrap(0);
}

/**
   @brief Reloads a previously-generated forest for use by prediction.  Trees are stored sequentially in long vectors.
   
   @param sPreds are the predictors for each terminal of each tree.

   @param sSplits are the split values for each nonterminal.

   @param sBump are the increments to the left-hand offspring index.

   @param sOrigins is a vector of tree origin offsets.

   @param sFacOff are offsets into a vector holding bit encodings of the LHS for factor-valued predictors.

   @param sFacSplits are bit encodings for left-hand subset membership decisions.

   @return Wrapped zero.
 */

RcppExport SEXP RcppReload(SEXP sPreds, SEXP sSplits, SEXP sBump, SEXP sOrigins, SEXP sFacOff, SEXP sFacSplits) {
  IntegerVector preds(sPreds);
  NumericVector splits(sSplits);
  IntegerVector bump(sBump);
  IntegerVector origins(sOrigins);
  IntegerVector facOff(sFacOff);
  IntegerVector facSplits(sFacSplits);

  if (facSplits.length() == 0)
    Predict::ForestReload(origins.length(), preds.length(), preds.begin(), splits.begin(), bump.begin(), origins.begin(), 0, 0);
  else
    Predict::ForestReload(origins.length(), preds.length(), preds.begin(), splits.begin(), bump.begin(), origins.begin(), facOff.begin(), facSplits.begin());

  return wrap(0);
}

/**
   @brief Reloads quantile information from a previously-built forest.

   @param sQYRanked is a vector of ranked response values.

   @param sQRankOrigin is a vector of tree origin offsets.

   @param sQRank is a vector of ranks.

   @param sQRankCount is a vector recording the rank counts.

   @return Wrapped zero.
 */
RcppExport SEXP RcppReloadQuant(SEXP sTree, SEXP sQYRanked, SEXP sQRank, SEXP sQSCount) {
  NumericVector qYRanked(sQYRanked);
  IntegerVector qRank(sQRank);
  IntegerVector qSCount(sQSCount);

  Predict::ForestReloadQuant(as<int>(sTree), qYRanked.begin(), qRank.begin(), qSCount.begin());

  return wrap(0);
}


/**
   @brief Predicts quantiles

   @param sQuantVect is a vector of quantile training information.

   @param sQPred is an output vector containing predicted quantiles.

   @param sY is an output vector containing predicted responses.

   @return Wrapped zero, with output parameters.
 */
RcppExport SEXP RcppPredictQuant(SEXP sQuantVec, SEXP sQBin, SEXP sQPred, SEXP sY) {
  NumericVector quantVec(sQuantVec);
  NumericVector qPred(sQPred);
  NumericVector y(sY);

  Predict::PredictQuant(y.length(), quantVec.begin(), quantVec.length(), as<int>(sQBin), qPred.begin(), y.begin());

  return wrap(0);
}

/**
   @brief Predicts from a regression forest.

   @param sY is an output vector containing the predicted response values.

   @return Wrapped zero, with output parameter vector.
 */
RcppExport SEXP RcppPredictReg(SEXP sY) {
  NumericVector y(sY);

  Predict::PredictReg(y.begin());

  return wrap(0);
}

/**
   @brief Predicts from a classification forest.

   @param sY is an output vector containing the (1-based) predicted response values.

   @param sCtgWidth is the number of categories in the response.

   @return Wrapped zero, with output pameter vector.
 */
RcppExport SEXP RcppPredictCtg(SEXP sY, SEXP sCtgWidth, SEXP sCensus) {
  IntegerVector y(sY);
  int ctgWidth = as<int>(sCtgWidth);
  if (Rf_isNull(sCensus)) {
    Predict::PredictCtg(y.begin(), y.length(), ctgWidth, 0);
  }
  else {
    IntegerVector census(sCensus);
    Predict::PredictCtg(y.begin(), y.length(), ctgWidth, census.begin());
  }
  y = y + 1;

  return wrap(0);
}

