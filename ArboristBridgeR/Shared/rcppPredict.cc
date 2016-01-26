// Copyright (C)  2012-2016   Mark Seligman
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

#include "rcppPredblock.h"
#include "rcppForest.h"
#include "rcppLeaf.h"
#include "predict.h"

//#include <iostream>

/**
   @brief Utility for computing mean-square error of prediction.
 */
double RcppMSE(NumericVector yValid, NumericVector y, double &rsq) {
  double sse = 0.0;
  for (int i = 0; i < y.length(); i++) {
    double error = yValid[i] - y[i];
    sse += error * error;
  }

  // TODO:  Repair assumption that every row sampled.

  double mse = sse / y.length();
  rsq = 1.0 - (mse * y.length()) / (var(y) * (y.length() - 1.0));
  return mse;
}


/**
   @brief Out-of-bag predction for regression.

   @param sPredInfo is a copy-out vector with Info values.

   @param sError holds a copy-out scalar with mean-square error.

   @return Wrapped zero, with copy-out parameters.
 */
RcppExport SEXP RcppPredictReg(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest, SEXP sBag) {
  int nPredNum, nPredFac, nRow;
  NumericMatrix blockNum;
  IntegerMatrix blockFac;
  RcppPredblockUnwrap(sPredBlock, nRow, nPredNum, nPredFac, blockNum, blockFac);
    
  int *pred, *bump, *origin, *facOrig, nTree, height;
  unsigned int *facSplit;
  double *split;
  RcppForestUnwrap(sForest, pred, split, bump, origin, facOrig, facSplit, nTree, height);

  double *yRanked;
  unsigned int *rank;
  unsigned int *sCount;
  RcppLeafUnwrapReg(sLeaf, yRanked, rank, sCount);
  unsigned int *bag = Rf_isNull(sBag) ? 0 : (unsigned int *) IntegerVector(sBag).begin();
  NumericVector yPred = NumericVector(nRow);
  Predict::Regression(nPredNum > 0 ? transpose(blockNum).begin() : 0, nPredFac > 0 ? transpose(blockFac).begin() : 0, nRow, nPredNum, nPredFac, nTree, height, pred, split, bump, origin, facOrig, facSplit, yPred.begin(), bag);


  List prediction;
  if (Rf_isNull(sYTest)) { // Prediction
    prediction = List::create(
			 _["yPred"] = yPred,
			 _["qPred"] = NumericMatrix(0)
		     );
    prediction.attr("class") = "predReg";
  }
  else { // Validation
    NumericVector yTest(sYTest);
    double rsq;
    double mse = RcppMSE(yPred, yTest, rsq);
    prediction = List::create(
			 _["yPred"] = yPred,
			 _["mse "]= mse,
			 _["rsq"] = rsq,
			 _["qPred"] = NumericMatrix(0)
		     );
    prediction.attr("class") = "validReg";
  }

  return prediction;
}


RcppExport SEXP RcppValidateReg(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest, SEXP sBag) {
  return RcppPredictReg(sPredBlock, sForest, sLeaf, sYTest, sBag);
}


RcppExport SEXP RcppTestReg(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest) {
  return RcppPredictReg(sPredBlock, sForest, sLeaf, sYTest, R_NilValue);
}


/**
   @brief Out-of-bag prediction for classification.

   @param sPredInfo is a copy-out vector reporting Info values.

   @param sConf is a copy-out confusion matrix

   @param sError is a copy-out scalar with mean-square error.

   @return Wrapped zero, with copy-out parameters.
 */
RcppExport SEXP RcppPredictCtg(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest, SEXP sBag, bool doProb) {
  int nPredNum, nPredFac, nRow;
  NumericMatrix blockNum;
  IntegerMatrix blockFac;
  RcppPredblockUnwrap(sPredBlock, nRow, nPredNum, nPredFac, blockNum, blockFac);
    
  int *pred, *bump, *origin, *facOrig, nTree, height;
  unsigned int *facSplit;
  double *split;
  RcppForestUnwrap(sForest, pred, split, bump, origin, facOrig, facSplit, nTree, height);

  double *leafWeight;
  CharacterVector levels;
  RcppLeafUnwrapCtg(sLeaf, leafWeight, levels);

  unsigned int ctgWidth = levels.length();
  unsigned int *bag = Rf_isNull(sBag) ? 0 : (unsigned int *) IntegerVector(sBag).begin();

  IntegerMatrix conf = Rf_isNull(sYTest) ? IntegerMatrix(0) : IntegerMatrix(ctgWidth, ctgWidth);
  NumericVector error = Rf_isNull(sYTest) ? NumericVector(0) : NumericVector(ctgWidth);
  IntegerVector yTest = Rf_isNull(sYTest) ? IntegerVector(0) : IntegerVector(sYTest) - 1;

  IntegerMatrix census = IntegerMatrix(nRow, ctgWidth);
  IntegerVector yPred = IntegerVector(nRow);
  NumericMatrix prob = doProb ? NumericMatrix(nRow, ctgWidth) : NumericMatrix(0);
  Predict::Classification(nPredNum > 0 ? transpose(blockNum).begin() : 0, nPredFac > 0 ? transpose(blockFac).begin() : 0, nRow, nPredNum, nPredFac, nTree, height, pred, split, bump, origin, facOrig, facSplit, ctgWidth, leafWeight, yPred.begin(), census.begin(), Rf_isNull(sYTest) ? 0 : yTest.begin(), conf.begin(), error.begin(), doProb ? prob.begin() : 0, bag);

  List predBlock(sPredBlock);
  census.attr("dimnames") = List::create(predBlock["rowNames"], levels);
  if (doProb) {
    prob.attr("dimnames") = List::create(predBlock["rowNames"], levels);
  }

  List prediction;
  if (!Rf_isNull(sYTest)) {
    conf.attr("dimnames") = List::create(levels, levels);
    prediction = List::create(
         _["misprediction"] = error,
 	 _["confusion"] = conf,
	 _["yPred"] = yPred + 1,
	 _["census"] = census,
	 _["prob"] = prob
    );
    prediction.attr("class") = "ValidCtg";
  }
  else {
    prediction = List::create(
	 _["yPred"] = yPred + 1,
	 _["census"] = census,
	 _["prob"] = prob
   );
   prediction.attr("class") = "PredictCtg";
  }

  return prediction;
}


RcppExport SEXP RcppValidateVotes(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest, SEXP sBag) {
  return RcppPredictCtg(sPredBlock, sForest, sLeaf, sYTest, sBag, false);
}


RcppExport SEXP RcppValidateProb(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest, SEXP sBag) {
  return RcppPredictCtg(sPredBlock, sForest, sLeaf, sYTest, sBag, true);
}


/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sForest contains the trained forest.

   @param sLeaf contains the trained leaves.

   @param sVotes outputs the vote predictions.

   @return Prediction object.
 */
RcppExport SEXP RcppTestVotes(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest) {
  return RcppPredictCtg(sPredBlock, sForest, sLeaf, sYTest, R_NilValue, false);
}


/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sForest contains the trained forest.

   @param sLeaf contains the trained leaves.

   @param sVotes outputs the vote predictions.

   @return Prediction object.
 */
RcppExport SEXP RcppTestProb(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest) {
  return RcppPredictCtg(sPredBlock, sForest, sLeaf, sYTest, R_NilValue, true);
}


/**
   @brief Validation with quantiles.

   @param sPredBlock contains the blocked observations.

   @param sForest contains the trained forest.

   @param sLeaf contains the trained leaves.

   @param sVotes outputs the vote predictions.

   @param sQuantVec is a vector of quantile training data.
   
   @param sQBin is the bin parameter.

   @param sY is the predicted vector.

   @param sBag is the training bag set.

   @return Wrapped zero, with copy-out vector paramters.
*/
RcppExport SEXP RcppPredictQuant(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sQuantVec, SEXP sQBin, SEXP sYTest, SEXP sBag) {
  int nPredNum, nPredFac, nRow;
  NumericMatrix blockNum;
  IntegerMatrix blockFac;
  RcppPredblockUnwrap(sPredBlock, nRow, nPredNum, nPredFac, blockNum, blockFac);
    
  int *pred, *bump, *origin, *facOrig, nTree, height;
  unsigned int *facSplit;
  double *split;
  RcppForestUnwrap(sForest, pred, split, bump, origin, facOrig, facSplit, nTree, height);

  double *yRanked;
  unsigned int *rank;
  unsigned int *sCount;
  RcppLeafUnwrapReg(sLeaf, yRanked, rank, sCount);

  NumericVector yPred(nRow);
  NumericVector quantVec(sQuantVec);
  unsigned int *bag = Rf_isNull(sBag) ? 0 : (unsigned int *) IntegerVector(sBag).begin();
  NumericMatrix qPred = NumericMatrix(nRow, quantVec.length());
  Predict::Quantiles(nPredNum > 0 ? transpose(blockNum).begin() : 0, nPredFac > 0 ? transpose(blockFac).begin() : 0, nRow, nPredNum, nPredFac, nTree, height, pred, split, bump, origin, facOrig, facSplit, rank, sCount, yRanked, yPred.begin(), quantVec.begin(), quantVec.length(), as<int>(sQBin), qPred.begin(), bag);

  List prediction;
  if (!Rf_isNull(sYTest)) {
    double rsq;
    double mse = RcppMSE(yPred, NumericVector(sYTest), rsq);
    prediction = List::create(
 	 _["yPred"] = yPred,
	 _["qPred"] = qPred,
	 _["mse"] = mse,
	 _["rsq"] = rsq
	  );
    prediction.attr("class") = "ValidReg";
  }
  else {
    prediction = List::create(
		 _["yPred"] = yPred,
		 _["qPred"] = qPred
	     );
    prediction.attr("class") = "PredictReg";
  }

  return prediction;
}


RcppExport SEXP RcppValidateQuant(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sQuantVec, SEXP sQBin, SEXP sYTest, SEXP sBag) {
  return RcppPredictQuant(sPredBlock, sForest, sLeaf, sQuantVec, sQBin, sYTest, sBag);
}


RcppExport SEXP RcppTestQuant(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sQuantVec, SEXP sQBin, SEXP sYTest) {
  return RcppPredictQuant(sPredBlock, sForest, sLeaf, sQuantVec, sQBin, sYTest, R_NilValue);
}
