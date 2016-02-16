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

#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

#include "rcppPredblock.h"
#include "rcppForest.h"
#include "rcppLeaf.h"
#include "predict.h"
#include "forest.h"

//#include <iostream>

/**
   @brief Utility for computing mean-square error of prediction.
 */
double RcppMSE(const double yValid[], NumericVector y, double &rsq) {
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

  std::vector<unsigned int> origin, facOrig, facSplit;
  std::vector<ForestNode> *forestNode;
  RcppForestUnwrap(sForest, origin, facOrig, facSplit, forestNode);
  
  std::vector<double> yRanked;
  std::vector<unsigned int> rank, sCount;
  RcppLeafUnwrapReg(sLeaf, yRanked, rank, sCount);

  std::vector<double> yPred(nRow);
  std::vector<unsigned int> dummy;
  Predict::Regression(nPredNum > 0 ? transpose(blockNum).begin() : 0, nPredFac > 0 ? transpose(blockFac).begin() : 0, nPredNum, nPredFac, *forestNode, origin, facOrig, facSplit, yPred, Rf_isNull(sBag) ? dummy : as<std::vector<unsigned int> >(sBag));

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
    double mse = RcppMSE(&yPred[0], yTest, rsq);
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
    
  std::vector<unsigned int> origin, facOrig, facSplit;
  std::vector<ForestNode> *forestNode;
  RcppForestUnwrap(sForest, origin, facOrig, facSplit, forestNode);

  double *leafWeight;
  CharacterVector levels;
  RcppLeafUnwrapCtg(sLeaf, leafWeight, levels);

  unsigned int ctgWidth = levels.length();

  bool validate = !Rf_isNull(sYTest);
  IntegerVector confCore = validate ? IntegerVector(ctgWidth * ctgWidth) : IntegerVector(0);
  NumericVector error = validate ? NumericVector(ctgWidth) : NumericVector(0);
  IntegerVector yTest = validate ? IntegerVector(sYTest) - 1 : IntegerVector(0);

  IntegerVector censusCore = IntegerVector(nRow * ctgWidth);
  std::vector<int> yPred(nRow);
  NumericVector probCore = doProb ? NumericVector(nRow * ctgWidth) : NumericVector(0);
  std::vector<unsigned int> dummy;
  Predict::Classification(nPredNum > 0 ? transpose(blockNum).begin() : 0, nPredFac > 0 ? transpose(blockFac).begin() : 0, nPredNum, nPredFac, *forestNode, origin, facOrig, facSplit, ctgWidth, leafWeight, yPred, censusCore.begin(), validate ? yTest.begin() : 0, validate ? confCore.begin() : 0, validate ? error.begin() : 0, doProb ? probCore.begin() : 0, Rf_isNull(sBag) ? dummy : as<std::vector<unsigned int> >(sBag));

  List predBlock(sPredBlock);
  IntegerMatrix census = transpose(IntegerMatrix(ctgWidth, nRow, censusCore.begin()));
  census.attr("dimnames") = List::create(predBlock["rowNames"], levels);
  NumericMatrix prob = doProb ? transpose(NumericMatrix(ctgWidth, nRow, probCore.begin())) : NumericMatrix(0);
  if (doProb) {
    prob.attr("dimnames") = List::create(predBlock["rowNames"], levels);
  }

  for (int i = 0; i < nRow; i++) // Bases to unity for front end.
    yPred[i] = yPred[i] + 1;

  List prediction;
  if (validate) {
    IntegerMatrix conf = transpose(IntegerMatrix(ctgWidth, ctgWidth, confCore.begin()));
    conf.attr("dimnames") = List::create(levels, levels);
    prediction = List::create(
         _["misprediction"] = error,
 	 _["confusion"] = conf,
	 _["yPred"] = yPred,
	 _["census"] = census,
	 _["prob"] = prob
    );
    prediction.attr("class") = "ValidCtg";
  }
  else {
    prediction = List::create(
      _["yPred"] = yPred,
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
    
  std::vector<unsigned int> origin, facOrig, facSplit;
  std::vector<ForestNode> *forestNode;
  RcppForestUnwrap(sForest, origin, facOrig, facSplit, forestNode);

  std::vector<double> yRanked;
  std::vector<unsigned int> rank, sCount;
  RcppLeafUnwrapReg(sLeaf, yRanked, rank, sCount);

  std::vector<double> yPred(nRow);
  std::vector<double> quantVecCore(as<std::vector<double> >(sQuantVec));
  std::vector<double> qPredCore(nRow * quantVecCore.size());
  std::vector<unsigned int> dummy;
  Predict::Quantiles(nPredNum > 0 ? transpose(blockNum).begin() : 0, nPredFac > 0 ? transpose(blockFac).begin() : 0, nPredNum, nPredFac, *forestNode, origin, facOrig, facSplit, rank, sCount, yRanked, yPred, quantVecCore, as<int>(sQBin), qPredCore,  Rf_isNull(sBag) ? dummy : as<std::vector<unsigned int> >(sBag));

  NumericMatrix qPred(transpose(NumericMatrix(quantVecCore.size(), nRow, qPredCore.begin())));
  List prediction;
  if (!Rf_isNull(sYTest)) {
    double rsq;
    double mse = RcppMSE(&yPred[0], NumericVector(sYTest), rsq);
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
