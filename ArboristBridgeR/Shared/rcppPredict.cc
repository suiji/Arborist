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
#include "leaf.h"

#include <algorithm>
//#include <iostream>


/**
   @brief Utility for computing mean-square error of prediction.
   
   @param yValid is the test vector.

   @param y is the vector of predictions.

   @param rsq outputs the r-squared statistic.

   @return mean-square error, with output parameter.
 */
double MSE(const double yValid[], NumericVector y, double &rsq) {
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
   @brief Predction for regression.

   @return Wrapped zero, with copy-out parameters.
 */
RcppExport SEXP RcppPredictReg(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest, SEXP sBag) {
  int nPredNum, nPredFac, nRow;
  NumericMatrix blockNum;
  IntegerMatrix blockFac;
  PredblockUnwrap(sPredBlock, nRow, nPredNum, nPredFac, blockNum, blockFac);

  std::vector<unsigned int> origin, facOrig, facSplit;
  std::vector<ForestNode> *forestNode;
  ForestUnwrap(sForest, origin, facOrig, facSplit, forestNode);
  
  std::vector<double> yRanked;
  std::vector<unsigned int> leafOrigin;
  std::vector<LeafNode> *leafNode;
  std::vector<BagRow> *bagRow;
  std::vector<unsigned int> rank;
  LeafUnwrapReg(sLeaf, yRanked, leafOrigin, leafNode, bagRow, rank);

  std::vector<double> yPred(nRow);
  std::vector<unsigned int> dummy;
  Predict::Regression(nPredNum > 0 ? transpose(blockNum).begin() : 0, nPredFac > 0 ? transpose(blockFac).begin() : 0, nPredNum, nPredFac, *forestNode, origin, facOrig, facSplit, leafOrigin, *leafNode, *bagRow, rank, yPred, Rf_isNull(sBag) ? dummy : as<std::vector<unsigned int> >(sBag));

  List prediction;
  if (Rf_isNull(sYTest)) { // Prediction
    prediction = List::create(
			 _["yPred"] = yPred,
			 _["qPred"] = NumericMatrix(0)
		     );
    prediction.attr("class") = "PredictReg";
  }
  else { // Validation
    NumericVector yTest(sYTest);
    double rsq;
    double mse = MSE(&yPred[0], yTest, rsq);
    prediction = List::create(
			 _["yPred"] = yPred,
			 _["mse "]= mse,
			 _["rsq"] = rsq,
			 _["qPred"] = NumericMatrix(0)
		     );
    prediction.attr("class") = "ValidReg";
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
   @brief Prediction for classification.

   @return Prediction list.
 */
RcppExport SEXP RcppPredictCtg(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest, SEXP sBag, bool doProb) {
  int nPredNum, nPredFac, nRow;
  NumericMatrix blockNum;
  IntegerMatrix blockFac;
  PredblockUnwrap(sPredBlock, nRow, nPredNum, nPredFac, blockNum, blockFac);
    
  std::vector<unsigned int> origin, facOrig, facSplit;
  std::vector<ForestNode> *forestNode;
  ForestUnwrap(sForest, origin, facOrig, facSplit, forestNode);

  std::vector<unsigned int> leafOrigin;
  std::vector<LeafNode> *leafNode;
  std::vector<BagRow> *bagRow;
  std::vector<double> weight;
  CharacterVector levelsTrain;
  LeafUnwrapCtg(sLeaf, leafOrigin, leafNode, bagRow, weight, levelsTrain);

  unsigned int ctgWidth = levelsTrain.length();
  bool validate = !Rf_isNull(sYTest);
  IntegerVector yTest = validate ? IntegerVector(sYTest) - 1 : IntegerVector(0);
  CharacterVector levelsTest = validate ? as<CharacterVector>(IntegerVector(sYTest).attr("levels")) : CharacterVector(0);
  IntegerVector levelMatch = validate ? match(levelsTest, levelsTrain) : IntegerVector(0);
  unsigned int testWidth;
  unsigned int testLength = yTest.length();
  bool dimFixup = false;
  if (validate) {
    if (is_true(any(levelsTest != levelsTrain))) {
      dimFixup = true;
      IntegerVector sq = seq(0, levelsTest.length() - 1);
      IntegerVector idxNonMatch = sq[is_na(levelMatch)];
      if (idxNonMatch.length() > 0) {
	warning("Unreachable test levels not encountered in training");
	int proxy = ctgWidth + 1;
	for (int i = 0; i < idxNonMatch.length(); i++) {
	  int idx = idxNonMatch[i];
	  levelMatch[idx] = proxy++;
        }
      }

    // Matches are one-based.
      for (unsigned int i = 0; i < testLength; i++) {
        yTest[i] = levelMatch[yTest[i]] - 1;
      }
      testWidth = max(yTest) + 1;
    }
    else {
      testWidth = levelsTest.length();
    }
  }
  else {
    testWidth = 0;
  }
  std::vector<unsigned int> testCore(testLength);
  for (unsigned int i = 0; i < testLength; i++) {
    testCore[i] = yTest[i];
  }

  IntegerVector confCore(testWidth * ctgWidth);
  std::vector<double> misPredCore(testWidth);
  IntegerVector censusCore = IntegerVector(nRow * ctgWidth);
  std::vector<int> yPred(nRow);
  NumericVector probCore = doProb ? NumericVector(nRow * ctgWidth) : NumericVector(0);
  std::vector<unsigned int> dummy;
  Predict::Classification(nPredNum > 0 ? transpose(blockNum).begin() : 0, nPredFac > 0 ? transpose(blockFac).begin() : 0, nPredNum, nPredFac, *forestNode, origin, facOrig, facSplit, leafOrigin, *leafNode, *bagRow, weight, yPred, censusCore.begin(), testCore, validate ? confCore.begin() : 0, misPredCore, doProb ? probCore.begin() : 0, Rf_isNull(sBag) ? dummy : as<std::vector<unsigned int> >(sBag));

  List predBlock(sPredBlock);
  IntegerMatrix census = transpose(IntegerMatrix(ctgWidth, nRow, censusCore.begin()));
  census.attr("dimnames") = List::create(predBlock["rowNames"], levelsTrain);
  NumericMatrix prob = doProb ? transpose(NumericMatrix(ctgWidth, nRow, probCore.begin())) : NumericMatrix(0);
  if (doProb) {
    prob.attr("dimnames") = List::create(predBlock["rowNames"], levelsTrain);
  }

  for (int i = 0; i < nRow; i++) // Bases to unity for front end.
    yPred[i] = yPred[i] + 1;

  List prediction;
  if (validate) {
    IntegerMatrix conf = transpose(IntegerMatrix(ctgWidth, testWidth, confCore.begin()));
    NumericVector misPred(levelsTest.length());
    if (dimFixup) {
      IntegerMatrix confOut(levelsTest.length(), ctgWidth);
      for (int i = 0; i < levelsTest.length(); i++) {
        confOut(i, _) = conf(levelMatch[i] - 1, _);
	misPred[i] = misPredCore[levelMatch[i] - 1];
      }
      conf = confOut;
    }
    else {
      for (int i = 0; i < levelsTest.length(); i++)
	misPred[i] = misPredCore[i];
    }
    misPred.attr("names") = levelsTest;
    conf.attr("dimnames") = List::create(levelsTest, levelsTrain);
    prediction = List::create(
      _["misprediction"] = misPred,
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
  PredblockUnwrap(sPredBlock, nRow, nPredNum, nPredFac, blockNum, blockFac);
    
  std::vector<unsigned int> origin, facOrig, facSplit;
  std::vector<ForestNode> *forestNode;
  ForestUnwrap(sForest, origin, facOrig, facSplit, forestNode);

  std::vector<double> yRanked;
  std::vector<unsigned int> leafOrigin;
  std::vector<LeafNode> *leafNode;
  std::vector<BagRow> *bagRow;
  std::vector<unsigned int> rank;
  LeafUnwrapReg(sLeaf, yRanked, leafOrigin, leafNode, bagRow, rank);

  std::vector<double> yPred(nRow);
  std::vector<double> quantVecCore(as<std::vector<double> >(sQuantVec));
  std::vector<double> qPredCore(nRow * quantVecCore.size());
  std::vector<unsigned int> dummy;
  Predict::Quantiles(nPredNum > 0 ? transpose(blockNum).begin() : 0, nPredFac > 0 ? transpose(blockFac).begin() : 0, nPredNum, nPredFac, *forestNode, origin, facOrig, facSplit, leafOrigin, *leafNode, *bagRow, rank, yRanked, yPred, quantVecCore, as<int>(sQBin), qPredCore,  Rf_isNull(sBag) ? dummy : as<std::vector<unsigned int> >(sBag));

  NumericMatrix qPred(transpose(NumericMatrix(quantVecCore.size(), nRow, qPredCore.begin())));
  List prediction;
  if (!Rf_isNull(sYTest)) {
    double rsq;
    double mse = MSE(&yPred[0], NumericVector(sYTest), rsq);
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
