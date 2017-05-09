// Copyright (C)  2012-2017  Mark Seligman
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

using namespace Rcpp;

#include "rcppPredblock.h"
#include "rcppForest.h"
#include "rcppLeaf.h"
#include "predict.h"

#include "forest.h"
#include "leaf.h"

#include <algorithm>

//#include <iostream>
//using namespace std;

/**
   @brief Utility for computing mean-square error of prediction.
   
   @param yValid is the test vector.

   @param y is the vector of predictions.

   @param rsq outputs the r-squared statistic.

   @return mean squared error, with output parameter.
 */
double MSE(const double yValid[], NumericVector y, double &rsq, double &mae) {
  double sse = 0.0;
  mae = 0.0;
  for (R_len_t i = 0; i < y.length(); i++) {
    double error = yValid[i] - y[i];
    sse += error * error;
    mae += abs(error);
  }
  rsq = 1.0 - sse / (var(y) * (y.length() - 1.0));
  mae /= y.length();

  return sse / y.length();
}


/**
   @brief Predction for regression.

   @return Wrapped zero, with copy-out parameters.
 */
RcppExport SEXP RcppPredictReg(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest, bool validate) {
  unsigned int nPredNum, nPredFac, nRow;
  NumericMatrix blockNum;
  IntegerMatrix blockFac;
  std::vector<double> valNum;
  std::vector<unsigned int> rowStart;
  std::vector<unsigned int> runLength;
  std::vector<unsigned int> predStart;
  RcppPredblock::Unwrap(sPredBlock, nRow, nPredNum, nPredFac, blockNum, blockFac, valNum, rowStart, runLength, predStart);

  unsigned int *origin, *facOrig, *facSplit;
  ForestNode *forestNode;
  unsigned int nTree, nFac, nodeEnd;
  size_t facLen;
  RcppForest::Unwrap(sForest, origin, nTree, facSplit, facLen, facOrig, nFac, forestNode, nodeEnd);
  
  std::vector<double> yTrain;
  std::vector<unsigned int> leafOrigin;
  LeafNode *leafNode;
  unsigned int leafCount;
  BagLeaf *bagLeaf;
  unsigned int bagLeafTot;
  unsigned int *bagBits;
  RcppLeaf::UnwrapReg(sLeaf, yTrain, leafOrigin, leafNode, leafCount, bagLeaf, bagLeafTot, bagBits, validate);

  std::vector<double> yPred(nRow);
  Predict::Regression(valNum, rowStart, runLength, predStart, (valNum.size() == 0 && nPredNum > 0) ? transpose(blockNum).begin() : 0, nPredFac > 0 ? (unsigned int *) transpose(blockFac).begin() : 0, nPredNum, nPredFac, forestNode, origin, nTree, facSplit, facLen, facOrig, nFac, leafOrigin, leafNode, leafCount, bagBits, yTrain, yPred);

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
    double rsq, mae;
    double mse = MSE(&yPred[0], yTest, rsq, mae);
    prediction = List::create(
			 _["yPred"] = yPred,
			 _["mse"] = mse,
			 _["mae"] = mae,
			 _["rsq"] = rsq,
			 _["qPred"] = NumericMatrix(0)
		     );
    prediction.attr("class") = "ValidReg";
  }
  RcppLeaf::Clear();
  RcppForest::Clear();

  return prediction;
}


RcppExport SEXP RcppValidateReg(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest) {
  return RcppPredictReg(sPredBlock, sForest, sLeaf, sYTest, true);
}


RcppExport SEXP RcppTestReg(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest) {
  return RcppPredictReg(sPredBlock, sForest, sLeaf, sYTest, false);
}


/**
   @brief Prediction for classification.

   @return Prediction list.
 */
RcppExport SEXP RcppPredictCtg(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest, bool validate, bool doProb) {
  unsigned int nPredNum, nPredFac, nRow;
  NumericMatrix blockNum;
  IntegerMatrix blockFac;
  std::vector<double> valNum;
  std::vector<unsigned int> rowStart;
  std::vector<unsigned int> runLength;
  std::vector<unsigned int> predStart;
  RcppPredblock::Unwrap(sPredBlock, nRow, nPredNum, nPredFac, blockNum, blockFac, valNum, rowStart, runLength, predStart);
    
  unsigned int *origin, *facOrig, *facSplit;
  ForestNode *forestNode;
  unsigned int nTree, nFac, nodeEnd;
  size_t facLen;
  RcppForest::Unwrap(sForest, origin, nTree, facSplit, facLen, facOrig, nFac, forestNode, nodeEnd);

  std::vector<unsigned int> leafOrigin;
  LeafNode *leafNode;
  unsigned int leafCount;
  BagLeaf *bagLeaf;
  unsigned int bagLeafTot;
  unsigned int *bagBits;
  double *weight;
  unsigned int rowTrain;
  CharacterVector levelsTrain;
  RcppLeaf::UnwrapCtg(sLeaf, leafOrigin, leafNode, leafCount, bagLeaf, bagLeafTot, bagBits, weight, rowTrain, levelsTrain, validate);

  unsigned int ctgWidth = levelsTrain.length();
  bool test = !Rf_isNull(sYTest);
  IntegerVector yTest = test ? IntegerVector(sYTest) - 1 : IntegerVector(0);
  CharacterVector levelsTest = test ? as<CharacterVector>(IntegerVector(sYTest).attr("levels")) : CharacterVector(0);
  IntegerVector levelMatch = test ? match(levelsTest, levelsTrain) : IntegerVector(0);
  unsigned int testWidth;
  unsigned int testLength = yTest.length();
  bool dimFixup = false;
  if (test) {
    if (is_true(any(levelsTest != levelsTrain))) {
      dimFixup = true;
      IntegerVector sq = seq(0, levelsTest.length() - 1);
      IntegerVector idxNonMatch = sq[is_na(levelMatch)];
      if (idxNonMatch.length() > 0) {
	warning("Unreachable test levels not encountered in training");
	int proxy = ctgWidth + 1;
	for (R_len_t i = 0; i < idxNonMatch.length(); i++) {
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

  std::vector<unsigned int> confCore(testWidth * ctgWidth);
  std::vector<double> misPredCore(testWidth);
  std::vector<unsigned int> censusCore(nRow * ctgWidth);
  std::vector<unsigned int> yPred(nRow);
  NumericVector probCore = doProb ? NumericVector(nRow * ctgWidth) : NumericVector(0);
  Predict::Classification(valNum, rowStart, runLength, predStart, (valNum.size() == 0 && nPredNum > 0) ? transpose(blockNum).begin() : 0, nPredFac > 0 ? (unsigned int*) transpose(blockFac).begin() : 0, nPredNum, nPredFac, forestNode, origin, nTree, facSplit, facLen, facOrig, nFac, leafOrigin, leafNode, leafCount, bagBits, rowTrain, weight, ctgWidth, yPred, &censusCore[0], testCore, test ? &confCore[0] : 0, misPredCore, doProb ? probCore.begin() : 0);

  List predBlock(sPredBlock);
  IntegerMatrix census = transpose(IntegerMatrix(ctgWidth, nRow, &censusCore[0]));
  census.attr("dimnames") = List::create(predBlock["rowNames"], levelsTrain);
  NumericMatrix prob = doProb ? transpose(NumericMatrix(ctgWidth, nRow, probCore.begin())) : NumericMatrix(0);
  if (doProb) {
    prob.attr("dimnames") = List::create(predBlock["rowNames"], levelsTrain);
  }

  // OOB error is mean(prediction != training class)
  unsigned int missed = 0;
  for (unsigned int i = 0; i < nRow; i++) { // Bases to unity for front end.
    if (test) {
      missed += (unsigned int) yTest[i] != yPred[i];
    }
    yPred[i] = yPred[i] + 1;
  }
  double oobError = double(missed) / nRow;


  List prediction;
  if (test) {
    IntegerMatrix conf = transpose(IntegerMatrix(ctgWidth, testWidth, &confCore[0]));
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
      for (R_len_t i = 0; i < levelsTest.length(); i++) {
	misPred[i] = misPredCore[i];
      }
    }

    misPred.attr("names") = levelsTest;
    conf.attr("dimnames") = List::create(levelsTest, levelsTrain);
    prediction = List::create(
      _["misprediction"] = misPred,
      _["oobError"] = oobError,
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

  RcppLeaf::Clear();
  RcppForest::Clear();
  return prediction;
}


RcppExport SEXP RcppValidateVotes(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest) {
  return RcppPredictCtg(sPredBlock, sForest, sLeaf, sYTest, true, false);
}


RcppExport SEXP RcppValidateProb(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest) {
  return RcppPredictCtg(sPredBlock, sForest, sLeaf, sYTest, true, true);
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
  return RcppPredictCtg(sPredBlock, sForest, sLeaf, sYTest, false, false);
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
  return RcppPredictCtg(sPredBlock, sForest, sLeaf, sYTest, false, true);
}


/**
   @brief Prediction with quantiles.

   @param sPredBlock contains the blocked observations.

   @param sForest contains the trained forest.

   @param sLeaf contains the trained leaves.

   @param sVotes outputs the vote predictions.

   @param sQuantVec is a vector of quantile training data.
   
   @param sQBin is the bin parameter.

   @param sYTest is the test vector.

   @param bag is true iff validating.

   @return Prediction list.
*/
RcppExport SEXP RcppPredictQuant(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sQuantVec, SEXP sQBin, SEXP sYTest, bool validate) {
  unsigned int nPredNum, nPredFac, nRow;
  NumericMatrix blockNum;
  IntegerMatrix blockFac;
  std::vector<double> valNum;
  std::vector<unsigned int> rowStart;
  std::vector<unsigned int> runLength;
  std::vector<unsigned int> predStart;
  RcppPredblock::Unwrap(sPredBlock, nRow, nPredNum, nPredFac, blockNum, blockFac, valNum, rowStart, runLength, predStart);
    
  unsigned int *origin, *facOrig, *facSplit;
  ForestNode *forestNode;
  unsigned int nTree, nFac, nodeEnd;
  size_t facLen;
  RcppForest::Unwrap(sForest, origin, nTree, facSplit, facLen, facOrig, nFac, forestNode, nodeEnd);

  std::vector<double> yTrain;
  std::vector<unsigned int> leafOrigin;
  LeafNode *leafNode;
  unsigned int leafCount;
  BagLeaf *bagLeaf;
  unsigned int bagLeafTot;
  unsigned int *bagBits;

  // Quantile prediction requires full bagging information regardless
  // whether validating.
  RcppLeaf::UnwrapReg(sLeaf, yTrain, leafOrigin, leafNode, leafCount, bagLeaf, bagLeafTot, bagBits, true);

  std::vector<double> yPred(nRow);
  std::vector<double> quantVecCore(as<std::vector<double> >(sQuantVec));
  std::vector<double> qPredCore(nRow * quantVecCore.size());
  Predict::Quantiles(valNum, rowStart, runLength, predStart, (valNum.size() == 0 && nPredNum > 0) ? transpose(blockNum).begin() : 0, nPredFac > 0 ? (unsigned int*) transpose(blockFac).begin() : 0, nPredNum, nPredFac, forestNode, origin, nTree, facSplit, facLen, facOrig, nFac, leafOrigin, leafNode, leafCount, bagLeaf, bagLeafTot, bagBits, yTrain, yPred, quantVecCore, as<unsigned int>(sQBin), qPredCore, validate);
  
  NumericMatrix qPred(transpose(NumericMatrix(quantVecCore.size(), nRow, qPredCore.begin())));
  List prediction;
  if (!Rf_isNull(sYTest)) {
    double rsq, mae;
    double mse = MSE(&yPred[0], NumericVector(sYTest), rsq, mae);
    prediction = List::create(
 	 _["yPred"] = yPred,
	 _["qPred"] = qPred,
	 _["mse"] = mse,
	 _["mae"] = mae,
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

  RcppLeaf::Clear();
  RcppForest::Clear();
  return prediction;
}


RcppExport SEXP RcppValidateQuant(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sYTest, SEXP sQuantVec, SEXP sQBin) {
  return RcppPredictQuant(sPredBlock, sForest, sLeaf, sQuantVec, sQBin, sYTest, true);
}


RcppExport SEXP RcppTestQuant(SEXP sPredBlock, SEXP sForest, SEXP sLeaf, SEXP sQuantVec, SEXP sQBin, SEXP sYTest) {
  return RcppPredictQuant(sPredBlock, sForest, sLeaf, sQuantVec, sQBin, sYTest, false);
}
