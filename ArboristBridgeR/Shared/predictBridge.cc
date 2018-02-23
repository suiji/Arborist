// Copyright (C)  2012-2018  Mark Seligman
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
   @file predictBridge.cc

   @brief C++ interface to R entry for prediction methods.

   @author Mark Seligman
 */

#include <Rcpp.h>
using namespace Rcpp;


#include "frameblockBridge.h"
#include "forestBridge.h"
#include "leafBridge.h"
#include "predictBridge.h"

#include <algorithm>


/**
   @brief Utility for computing mean-square error of prediction.
   
   @param yValid is the test vector.

   @param y is the vector of predictions.

   @param rsq outputs the r-squared statistic.

   @return mean squared error, with output parameter.
 */
double PredictBridge::MSE(const double yValid[],
			  NumericVector y,
			  double &rsq,
			  double &mae) {
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
List PredictBridge::Reg(SEXP sPredBlock,
			SEXP sForest,
			SEXP sLeaf,
			SEXP sYTest,
			bool validate) {
  BEGIN_RCPP
  auto framePredict = FrameblockBridge::Unwrap(sPredBlock);
  auto forest = ForestBridge::Unwrap(sForest);
  auto leafReg = LeafRegBridge::Unwrap(List(sLeaf));

  vector<double> yPred(framePredict->NRow());
  Predict::Regression(framePredict, forest, leafReg, yPred);

  delete leafReg;
  delete framePredict;
  delete forest;

  FrameblockBridge::Clear();

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
  
  return prediction;

  END_RCPP
}


RcppExport SEXP ValidateReg(SEXP sPredBlock,
				  SEXP sForest,
				  SEXP sLeaf,
				  SEXP sYTest) {
  return PredictBridge::Reg(sPredBlock, sForest, sLeaf, sYTest, true);
}


RcppExport SEXP TestReg(SEXP sPredBlock,
			      SEXP sForest,
			      SEXP sLeaf,
			      SEXP sYTest) {
  return PredictBridge::Reg(sPredBlock, sForest, sLeaf, sYTest, false);
}


/**
   @brief Prediction for classification.

   @return Prediction list.
 */
List PredictBridge::Ctg(SEXP sPredBlock,
			SEXP sForest,
			SEXP sLeaf,
			SEXP sYTest,
			bool validate,
			bool doProb) {
  auto framePredict = FrameblockBridge::Unwrap(sPredBlock);
  auto leafCtg = LeafCtgBridge::Unwrap(List(sLeaf), validate);
  CharacterVector levelsTrain = leafCtg->Levels();
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
  vector<unsigned int> testCore(testLength);
  for (unsigned int i = 0; i < testLength; i++) {
    testCore[i] = yTest[i];
  }

  unsigned int nRow = framePredict->NRow();

  vector<unsigned int> confCore(testWidth * ctgWidth);
  vector<double> misPredCore(testWidth);
  vector<unsigned int> censusCore(nRow * ctgWidth);
  vector<unsigned int> yPred(nRow);
  auto forest = ForestBridge::Unwrap(sForest);
  NumericVector probCore = doProb ? NumericVector(nRow * ctgWidth) : NumericVector(0);
  Predict::Classification(framePredict, forest, leafCtg, yPred, &censusCore[0], testCore, test ? &confCore[0] : 0, misPredCore, doProb ? probCore.begin() : 0);
  delete leafCtg;
  delete framePredict;
  delete forest;
  
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

  FrameblockBridge::Clear();
  return prediction;
}


RcppExport SEXP ValidateVotes(SEXP sPredBlock,
				    SEXP sForest,
				    SEXP sLeaf,
				    SEXP sYTest) {
  return PredictBridge::Ctg(sPredBlock, sForest, sLeaf, sYTest, true, false);
}


RcppExport SEXP ValidateProb(SEXP sPredBlock,
				   SEXP sForest,
				   SEXP sLeaf,
				   SEXP sYTest) {
  return PredictBridge::Ctg(sPredBlock, sForest, sLeaf, sYTest, true, true);
}


/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sForest contains the trained forest.

   @param sLeaf contains the trained leaves.

   @param sVotes outputs the vote predictions.

   @return Prediction object.
 */
RcppExport SEXP TestVotes(SEXP sPredBlock,
				SEXP sForest,
				SEXP sLeaf,
				SEXP sYTest) {
  return PredictBridge::Ctg(sPredBlock, sForest, sLeaf, sYTest, false, false);
}


/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sForest contains the trained forest.

   @param sLeaf contains the trained leaves.

   @param sVotes outputs the vote predictions.

   @return Prediction object.
 */
RcppExport SEXP TestProb(SEXP sPredBlock,
			       SEXP sForest,
			       SEXP sLeaf,
			       SEXP sYTest) {
  return PredictBridge::Ctg(sPredBlock, sForest, sLeaf, sYTest, false, true);
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
List PredictBridge::Quant(SEXP sPredBlock,
			  SEXP sForest,
			  SEXP sLeaf,
			  SEXP sQuantVec,
			  SEXP sQBin,
			  SEXP sYTest,
			  bool validate) {
  BEGIN_RCPP
  auto framePredict = FrameblockBridge::Unwrap(sPredBlock);
  auto forest = ForestBridge::Unwrap(sForest);

  // Quantile prediction requires full bagging information regardless
  // whether validating.
  auto leafReg = LeafRegBridge::Unwrap(List(sLeaf), true);

  unsigned int nRow = framePredict->NRow();
  vector<double> yPred(nRow);
  vector<double> quantVecCore(as<vector<double> >(sQuantVec));
  vector<double> qPredCore(nRow * quantVecCore.size());
  Predict::Quantiles(framePredict, forest, leafReg, yPred, quantVecCore, as<unsigned int>(sQBin), qPredCore, validate);
  delete leafReg;
  delete framePredict;
  delete forest;
  
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

  FrameblockBridge::Clear();
  return prediction;

  END_RCPP
}


RcppExport SEXP ValidateQuant(SEXP sPredBlock,
				    SEXP sForest,
				    SEXP sLeaf,
				    SEXP sYTest,
				    SEXP sQuantVec,
				    SEXP sQBin) {
  return PredictBridge::Quant(sPredBlock, sForest, sLeaf, sQuantVec, sQBin, sYTest, true);
}


RcppExport SEXP TestQuant(SEXP sPredBlock,
				SEXP sForest,
				SEXP sLeaf,
				SEXP sQuantVec,
				SEXP sQBin,
				SEXP sYTest) {
  return PredictBridge::Quant(sPredBlock, sForest, sLeaf, sQuantVec, sQBin, sYTest, false);
}
