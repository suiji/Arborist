// Copyright (C)  2012-2024  Mark Seligman
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
   @file predictR.cc

   @brief C++ interface to R entry for prediction methods.

   @author Mark Seligman
 */

#include "predictbridge.h"
#include "forestbridge.h"
#include "predictR.h"
#include "samplerR.h"
#include "leafR.h"
#include "forestR.h"
#include "trainR.h"
#include "samplerbridge.h"
#include "signatureR.h"

#include <memory>
#include <algorithm>


const string PredictR::strQuantVec = "quantVec";
const string PredictR::strImpPermute = "impPermute";
const string PredictR::strIndexing = "indexing";
const string PredictR::strBagging = "bagging";
const string PredictR::strTrapUnobserved = "trapUnobserved";
const string PredictR::strNThread = "nThread";
const string PredictR::strCtgProb = "ctgProb";

// [[Rcpp::export]]
RcppExport SEXP predictRcpp(const SEXP sDeframe,
		 const SEXP sTrain,
		 const SEXP sSampler,
		 const SEXP sYTest,
		 const SEXP sArgs) {
  return PredictR::predict(List(sDeframe), List(sTrain), List(sSampler), List(sArgs), sYTest);
}


// [[Rcpp::export]]
RcppExport SEXP validateRcpp(const SEXP sDeframe,
		  const SEXP sTrain,
		  const SEXP sSampler,
		  const SEXP sArgs) {
  List lSampler(sSampler);
  SEXP sYTest(lSampler["yTrain"]);  // Testing against the training response.
  return PredictR::predict(List(sDeframe), List(sTrain), lSampler, List(sArgs), sYTest);
}


List PredictR::predict(const List& lDeframe,
		       const List& lTrain,
		       const List& lSampler,
		       const List& lArgs,
		       const SEXP sYTest) {
  bool verbose = as<bool>(lArgs["verbose"]);
  if (verbose)
    Rcout << "Entering prediction" << endl;

  initPerInvocation(lArgs);
  ForestBridge::init(as<IntegerVector>(lTrain[TrainR::strPredMap]).length());

  List prediction;
  SamplerBridge samplerBridge(SamplerR::unwrapPredict(lSampler, lDeframe, as<bool>(lArgs[PredictR::strBagging])));
  ForestBridge forestBridge(ForestR::unwrap(lTrain, samplerBridge));
  if (Rf_isFactor((SEXP) lSampler[SamplerR::strYTrain]))
    prediction = predictCtg(lDeframe, lSampler, samplerBridge, forestBridge, sYTest);
  else
    prediction = predictReg(lDeframe, samplerBridge, forestBridge, sYTest);

  ForestBridge::deInit();

  if (verbose)
    Rcout << "Prediction completed" << endl;

  return prediction;
}


// [[Rcpp::export]]
List PredictR::predictReg(const List& lDeframe,
			  const SamplerBridge& samplerBridge,
			  ForestBridge& forestBridge,
			  const SEXP sYTest) {
  unique_ptr<PredictRegBridge> pBridge(samplerBridge.predictReg(forestBridge, regTest(sYTest)));
  return summary(lDeframe, sYTest, pBridge.get());
}


// [[Rcpp::export]]
vector<double> PredictR::regTest(const SEXP sYTest) {
  vector<double> yTest;
  if (!Rf_isNull(sYTest)) {
    NumericVector yTestFE(as<NumericVector>(sYTest));
    yTest = as<vector<double>>(yTestFE);
  }
  return yTest;
}


// [[Rcpp::export]]
List PredictR::predictCtg(const List& lDeframe,
			  const List& lSampler,
			  const SamplerBridge& samplerBridge,
			  ForestBridge& forestBridge,
			  const SEXP sYTest) {
  unique_ptr<PredictCtgBridge> pBridge(samplerBridge.predictCtg(forestBridge, ctgTest(lSampler, sYTest)));

  return LeafCtgRf::summary(lDeframe, lSampler, pBridge.get(), sYTest);
}


vector<unsigned int> PredictR::ctgTest(const List& lSampler, const SEXP sYTest) {
  if (!Rf_isNull(sYTest)) { // Makes zero-based copy.
    IntegerVector yTrain(as<IntegerVector>(lSampler[SamplerR::strYTrain]));
    TestCtgR testCtg(sYTest, as<CharacterVector>(yTrain.attr("levels")));
    return testCtg.yTestZero;
  }
  else {
    return vector<unsigned int>(0);
  }  
}


vector<double> PredictR::quantVec(const List& lArgs) {
  vector<double> quantile;
  if (!Rf_isNull(lArgs[strQuantVec])) {
    NumericVector quantVec(as<NumericVector>(lArgs[strQuantVec]));
    quantile = vector<double>(quantVec.begin(), quantVec.end());
  }
  return quantile;
}


// [[Rcpp::export]]
List PredictR::summary(const List& lDeframe, SEXP sYTest, const PredictRegBridge* pBridge) {
  List summaryReg;
  if (Rf_isNull(sYTest)) {
    summaryReg = List::create(
			      _["prediction"] = getPrediction(pBridge)
			      );
  }
  else if (!pBridge->permutes()) { // Validation, no importance.
    summaryReg = List::create(
			      _["prediction"] = getPrediction(pBridge),
			      _["validation"] = getValidation(pBridge, NumericVector((SEXP)sYTest))
			      );
  }
  else { // Validation + importance
    summaryReg = List::create(
			      _["prediction"] = getPrediction(pBridge),
			      _["validation"] = getValidation(pBridge, NumericVector((SEXP)sYTest)),
			      _["importance"] = getImportance(pBridge, NumericVector((SEXP) sYTest), SignatureR::unwrapColNames(lDeframe))
			      );
  }
  summaryReg.attr("class") = "SummaryReg";

  return summaryReg;
}


TestCtgR::TestCtgR(const IntegerVector& yTestOne,
                 const CharacterVector& levelsTrain_) :
  levelsTrain(levelsTrain_),
  levels(CharacterVector(as<CharacterVector>(yTestOne.attr("levels")))),
  test2Merged(mergeLevels(levels)),
  yTestZero(reconcile(test2Merged, yTestOne)),
  ctgMerged(*max_element(yTestZero.begin(), yTestZero.end()) + 1) {
}


// [[Rcpp::export]]
List PredictR::getPrediction(const PredictRegBridge* pBridge) {
  List prediction = List::create(
				 _["yPred"] = pBridge->getYPred(),
				 _["qPred"] = getQPred(pBridge),
				 _["qEst"] = pBridge->getQEst(),
				 _["indices"] = getIndices(pBridge)
				 );
  prediction.attr("class") = "PredictReg";
  return prediction;
}


// [[Rcpp::export]]
NumericMatrix PredictR::getIndices(const PredictRegBridge* pBridge) {
  auto indices = pBridge->getIndices();
  size_t nObs = pBridge->getNObs();
  return indices.empty() ? NumericMatrix(0) : NumericMatrix(nObs, indices.size() / nObs, indices.begin());
}


// [[Rcpp::export]]
NumericMatrix PredictR::getQPred(const PredictRegBridge* pBridge) {
  size_t nObs = pBridge->getNObs();
  vector<double> qPred = pBridge->getQPred();
  return qPred.empty() ? NumericMatrix(0) : transpose(NumericMatrix(qPred.size() / nObs, nObs, qPred.begin()));
}


// [[Rcpp::export]]
List PredictR::getValidation(const PredictRegBridge* pBridge,
			 const NumericVector& yTestFE) {
  double sse = pBridge->getSSE();
  size_t nRow = yTestFE.length();
  List validation = List::create(_["mse"] = sse / nRow,
				 _["rsq"] = nRow == 1 ? 0.0 : 1.0 - sse / (var(yTestFE) * (nRow - 1)),
				 _["mae"] = pBridge->getSAE() / nRow
				 );
  validation.attr("class") = "ValidReg";
  return validation;
}


// [[Rcpp::export]]
List PredictR::getImportance(const PredictRegBridge* pBridge,
				   const NumericVector& yTestFE,
				   const CharacterVector& predNames) {
  vector<vector<double>> ssePerm = pBridge->getSSEPermuted();
  unsigned int nPerm = ssePerm[0].size();
  unsigned int nPred = ssePerm.size();

  NumericMatrix mseOut(nPerm, nPred);
  for (unsigned int predIdx = 0; predIdx != nPred; predIdx++) {
    NumericVector ssePred = NumericVector(ssePerm[predIdx].begin(), ssePerm[predIdx].end());
    mseOut.column(predIdx) = NumericVector(ssePred / yTestFE.length());
  }
  mseOut.attr("names") = predNames;

  vector<vector<double>> saePerm = pBridge->getSAEPermuted();
  NumericMatrix maeOut(nPerm, nPred);
  for (unsigned int predIdx = 0; predIdx != nPred; predIdx++) {
    NumericVector saePred = NumericVector(saePerm[predIdx].begin(), saePerm[predIdx].end());
    maeOut.column(predIdx) = NumericVector(saePred / yTestFE.length());
  }
  maeOut.attr("names") = predNames;

  List importance = List::create(_["mse"] = mseOut,
				 _["mae"] = maeOut);
  importance("class") = "ImportanceReg";
  return importance;
}


// [[Rcpp::export]]
IntegerVector TestCtgR::mergeLevels(const CharacterVector& levelsTest) {
  IntegerVector test2Merged(match(levelsTest, levelsTrain));
  IntegerVector sq = seq(0, test2Merged.length() - 1);
  IntegerVector idxNA = sq[is_na(test2Merged)];
  if (idxNA.length() > 0) {
    warning("Uninferable test levels not encountered in training");
    int proxy = levelsTrain.length() + 1;
    for (R_len_t i = 0; i < idxNA.length(); i++) {
      int idx = idxNA[i];
      test2Merged[idx] = proxy++;
    }
  }
  return test2Merged - 1;
}


vector<unsigned int> TestCtgR::reconcile(const IntegerVector& test2Merged,
					const IntegerVector& yTestOne) {
  IntegerVector yZero = yTestOne - 1;
  vector<unsigned int> yZeroOut(yZero.length());
  for (R_len_t i = 0; i < yZero.length(); i++) {
    yZeroOut[i] = test2Merged[yZero[i]];
  }
  return yZeroOut;
}


// [[Rcpp::export]]
List LeafCtgRf::summary(const List& lDeframe, const List& lSampler, const PredictCtgBridge* pBridge, SEXP sYTest) {
  IntegerVector yTrain(as<IntegerVector>(lSampler["yTrain"]));
  CharacterVector levelsTrain(as<CharacterVector>(yTrain.attr("levels")));
  CharacterVector ctgNames(SignatureR::unwrapRowNames(lDeframe));

  List summaryCtg;
  if (Rf_isNull(sYTest)) {
    summaryCtg = List::create(
			      _["prediction"] = getPrediction(pBridge, levelsTrain, ctgNames)
			      );
  }
  else {
    TestCtgR testCtg(IntegerVector((SEXP) sYTest), levelsTrain);
    if (!pBridge->permutes()) {
      summaryCtg = List::create(
			      _["prediction"] = getPrediction(pBridge, levelsTrain, ctgNames),
			      _["validation"] = testCtg.getValidation(pBridge)
			      );
    }
    else {
      summaryCtg = List::create(
			      _["prediction"] = getPrediction(pBridge, levelsTrain, ctgNames),
			      _["validation"] = testCtg.getValidation(pBridge),
			      _["importance"] = testCtg.getImportance(pBridge, SignatureR::unwrapColNames(lDeframe))
			      );
    }
  }

  summaryCtg.attr("class") = "SummaryCtg";
  return summaryCtg;
}


// [[Rcpp::export]]
List LeafCtgRf::getPrediction(const PredictCtgBridge* pBridge,
			      const CharacterVector& levelsTrain,
			      const CharacterVector& ctgNames) {
  auto yPred = pBridge->getYPred();
  IntegerVector yPredZero(yPred.begin(), yPred.end());
  IntegerVector yPredOne(yPredZero + 1);
  yPredOne.attr("class") = "factor";
  yPredOne.attr("levels") = levelsTrain;
  List prediction = List::create(
				 _["yPred"] = yPredOne,
				 _["census"] = getCensus(pBridge, levelsTrain, ctgNames),
				 _["prob"] = getProb(pBridge, levelsTrain, ctgNames),
				 _["indices"] = getIndices(pBridge)
				 );
  prediction.attr("class") = "PredictCtg";
  return prediction;
}


// [[Rcpp::export]]
NumericMatrix LeafCtgRf::getIndices(const PredictCtgBridge* pBridge) {
  auto indices = pBridge->getIndices();
  size_t nObs = pBridge->getNObs();
  return indices.empty() ? NumericMatrix(0) : NumericMatrix(nObs, indices.size() / nObs, indices.begin());
}


// [[Rcpp::export]]
List TestCtgR::getValidation(const PredictCtgBridge* pBridge) {
  List validCtg = List::create(
			       _["confusion"] = getConfusion(pBridge, levelsTrain),
			       _["misprediction"] = getMisprediction(pBridge),
			       _["oobError"] = pBridge->getOOBError()
			       );
  validCtg.attr("class") = "ValidCtg";
  return validCtg;
}


// [[Rcpp::export]]
List TestCtgR::getImportance(const PredictCtgBridge* pBridge,
			    const CharacterVector& predNames) {
  List importanceCtg = List::create(
				    _["oobErr"] = oobErrPermuted(pBridge, predNames),
				    _["mispred"] = mispredPermuted(pBridge, predNames)
				    );
  importanceCtg.attr("class") = "importanceCtg";
  return importanceCtg;
}


// [[Rcpp::export]]
NumericVector TestCtgR::getMisprediction(const PredictCtgBridge* pBridge) const {
  auto mispred = pBridge->getMisprediction();
  NumericVector mispredOut = as<NumericVector>(NumericVector(mispred.begin(), mispred.end())[test2Merged]);
  mispredOut.attr("names") = levels;
  return mispredOut;
}


// [[Rcpp::export]]
List TestCtgR::mispredPermuted(const PredictCtgBridge* pBridge,
			       const CharacterVector& predNames) const {
  vector<vector<vector<double>>> mispredCore = pBridge->getMispredPermuted();
  unsigned int nPred = mispredCore.size();
  unsigned int nPermute = mispredCore[0].size();
  unsigned int nCtg = mispredCore[0][0].size();

  List mispredOut(nPred);
  for (unsigned int predIdx = 0; predIdx != nPred; predIdx++) {
    mispredOut(predIdx) = NumericMatrix(nPermute, nCtg);
    NumericMatrix&& predMispredict = as<NumericMatrix>(mispredOut[predIdx]);
    predMispredict.attr("dimnames") = List::create(CharacterVector(nPermute), levels);
    for (unsigned int permIdx = 0; permIdx != nPermute; permIdx++) {
      predMispredict.row(permIdx) = NumericVector(mispredCore[predIdx][permIdx].begin(), mispredCore[predIdx][permIdx].end());
    }
  }
  mispredOut.attr("names") = predNames;
  
  return mispredOut;
}


// [[Rcpp::export]]
NumericMatrix TestCtgR::oobErrPermuted(const PredictCtgBridge* pBridge,
				     const CharacterVector& predNames) const {
  vector<vector<double>> oobPerm = pBridge->getOOBErrorPermuted();
  unsigned int nPerm = oobPerm[0].size();
  unsigned int nPred = oobPerm.size();
  NumericMatrix oobErrOut(nPerm, nPred);
  for (unsigned int predIdx = 0; predIdx != nPred; predIdx++) {
    oobErrOut.column(predIdx) = NumericVector(oobPerm[predIdx].begin(), oobPerm[predIdx].end());
  }
  oobErrOut.attr("dimnames") = List::create(CharacterVector(nPerm), predNames);

  return oobErrOut;
}


// [[Rcpp::export]]
IntegerMatrix LeafCtgRf::getCensus(const PredictCtgBridge* pBridge,
                                   const CharacterVector& levelsTrain,
                                   const CharacterVector& ctgNames) {
  IntegerMatrix census = transpose(IntegerMatrix(levelsTrain.length(), pBridge->getNObs(), &(pBridge->getCensus())[0]));
  census.attr("dimnames") = List::create(ctgNames, levelsTrain);
  return census;
}


// [[Rcpp::export]]
NumericMatrix LeafCtgRf::getProb(const PredictCtgBridge* pBridge,
                                 const CharacterVector& levelsTrain,
                                 const CharacterVector& ctgNames) {
  if (!pBridge->getProb().empty()) {
    NumericMatrix prob = transpose(NumericMatrix(levelsTrain.length(), pBridge->getNObs(), &(pBridge->getProb())[0]));
    prob.attr("dimnames") = List::create(ctgNames, levelsTrain);
    return prob;
  }
  else {
    return NumericMatrix(0);
  }
}


// [[Rcpp::export]]
NumericMatrix TestCtgR::getConfusion(const PredictCtgBridge* pBridge,
				    const CharacterVector& levelsTrain) const {
  // Converts to numeric vector to accommodate wide rows in R.
  auto confusion = pBridge->getConfusion();
  NumericVector confNum(confusion.begin(), confusion.end());
  unsigned int ctgTrain = levelsTrain.length();
  unsigned int ctgTest = levels.length();
  NumericMatrix conf = transpose(NumericMatrix(ctgTrain, ctgTest, &confNum[0]));
  NumericMatrix confOut(ctgTest, ctgTrain);
  for (unsigned int i = 0; i < ctgTest; i++) {
    confOut(i, _) = conf(test2Merged[i], _);
  }
  confOut.attr("dimnames") = List::create(levels, levelsTrain);

  return confOut;
}
