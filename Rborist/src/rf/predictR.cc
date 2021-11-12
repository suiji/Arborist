// Copyright (C)  2012-2021  Mark Seligman
//
// This file is part of rf.
//
// rfR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rfR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file predictRf.cc

   @brief C++ interface to R entry for prediction methods.

   @author Mark Seligman
 */

#include "predictbridge.h"
#include "predictR.h"
#include "samplerbridge.h"
#include "samplerR.h"
#include "forestR.h"
#include "forestbridge.h"
#include "rleframeR.h"
#include "signature.h"

#include <algorithm>

RcppExport SEXP ValidateReg(const SEXP sDeframe,
                            const SEXP sTrain,
                            SEXP sYTest,
			    SEXP sPermute,
                            SEXP sNThread) {
  BEGIN_RCPP

    return PBRf::predictReg(List(sDeframe), List(sTrain), sYTest, true, as<bool>(sPermute), as<unsigned int>(sNThread));

  END_RCPP
}


RcppExport SEXP TestReg(const SEXP sDeframe,
                        const SEXP sTrain,
                        SEXP sYTest,
                        SEXP sOOB,
                        SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictReg(List(sDeframe), List(sTrain), sYTest, as<bool>(sOOB), false, as<unsigned int>(sNThread));

  END_RCPP
}


List PBRf::predictReg(const List& lDeframe,
		      const List& lTrain,
		      SEXP sYTest,
		      bool oob,
		      unsigned int nPermute,
		      unsigned int nThread) {
  BEGIN_RCPP

  unique_ptr<PredictRegBridge> pBridge(unwrapReg(lDeframe, lTrain, sYTest, oob, nPermute, nThread));
  pBridge->predict();

  return summary(lDeframe, sYTest, pBridge.get());
  
  END_RCPP
}


unique_ptr<PredictRegBridge> PBRf::unwrapReg(const List& lDeframe,
                                          const List& lTrain,
					  SEXP sYTest,
                                          bool bagging,
					  unsigned int nPermute,
                                          unsigned int nThread,
                                          vector<double> quantile) {
  unique_ptr<RLEFrame> rleFrame(RLEFrameR::unwrap(lDeframe));
  unique_ptr<ForestBridge> forestBridge(ForestRf::unwrap(lTrain));
  return make_unique<PredictRegBridge>(move(rleFrame),
				       move(forestBridge),
				       move(SamplerR::unwrap(lTrain, lDeframe, bagging)),
				       move(regTest(sYTest)),
				       nPermute,
				       nThread,
				       move(quantile));
}


vector<double> PBRf::regTest(SEXP sYTest) {
  vector<double> yTest;
  if (!Rf_isNull(sYTest)) {
    NumericVector yTestFE((SEXP) sYTest);
    yTest = as<vector<double>>(yTestFE);
  }
  return yTest;
}


List PBRf::summary(const List& lDeframe, SEXP sYTest, const PredictRegBridge* pBridge) {
  BEGIN_RCPP

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
			      _["importance"] = getImportance(pBridge, NumericVector((SEXP) sYTest), Signature::unwrapColNames(lDeframe))
			      );
  }
  summaryReg.attr("class") = "SummaryReg";

  return summaryReg;
  END_RCPP
}


RcppExport SEXP ValidateVotes(const SEXP sDeframe,
                              const SEXP sTrain,
                              SEXP sYTest,
			      SEXP sPermute,
                              SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictCtg(List(sDeframe), List(sTrain), sYTest, true, false, as<unsigned int>(sPermute), as<unsigned int>(sNThread));

  END_RCPP
}


RcppExport SEXP ValidateProb(const SEXP sDeframe,
                             const SEXP sTrain,
                             SEXP sYTest,
			     SEXP sPermute,
                             SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictCtg(List(sDeframe), List(sTrain), sYTest, true, true, as<unsigned int>(sPermute), as<unsigned int>(sNThread));

  END_RCPP
}


RcppExport SEXP TestVotes(const SEXP sDeframe,
                          const SEXP sTrain,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictCtg(List(sDeframe), List(sTrain), sYTest, as<bool>(sOOB), false, false, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP TestProb(const SEXP sDeframe,
                         const SEXP sTrain,
                         SEXP sYTest,
                         SEXP sOOB,
                         SEXP sNThread) {
  BEGIN_RCPP
  return PBRf::predictCtg(List(sDeframe), List(sTrain), sYTest, as<bool>(sOOB), true, false, as<unsigned int>(sNThread));
  END_RCPP
}


List PBRf::predictCtg(const List& lDeframe,
                      const List& lTrain,
                      SEXP sYTest,
                      bool bagging,
                      bool doProb,
		      unsigned int permute,
                      unsigned int nThread) {
  BEGIN_RCPP

  unique_ptr<PredictCtgBridge> pBridge(unwrapCtg(lDeframe, lTrain, sYTest, bagging, doProb, permute, nThread));
  pBridge->predict();

  return LeafCtgRf::summary(lDeframe, lTrain, pBridge.get(), sYTest);

  END_RCPP
}


unique_ptr<PredictCtgBridge> PBRf::unwrapCtg(const List& lDeframe,
					     const List& lTrain,
					     SEXP sYTest,
					     bool bagging,
					     bool doProb,
					     unsigned int permute,
					     unsigned int nThread) {
  unique_ptr<RLEFrame> rleFrame(RLEFrameR::unwrap(lDeframe));
  unique_ptr<ForestBridge> forestBridge(ForestRf::unwrap(lTrain));
  return make_unique<PredictCtgBridge>(move(rleFrame),
				       move(forestBridge),
				       move(SamplerR::unwrap(lTrain, lDeframe, bagging)),
				       move(ctgTest(lTrain, sYTest)),
				       permute,
				       doProb,
				       nThread);
}


vector<unsigned int> PBRf::ctgTest(const List& lTrain, SEXP sYTest) {
  List lSampler((SEXP) lTrain["sampler"]);
  if (!Rf_isNull(sYTest)) { // Makes zero-based copy.
    IntegerVector yTrain(as<IntegerVector>(lSampler["yTrain"]));
    TestCtg testCtg(sYTest, as<CharacterVector>(yTrain.attr("levels")));
    return testCtg.yTestZero;
  }
  else {
    vector<unsigned int> yTest;
    return yTest;
  }  
}


RcppExport SEXP ValidateQuant(const SEXP sDeframe,
                              const SEXP sTrain,
                              SEXP sYTest,
			      SEXP sPermute,
                              SEXP sQuantVec,
                              SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictQuant(List(sDeframe), sTrain, sQuantVec, sYTest, true, as<unsigned int>(sPermute), as<unsigned int>(sNThread));

  END_RCPP
}


 RcppExport SEXP TestQuant(const SEXP sDeframe,
                          const SEXP sTrain,
                          SEXP sQuantVec,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictQuant(List(sDeframe), sTrain, sQuantVec, sYTest, as<bool>(sOOB), false, as<unsigned int>(sNThread));

  END_RCPP
}


List PBRf::predictQuant(const List& lDeframe,
                        const List& lTrain,
                        SEXP sQuantVec,
                        SEXP sYTest,
                        bool bagging,
			unsigned int permute,
                        unsigned int nThread) {
  BEGIN_RCPP

  NumericVector quantVec(sQuantVec);
  vector<double> quantile(quantVec.begin(), quantVec.end());
  unique_ptr<PredictRegBridge> pBridge(unwrapReg(lDeframe, lTrain, sYTest, bagging, permute, nThread, move(quantile)));
  pBridge->predict();

  return summary(lDeframe, sYTest, pBridge.get());
  
  END_RCPP
}


List PBRf::getPrediction(const PredictRegBridge* pBridge) {
  BEGIN_RCPP

  List prediction = List::create(
				 _["yPred"] = pBridge->getYPred(),
				 _["qPred"] = getQPred(pBridge),
				 _["qEst"] = getQEst(pBridge)
				 );
  prediction.attr("class") = "PredictReg";
  return prediction;

  END_RCPP
}


NumericMatrix PBRf::getQPred(const PredictRegBridge* pBridge) {
  BEGIN_RCPP

  size_t nRow(pBridge->getNRow());
  auto qPred = pBridge->getQPred();
  return qPred.empty() ? NumericMatrix(0) : transpose(NumericMatrix(qPred.size() / nRow, nRow, qPred.begin()));
    
  END_RCPP
}


// EXIT:  Conversion to NumericVector should be implicit from RCPP.
NumericVector PBRf::getQEst(const PredictRegBridge* pBridge) {
  BEGIN_RCPP

  auto qEst = pBridge->getQEst();
  return NumericVector(qEst.begin(), qEst.end());

  END_RCPP
}


List PBRf::getValidation(const PredictRegBridge* pBridge,
			 const NumericVector& yTestFE) {
  BEGIN_RCPP

  double sse = pBridge->getSSE();
  size_t nRow = yTestFE.length();
  List validation = List::create(_["mse"] = sse / nRow,
				 _["rsq"] = nRow == 1 ? 0.0 : 1.0 - sse / (var(yTestFE) * (nRow - 1)),
				 _["mae"] = pBridge->getSAE() / nRow
				 );
  validation.attr("class") = "ValidReg";
  return validation;

  END_RCPP
}


List PBRf::getImportance(const PredictRegBridge* pBridge,
			 const NumericVector& yTestFE,
			 const CharacterVector& predNames) {
  BEGIN_RCPP

  auto ssePerm = pBridge->getSSEPermuted();
  NumericVector mseOut(ssePerm.begin(), ssePerm.end());
  mseOut = mseOut / yTestFE.length();
  mseOut.attr("names") = predNames;

  List importance = List::create(_["msePermuted"] = mseOut);
  importance("class") = "ImportanceReg";
  return importance;

  END_RCPP
}


TestCtg::TestCtg(const IntegerVector& yTestOne,
                 const CharacterVector& levelsTrain_) :
  levelsTrain(levelsTrain_),
  levels(CharacterVector(as<CharacterVector>(yTestOne.attr("levels")))),
  test2Merged(mergeLevels(levels)),
  yTestZero(reconcile(test2Merged, yTestOne)),
  ctgMerged(*max_element(yTestZero.begin(), yTestZero.end()) + 1) {
}


IntegerVector TestCtg::mergeLevels(const CharacterVector& levelsTest) {
  BEGIN_RCPP
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
  END_RCPP
}


vector<unsigned int> TestCtg::reconcile(const IntegerVector& test2Merged,
					const IntegerVector& yTestOne) {
  IntegerVector yZero = yTestOne - 1;
  vector<unsigned int> yZeroOut(yZero.length());
  for (R_len_t i = 0; i < yZero.length(); i++) {
    yZeroOut[i] = test2Merged[yZero[i]];
  }
  return yZeroOut;
}


List LeafCtgRf::summary(const List& lDeframe, const List& lTrain, const PredictCtgBridge* pBridge, SEXP sYTest) {
  BEGIN_RCPP

  List lSampler((SEXP) lTrain["sampler"]);
  IntegerVector yTrain(as<IntegerVector>(lSampler["yTrain"]));
  CharacterVector levelsTrain(as<CharacterVector>(yTrain.attr("levels")));
  CharacterVector ctgNames(Signature::unwrapRowNames(lDeframe));

  List summaryCtg;
  if (Rf_isNull(sYTest)) {
    summaryCtg = List::create(
			      _["prediction"] = getPrediction(pBridge, levelsTrain, ctgNames)
			      );
  }
  else {
    TestCtg testCtg(IntegerVector((SEXP) sYTest), levelsTrain);
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
			      _["importance"] = testCtg.getImportance(pBridge, Signature::unwrapColNames(lDeframe))
			      );
    }
  }

  summaryCtg.attr("class") = "SummaryCtg";
  return summaryCtg;

  END_RCPP
}


List LeafCtgRf::getPrediction(const PredictCtgBridge* pBridge,
			      const CharacterVector& levelsTrain,
			      const CharacterVector& ctgNames) {
  BEGIN_RCPP
  auto yPred = pBridge->getYPred();
  IntegerVector yPredZero(yPred.begin(), yPred.end());
  IntegerVector yPredOne(yPredZero + 1);
  yPredOne.attr("class") = "factor";
  yPredOne.attr("levels") = levelsTrain;
  List prediction = List::create(
				 _["yPred"] = yPredOne,
				 _["census"] = getCensus(pBridge, levelsTrain, ctgNames),
				 _["prob"] = getProb(pBridge, levelsTrain, ctgNames)
				 );
  prediction.attr("class") = "PredictCtg";
  return prediction;

  END_RCPP
}


List TestCtg::getValidation(const PredictCtgBridge* pBridge) {
  BEGIN_RCPP
  List validCtg = List::create(
			       _["confusion"] = getConfusion(pBridge, levelsTrain),
			       _["misprediction"] = getMisprediction(pBridge),
			       _["oobError"] = pBridge->getOOBError()
			       );
  validCtg.attr("class") = "ValidCtg";
  return validCtg;
  
  END_RCPP
}


List TestCtg::getImportance(const PredictCtgBridge* pBridge,
			    const CharacterVector& predNames) {
  BEGIN_RCPP

  List importanceCtg = List::create(
				    _["mispredPermuted"] = mispredPermuted(pBridge, predNames),
				    _["oobErrPermuted"] = oobErrPermuted(pBridge, predNames)
				    );
  importanceCtg.attr("class") = "importanceCtg";
  return importanceCtg;
  
  END_RCPP
}


NumericVector TestCtg::getMisprediction(const PredictCtgBridge* pBridge) const {
  BEGIN_RCPP

  auto mispred = pBridge->getMisprediction();
  NumericVector mispredOut = as<NumericVector>(NumericVector(mispred.begin(), mispred.end())[test2Merged]);
  mispredOut.attr("names") = levels;
  return mispredOut;

  END_RCPP
}


NumericMatrix TestCtg::mispredPermuted(const PredictCtgBridge* pBridge,
				      const CharacterVector& predNames) const {
  BEGIN_RCPP

  auto mispredCore = pBridge->getMispredPermuted();
  NumericMatrix mispredOut(levels.length(), mispredCore.size());

  unsigned int col = 0;
  for (auto mispred : mispredCore) {
    mispredOut.column(col++) = as<NumericVector>(NumericVector(mispred.begin(), mispred.end())[test2Merged]);
  }
  mispredOut.attr("dimnames") = List::create(levels, predNames);
  
  return mispredOut;
  END_RCPP
}


NumericVector TestCtg::oobErrPermuted(const PredictCtgBridge* pBridge,
				     const CharacterVector& predNames) const {
  BEGIN_RCPP

  auto oobPerm = pBridge->getOOBErrorPermuted();
  NumericVector errOut(oobPerm.begin(), oobPerm.end());
  errOut.attr("names") = predNames;

  return errOut;
  END_RCPP
}


IntegerMatrix LeafCtgRf::getCensus(const PredictCtgBridge* pBridge,
                                   const CharacterVector& levelsTrain,
                                   const CharacterVector& ctgNames) {
  BEGIN_RCPP
  IntegerMatrix census = transpose(IntegerMatrix(levelsTrain.length(), pBridge->getNRow(), pBridge->getCensus()));
  census.attr("dimnames") = List::create(ctgNames, levelsTrain);
  return census;
  END_RCPP
}


NumericMatrix LeafCtgRf::getProb(const PredictCtgBridge* pBridge,
                                 const CharacterVector& levelsTrain,
                                 const CharacterVector& ctgNames) {
  BEGIN_RCPP
  if (!pBridge->getProb().empty()) {
    NumericMatrix prob = transpose(NumericMatrix(levelsTrain.length(), pBridge->getNRow(), &(pBridge->getProb())[0]));
    prob.attr("dimnames") = List::create(ctgNames, levelsTrain);
    return prob;
  }
  else {
    return NumericMatrix(0);
  }
  END_RCPP
}


NumericMatrix TestCtg::getConfusion(const PredictCtgBridge* pBridge,
				    const CharacterVector& levelsTrain) const {
  BEGIN_RCPP

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
  END_RCPP
}
