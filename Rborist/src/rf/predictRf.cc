// Copyright (C)  2012-2020  Mark Seligman
//
// This file is part of rfR.
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
#include "predictRf.h"
#include "bagbridge.h"
#include "bagRf.h"
#include "forestRf.h"
#include "forestbridge.h"
#include "leafbridge.h"
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
                                          bool oob,
					  unsigned int nPermute,
                                          unsigned int nThread,
                                          vector<double> quantile) {
  List lLeaf(checkLeafReg(lTrain));
  return make_unique<PredictRegBridge>(RLEFrameR::unwrap(lDeframe),
				       ForestRf::unwrap(lTrain),
				       BagRf::unwrap(lTrain, lDeframe, oob),
				       LeafPredictRf::unwrap(lTrain, lDeframe),
				       move(regTrain(lLeaf)),
				       meanTrain(lLeaf),
				       move(regTest(sYTest)),
				       oob,
				       nPermute,
				       nThread,
				       move(quantile));
}


vector<double> PBRf::regTrain(const List& lLeaf) {
  NumericVector yTrainFE((SEXP) lLeaf["yTrain"]);
  vector<double> yTrain(yTrainFE.begin(), yTrainFE.end());

  return yTrain;
}


vector<double> PBRf::regTest(SEXP sYTest) {
  vector<double> yTest;
  if (!Rf_isNull(sYTest)) {
    NumericVector yTestFE((SEXP) sYTest);
    yTest = as<vector<double>>(yTestFE);
  }
  return yTest;
}


double PBRf::meanTrain(const List& lLeaf) {
  return mean(NumericVector((SEXP) lLeaf["yTrain"]));
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
    CharacterVector predNames(Signature::unwrapColNames(lDeframe));
    summaryReg = List::create(
			      _["prediction"] = getPrediction(pBridge),
			      _["validation"] = getValidation(pBridge, NumericVector((SEXP)sYTest)),
			      _["importance"] = getImportance(pBridge, NumericVector((SEXP) sYTest), predNames)
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
                      bool oob,
                      bool doProb,
		      unsigned int permute,
                      unsigned int nThread) {
  BEGIN_RCPP

    unique_ptr<PredictCtgBridge> pBridge(unwrapCtg(lDeframe, lTrain, sYTest, oob, doProb, permute, nThread));
  pBridge->predict();

  return LeafCtgRf::summary(lDeframe, lTrain, pBridge.get(), sYTest);

  END_RCPP
}


unique_ptr<PredictCtgBridge> PBRf::unwrapCtg(const List& lDeframe,
                                          const List& lTrain,
					  SEXP sYTest,
                                          bool oob,
                                          bool doProb,
					  unsigned int permute,
                                          unsigned int nThread) {

  List lLeaf(checkLeafCtg(lTrain));
 return make_unique<PredictCtgBridge>(RLEFrameR::unwrap(lDeframe),
				       ForestRf::unwrap(lTrain),
				       BagRf::unwrap(lTrain, lDeframe, oob),
				       LeafPredictRf::unwrap(lTrain, lDeframe),
				       (unsigned int*) IntegerVector((SEXP) lLeaf["nodeHeight"]).begin(),
				       (double*) NumericVector((SEXP) lLeaf["weight"]).begin(),
				      ctgTrain(lLeaf),
				       move(ctgTest(lLeaf, sYTest)),
				       oob,
				       permute,
				       doProb,
				       nThread);
}


vector<unsigned int> PBRf::ctgTest(const List& lLeaf, SEXP sYTest) {
  if (!Rf_isNull(sYTest)) { // Makes zero-based copy.
    TestCtg testCtg(sYTest, as<CharacterVector>(lLeaf["levels"]));
    return testCtg.yTestZero;
  }
  else {
    vector<unsigned int> yTest;
    return yTest;
  }  
}


unsigned int PBRf::ctgTrain(const List& lLeaf) {
  return CharacterVector((SEXP) lLeaf["levels"]).length();
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
                        bool oob,
			unsigned int permute,
                        unsigned int nThread) {
  BEGIN_RCPP

  NumericVector quantVec(sQuantVec);
  vector<double> quantile(quantVec.begin(), quantVec.end());
  unique_ptr<PredictRegBridge> pBridge(unwrapReg(lDeframe, lTrain, sYTest, oob, permute, nThread, move(quantile)));
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
  vector<double> qPred(pBridge->getQPred());
  return qPred.empty() ? NumericMatrix(0) : transpose(NumericMatrix(qPred.size() / nRow, nRow, qPred.begin()));
    
  END_RCPP
}


// EXIT:  Conversion to NumericVector should be implicit from RCPP.
NumericVector PBRf::getQEst(const PredictRegBridge* pBridge) {
  BEGIN_RCPP

  vector<double> qEst(pBridge->getQEst());
  return NumericVector(qEst.begin(), qEst.end());

  END_RCPP
}


List PBRf::getValidation(const PredictRegBridge* pBridge,
			 const NumericVector& yTestFE) {
  BEGIN_RCPP

    //  const vector<double>& yTest = pBridge->getYTest();
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

  const vector<double>& ssePerm = pBridge->getSSEPermute();
  NumericVector mseOut(ssePerm.begin(), ssePerm.end());
  mseOut = mseOut / yTestFE.length();
  mseOut.attr("names") = predNames;

  List importance = List::create(_["msePermuted"] = mseOut);
  importance("class") = "ImportanceReg";
  return importance;

  END_RCPP
}

/**
   @brief References front-end member arrays and instantiates
   bridge-specific PredictReg handle.
 */
unique_ptr<LeafBridge> LeafPredictRf::unwrap(const List& lTrain,
					     const List& lDeframe) {
  List lLeaf((SEXP) lTrain["leaf"]);
  return make_unique<LeafBridge>((unsigned int*) IntegerVector((SEXP) lLeaf["nodeHeight"]).begin(),
				 (size_t) IntegerVector((SEXP) lLeaf["nodeHeight"]).length(),
				 (unsigned char*) RawVector((SEXP) lLeaf["node"]).begin(),
				 (unsigned int*) IntegerVector((SEXP) lLeaf["bagHeight"]).begin(),
				 (unsigned char*) RawVector((SEXP) lLeaf["bagSample"]).begin());
}


List PBRf::checkLeafReg(const List &lTrain) {
  BEGIN_RCPP

  List lLeaf((SEXP) lTrain["leaf"]);
  if (!lLeaf.inherits("LeafReg")) {
    stop("Expecting LeafReg");
  }

  return lLeaf;

  END_RCPP
}


/**
   @brief Ensures front end holds a PredictCtg.
 */
List PBRf::checkLeafCtg(const List &lTrain) {
  BEGIN_RCPP

  List leafCtg((SEXP) lTrain["leaf"]);
  if (!leafCtg.inherits("LeafCtg")) {
    stop("Expecting LeafCtg");
  }

  return leafCtg;

  END_RCPP
}


TestCtg::TestCtg(const IntegerVector& yTestOne,
                 const CharacterVector& levelsTrain_) :
  levelsTrain(levelsTrain_),
  levels(CharacterVector((SEXP) yTestOne.attr("levels"))),
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
  IntegerVector yZero = yTestOne -1;
  vector<unsigned int> yZeroOut(yZero.length());
  for (R_len_t i = 0; i < yZero.length(); i++) {
    yZeroOut[i] = test2Merged[yZero[i]];
  }
  return yZeroOut;
}


List LeafCtgRf::summary(const List& lDeframe, const List& lTrain, const PredictCtgBridge* pBridge, SEXP sYTest) {
  BEGIN_RCPP

  List lLeaf((SEXP) lTrain["leaf"]);
  CharacterVector levelsTrain((SEXP) lLeaf["levels"]);
  CharacterVector ctgNames(Signature::unwrapRowNames(lDeframe));

  List summaryCtg;
  if (Rf_isNull(sYTest)) {
    summaryCtg = List::create(
			      _["prediction"] = getPrediction(pBridge, levelsTrain, ctgNames)
			      );
  }
  else if (!pBridge->permutes()) {
    TestCtg testCtg(IntegerVector((SEXP) sYTest), levelsTrain);
    summaryCtg = List::create(
			      _["prediction"] = getPrediction(pBridge, levelsTrain, ctgNames),
			      _["validation"] = testCtg.getValidation(pBridge)
			      );
  }
  else {
    TestCtg testCtg(IntegerVector((SEXP) sYTest), levelsTrain);
    CharacterVector predNames(Signature::unwrapColNames(lDeframe));
    summaryCtg = List::create(
			      _["prediction"] = getPrediction(pBridge, levelsTrain, ctgNames),
			      _["validation"] = testCtg.getValidation(pBridge),
			      _["importance"] = testCtg.getImportance(pBridge, predNames)
			      );
  }

  summaryCtg.attr("class") = "SummaryCtg";
  return summaryCtg;

  END_RCPP
}


List LeafCtgRf::getPrediction(const PredictCtgBridge* pBridge,
			      const CharacterVector& levelsTrain,
			      const CharacterVector& ctgNames) {
  BEGIN_RCPP
  const vector<unsigned int>& yPred = pBridge->getYPred();
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
				    _["mispredPermuted"] = mispredPermute(pBridge, predNames),
				    _["oobErrPermuted"] = oobErrPermute(pBridge, predNames)
				    );
  importanceCtg.attr("class") = "importanceCtg";
  return importanceCtg;
  
  END_RCPP
}


NumericVector TestCtg::getMisprediction(const PredictCtgBridge* pBridge) const {
  BEGIN_RCPP

  const vector<double>& mispred = pBridge->getMisprediction();
  NumericVector misPred(mispred.begin(), mispred.end());
  NumericVector misPredOut = misPred[test2Merged];
  misPredOut.attr("names") = levels;
  return misPredOut;

  END_RCPP
}



NumericMatrix TestCtg::mispredPermute(const PredictCtgBridge* pBridge,
				      const CharacterVector& predNames) const {
  BEGIN_RCPP

  const vector<vector<double>> impCore(pBridge->getMispredPermute());

  unsigned int i = 0;
  NumericMatrix impOut(levels.length(), impCore.size());
  for (auto mispred : impCore) {
    NumericVector m(mispred.begin(), mispred.end());
    NumericVector mispredOut = m[test2Merged];
    impOut.column(i++) = mispredOut;
  }
  impOut.attr("dimnames") = List::create(levels, predNames);
  
  return impOut;
  END_RCPP
}


NumericVector TestCtg::oobErrPermute(const PredictCtgBridge* pBridge,
				     const CharacterVector& predNames) const {
  BEGIN_RCPP

  const vector<double>& oobPerm = pBridge->getOOBErrorPermute();
  NumericVector errOut(oobPerm.begin(), oobPerm.end());
  errOut.attr("names") = predNames;

  return errOut;
  END_RCPP
}


IntegerMatrix LeafCtgRf::getCensus(const PredictCtgBridge* pBridge,
                                   const CharacterVector& levelsTrain,
                                   const CharacterVector& ctgNames) {
  BEGIN_RCPP
  IntegerMatrix census = transpose(IntegerMatrix(pBridge->getNCtgTrain(), pBridge->getNRow(), pBridge->getCensus()));
  census.attr("dimnames") = List::create(ctgNames, levelsTrain);
  return census;
  END_RCPP
}


NumericMatrix LeafCtgRf::getProb(const PredictCtgBridge* pBridge,
                                 const CharacterVector& levelsTrain,
                                 const CharacterVector& ctgNames) {
  BEGIN_RCPP
  if (!pBridge->getProb().empty()) {
    NumericMatrix prob = transpose(NumericMatrix(pBridge->getNCtgTrain(), pBridge->getNRow(), &(pBridge->getProb())[0]));
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
  const vector<size_t>& confusion = pBridge->getConfusion();
  NumericVector confNum(confusion.begin(), confusion.end());
  unsigned int ctgTrain = pBridge->getNCtgTrain();
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
