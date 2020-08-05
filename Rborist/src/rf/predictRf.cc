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
#include "leafRf.h"
#include "leafbridge.h"
#include "rleframeR.h"

#include <algorithm>

RcppExport SEXP ValidateReg(const SEXP sDeframe,
                            const SEXP sTrain,
                            SEXP sYTest,
			    SEXP sImportance,
                            SEXP sNThread) {
  BEGIN_RCPP

    return PBRf::predictReg(List(sDeframe), List(sTrain), sYTest, true, as<bool>(sImportance), as<unsigned int>(sNThread));

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
		      bool importance,
		      unsigned int nThread) {
  BEGIN_RCPP
    
  unique_ptr<PredictBridge> pBridge(unwrapReg(lDeframe, lTrain, oob, importance, nThread));
  pBridge->predict();

  return LeafRegRf::summary(sYTest, pBridge.get());
  
  END_RCPP
}


unique_ptr<PredictBridge> PBRf::unwrapCtg(const List& lDeframe,
                                          const List& lTrain,
                                          bool oob,
                                          bool doProb,
					  bool importance,
                                          unsigned int nThread) {
  return make_unique<PredictBridge>(RLEFrameR::unwrap(lDeframe),
                                    ForestRf::unwrap(lTrain),
                                    BagRf::unwrap(lTrain, lDeframe, oob),
                                    LeafCtgRf::unwrap(lTrain, lDeframe, doProb),
				    importance,
                                    nThread);
}


unique_ptr<PredictBridge> PBRf::unwrapReg(const List& lDeframe,
                                          const List& lTrain,
                                          bool oob,
					  bool importance,
                                          unsigned int nThread) {
  return make_unique<PredictBridge>(RLEFrameR::unwrap(lDeframe),
                                    ForestRf::unwrap(lTrain),
                                    BagRf::unwrap(lTrain, lDeframe, oob),
                                    LeafRegRf::unwrap(lTrain, lDeframe),
				    importance,
                                    nThread);
}


unique_ptr<PredictBridge> PBRf::unwrapReg(const List& lDeframe,
                                          const List& lTrain,
                                          bool oob,
					  bool importance,
                                          unsigned int nThread,
                                          const vector<double>& quantile) {
  return make_unique<PredictBridge>(RLEFrameR::unwrap(lDeframe),
                                    ForestRf::unwrap(lTrain),
                                    BagRf::unwrap(lTrain, lDeframe, oob),
                                    LeafRegRf::unwrap(lTrain, lDeframe),
				    importance,
                                    quantile,
                                    nThread);
}


RcppExport SEXP ValidateVotes(const SEXP sDeframe,
                              const SEXP sTrain,
                              SEXP sYTest,
			      SEXP sImportance,
                              SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictCtg(List(sDeframe), List(sTrain), sYTest, true, false, as<bool>(sImportance), as<unsigned int>(sNThread));

  END_RCPP
}


RcppExport SEXP ValidateProb(const SEXP sDeframe,
                             const SEXP sTrain,
                             SEXP sYTest,
			     SEXP sImportance,
                             SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictCtg(List(sDeframe), List(sTrain), sYTest, true, true, as<bool>(sImportance), as<unsigned int>(sNThread));

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
		      bool importance,
                      unsigned int nThread) {
  BEGIN_RCPP

    unique_ptr<PredictBridge> pBridge(unwrapCtg(lDeframe, lTrain, oob, doProb, importance, nThread));
  pBridge->predict();

  return LeafCtgRf::summary(lDeframe, lTrain, pBridge.get(), sYTest);

  END_RCPP
}


RcppExport SEXP ValidateQuant(const SEXP sDeframe,
                              const SEXP sTrain,
                              SEXP sYTest,
			      SEXP sImportance,
                              SEXP sQuantVec,
                              SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictQuant(List(sDeframe), sTrain, sQuantVec, sYTest, true, as<bool>(sImportance), as<unsigned int>(sNThread));

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
			bool importance,
                        unsigned int nThread) {
  BEGIN_RCPP

  NumericVector quantVec(sQuantVec);
  vector<double> quantile(quantVec.begin(), quantVec.end());
  unique_ptr<PredictBridge> pBridge(unwrapReg(lDeframe, lTrain, oob, importance, nThread, quantile));
  pBridge->predict();

  return LeafRegRf::summary(sYTest, pBridge.get());
  
  END_RCPP
}
