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
#include "rleframe.h"

#include <algorithm>
RcppExport SEXP ValidateReg(const SEXP sDeframe,
                            const SEXP sTrain,
                            SEXP sYTest,
                            SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictReg(List(sDeframe), List(sTrain), sYTest, true, as<unsigned int>(sNThread));

  END_RCPP
}


RcppExport SEXP TestReg(const SEXP sDeframe,
                        const SEXP sTrain,
                        SEXP sYTest,
                        SEXP sOOB,
                        SEXP sNThread) {
  BEGIN_RCPP

  List lDeframe(sDeframe);
  return PBRf::predictReg(lDeframe, List(sTrain), sYTest, as<bool>(sOOB), as<unsigned int>(sNThread));

  END_RCPP
}

/**
   @brief Predction for regression.

   @return Wrapped zero, with copy-out parameters.
 */
List PBRf::predictReg(const List& lDeframe,
		      const List& lTrain,
		      SEXP sYTest,
		      bool oob,
		      unsigned int nThread) {
  BEGIN_RCPP
    unique_ptr<PredictBridge> pBridge(unwrapReg(lDeframe, lTrain, oob, nThread));

  predict(pBridge.get(), getNRow(lDeframe));

  return LeafRegRf::summary(sYTest, pBridge.get());
  
  END_RCPP
}


size_t PBRf::getNRow(const List& lDeframe) {
  return as<size_t>((SEXP) lDeframe["nRow"]);
}
  


unique_ptr<PredictBridge> PBRf::unwrapCtg(const List& lDeframe,
                                          const List& lTrain,
                                          bool oob,
                                          bool doProb,
                                          unsigned int nThread) {
  List lRLE((SEXP) lDeframe["rleFrame"]);
  return make_unique<PredictBridge>(oob,
				    RLEFrameR::unwrap(lRLE),
                                    ForestRf::unwrap(lTrain),
                                    BagRf::unwrap(lTrain, getNRow(lDeframe), oob),
                                    LeafCtgRf::unwrap(lTrain, getNRow(lDeframe), doProb),
                                    nThread);
}


unique_ptr<PredictBridge> PBRf::unwrapReg(const List& lDeframe,
                                          const List& lTrain,
                                          bool oob,
                                          unsigned int nThread) {
  List lRLE((SEXP) lDeframe["rleFrame"]);
  return make_unique<PredictBridge>(oob,
				    RLEFrameR::unwrap(lRLE),
                                    ForestRf::unwrap(lTrain),
                                    BagRf::unwrap(lTrain, getNRow(lDeframe), oob),
                                    LeafRegRf::unwrap(lTrain, getNRow(lDeframe)),
                                    nThread);
}


unique_ptr<PredictBridge> PBRf::unwrapReg(const List& lDeframe,
                                          const List& lTrain,
                                          bool oob,
                                          unsigned int nThread,
                                          const vector<double>& quantile) {
  List lRLE((SEXP) lDeframe["rleFrame"]);
  return make_unique<PredictBridge>(oob,
				    RLEFrameR::unwrap(lRLE),
                                    ForestRf::unwrap(lTrain),
                                    BagRf::unwrap(lTrain, getNRow(lDeframe), oob),
                                    LeafRegRf::unwrap(lTrain, getNRow(lDeframe)),
                                    quantile,
                                    nThread);
}


RcppExport SEXP ValidateVotes(const SEXP sDeframe,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictCtg(List(sDeframe), List(sTrain), sYTest, true, false, as<unsigned int>(sNThread));

  END_RCPP
}


RcppExport SEXP ValidateProb(const SEXP sDeframe,
                             const SEXP sTrain,
                             SEXP sYTest,
                             SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictCtg(List(sDeframe), List(sTrain), sYTest, true, true, as<unsigned int>(sNThread));

  END_RCPP
}


RcppExport SEXP TestVotes(const SEXP sDeframe,
                          const SEXP sTrain,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictCtg(List(sDeframe), List(sTrain), sYTest, as<bool>(sOOB), false, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP TestProb(const SEXP sDeframe,
                         const SEXP sTrain,
                         SEXP sYTest,
                         SEXP sOOB,
                         SEXP sNThread) {
  BEGIN_RCPP
    return PBRf::predictCtg(List(sDeframe), List(sTrain), sYTest, as<bool>(sOOB), true, as<unsigned int>(sNThread));
  END_RCPP
}


/**
   @brief predict for classification.

   @return predict list.
 */
List PBRf::predictCtg(const List& lDeframe,
                      const List& lTrain,
                      SEXP sYTest,
                      bool oob,
                      bool doProb,
                      unsigned int nThread) {
  BEGIN_RCPP

  unique_ptr<PredictBridge> pBridge(unwrapCtg(lDeframe, lTrain, oob, doProb, nThread));
  predict(pBridge.get(), getNRow(lDeframe));

  return LeafCtgRf::summary(lDeframe, lTrain, pBridge.get(), sYTest);

  END_RCPP
}

RcppExport SEXP ValidateQuant(const SEXP sDeframe,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sQuantVec,
                              SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictQuant(List(sDeframe), sTrain, sQuantVec, sYTest, true, as<unsigned int>(sNThread));

  END_RCPP
}


 RcppExport SEXP TestQuant(const SEXP sDeframe,
                          const SEXP sTrain,
                          SEXP sQuantVec,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictQuant(List(sDeframe), sTrain, sQuantVec, sYTest, as<bool>(sOOB), as<unsigned int>(sNThread));

  END_RCPP
}


List PBRf::predictQuant(const List& lDeframe,
                        const List& lTrain,
                        SEXP sQuantVec,
                        SEXP sYTest,
                        bool oob,
                        unsigned int nThread) {
  BEGIN_RCPP

  NumericVector quantVec(sQuantVec);
  vector<double> quantile(quantVec.begin(), quantVec.end());
  unique_ptr<PredictBridge> pBridge(unwrapReg(lDeframe, lTrain, oob, nThread, quantile));
  predict(pBridge.get(), getNRow(lDeframe));

  return LeafRegRf::summary(sYTest, pBridge.get());
  
  END_RCPP
}


void PBRf::predict(PredictBridge* pBridge,
                   size_t nRow) {
  size_t row = predictBlock(pBridge, 0, nRow);
  // Remainder rows handled in custom-fitted block.
  if (nRow > row) {
    (void) predictBlock(pBridge, row, nRow);
  }
}


size_t PBRf::predictBlock(PredictBridge* pBridge,
                          size_t rowStart,
                          size_t rowEnd) {
  size_t blockRows = PredictBridge::getBlockRows(rowEnd - rowStart);
  size_t row = rowStart;
  for (; row + blockRows <= rowEnd; row += blockRows) {
    pBridge->predictBlock(row, blockRows);
  }

  return row;
}
