// Copyright (C)  2012-2019  Mark Seligman
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

#include <algorithm>
RcppExport SEXP ValidateReg(const SEXP sFrame,
                            const SEXP sTrain,
                            SEXP sYTest,
                            SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictReg(List(sFrame), List(sTrain), sYTest, true, as<unsigned int>(sNThread));

  END_RCPP
}


RcppExport SEXP TestReg(const SEXP sFrame,
                        const SEXP sTrain,
                        SEXP sYTest,
                        SEXP sOOB,
                        SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictReg(List(sFrame), List(sTrain), sYTest, as<bool>(sOOB), as<unsigned int>(sNThread));

  END_RCPP
}

/**
   @brief Predction for regression.

   @return Wrapped zero, with copy-out parameters.
 */
List PBRf::predictReg(const List& lFrame,
                  const List& lTrain,
                  SEXP sYTest,
                  bool oob,
                  unsigned int nThread) {
  BEGIN_RCPP

  unique_ptr<PredictBridge> pBridge(unwrapReg(lFrame, lTrain, oob, nThread));
  predict(pBridge.get(), BlockBatch<NumericMatrix>::unwrap(lFrame).get(), BlockBatch<IntegerMatrix>::unwrap(lFrame).get(), getNRow(lFrame));

  return LeafRegRf::summary(sYTest, pBridge.get());
  
  END_RCPP
}


size_t PBRf::getNRow(const List& lFrame) {
  return as<size_t>((SEXP) lFrame["nRow"]);
}
  


unique_ptr<PredictBridge> PBRf::unwrapCtg(const List& lFrame,
                                          const List& lTrain,
                                          bool oob,
                                          bool doProb,
                                          unsigned int nThread) {
  checkFrame(lFrame);
  return make_unique<PredictBridge>(oob,
                                    ForestRf::unwrap(lTrain),
                                    BagRf::unwrap(lTrain, lFrame, oob),
                                    LeafCtgRf::unwrap(lTrain, lFrame, doProb),
                                    nThread);
}


SEXP PBRf::checkFrame(const List &frame) {
  BEGIN_RCPP
  if (!frame.inherits("Frame")) {
    stop("Expecting Frame");
  }

  if (!Rf_isNull(frame["blockFacRLE"])) {
    stop ("Sparse factors:  NYI");
  }
  END_RCPP
}


unique_ptr<PredictBridge> PBRf::unwrapReg(const List& lFrame,
                                          const List& lTrain,
                                          bool oob,
                                          unsigned int nThread) {
  checkFrame(lFrame);
  return make_unique<PredictBridge>(oob,
                                    ForestRf::unwrap(lTrain),
                                    BagRf::unwrap(lTrain, lFrame, oob),
                                    LeafRegRf::unwrap(lTrain, lFrame),
                                    nThread);
}


unique_ptr<PredictBridge> PBRf::unwrapReg(const List& lFrame,
                                          const List& lTrain,
                                          bool oob,
                                          unsigned int nThread,
                                          const vector<double>& quantile) {
  return make_unique<PredictBridge>(oob,
                                    ForestRf::unwrap(lTrain),
                                    BagRf::unwrap(lTrain, lFrame, oob),
                                    LeafRegRf::unwrap(lTrain, lFrame),
                                    quantile,
                                    nThread);
}


RcppExport SEXP ValidateVotes(const SEXP sFrame,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictCtg(List(sFrame), List(sTrain), sYTest, true, false, as<unsigned int>(sNThread));

  END_RCPP
}


RcppExport SEXP ValidateProb(const SEXP sFrame,
                             const SEXP sTrain,
                             SEXP sYTest,
                             SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictCtg(List(sFrame), List(sTrain), sYTest, true, true, as<unsigned int>(sNThread));

  END_RCPP
}


RcppExport SEXP TestVotes(const SEXP sFrame,
                          const SEXP sTrain,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread) {
  BEGIN_RCPP
    return PBRf::predictCtg(List(sFrame), List(sTrain), sYTest, as<bool>(sOOB), false, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP TestProb(const SEXP sFrame,
                         const SEXP sTrain,
                         SEXP sYTest,
                         SEXP sOOB,
                         SEXP sNThread) {
  BEGIN_RCPP
    return PBRf::predictCtg(List(sFrame), List(sTrain), sYTest, as<bool>(sOOB), true, as<unsigned int>(sNThread));
  END_RCPP
}


/**
   @brief predict for classification.

   @return predict list.
 */
List PBRf::predictCtg(const List& lFrame,
                      const List& lTrain,
                      SEXP sYTest,
                      bool oob,
                      bool doProb,
                      unsigned int nThread) {
  BEGIN_RCPP

  unique_ptr<PredictBridge> pBridge(unwrapCtg(lFrame, lTrain, oob, doProb, nThread));
  predict(pBridge.get(), BlockBatch<NumericMatrix>::unwrap(lFrame).get(), BlockBatch<IntegerMatrix>::unwrap(lFrame).get(), getNRow(lFrame));

  return LeafCtgRf::summary(lFrame, lTrain, pBridge.get(), sYTest);

  END_RCPP
}

RcppExport SEXP ValidateQuant(const SEXP sFrame,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sQuantVec,
                              SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictQuant(sFrame, sTrain, sQuantVec, sYTest, true, as<unsigned int>(sNThread));

  END_RCPP
}


RcppExport SEXP TestQuant(const SEXP sFrame,
                          const SEXP sTrain,
                          SEXP sQuantVec,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread) {
  BEGIN_RCPP

  return PBRf::predictQuant(sFrame, sTrain, sQuantVec, sYTest, as<bool>(sOOB), as<unsigned int>(sNThread));

  END_RCPP
}


List PBRf::predictQuant(const List& lFrame,
                        const List& lTrain,
                        SEXP sQuantVec,
                        SEXP sYTest,
                        bool oob,
                        unsigned int nThread) {
  BEGIN_RCPP

  NumericVector quantVec(sQuantVec);
  vector<double> quantile(quantVec.begin(), quantVec.end());
  unique_ptr<PredictBridge> pBridge(unwrapReg(lFrame, lTrain, oob, nThread, quantile));
  predict(pBridge.get(), BlockBatch<NumericMatrix>::unwrap(lFrame).get(), BlockBatch<IntegerMatrix>::unwrap(lFrame).get(), getNRow(lFrame));

  return LeafRegRf::summary(sYTest, pBridge.get());
  
  END_RCPP
}


void PBRf::predict(PredictBridge* pBridge,
                   BlockBatch<NumericMatrix>* blockNum,
                   BlockBatch<IntegerMatrix>* blockFac,
                   size_t nRow) {
  size_t row = predictBlock(pBridge, blockNum, blockFac, 0, nRow);
  // Remainder rows handled in custom-fitted block.
  if (nRow > row) {
    (void) predictBlock(pBridge, blockNum, blockFac, row, nRow);
  }
}


size_t PBRf::predictBlock(PredictBridge* pBridge,
                          BlockBatch<NumericMatrix>* blockNum,
                          BlockBatch<IntegerMatrix>* blockFac,
                          size_t rowStart,
                          size_t rowEnd) {
  size_t blockRows = PredictBridge::getBlockRows(rowEnd - rowStart);
  size_t row = rowStart;
  for (; row + blockRows <= rowEnd; row += blockRows) {
    NumericMatrix tpNum(blockNum->transpose(row, blockRows));
    IntegerMatrix tpFac(blockFac->transpose(row, blockRows));
    pBridge->predictBlock(BlockBatch<NumericMatrix>::coreBlock(tpNum).get(), BlockBatch<IntegerMatrix>::coreBlock(tpFac).get(), row);
  }

  return row;
}
