// Copyright (C)  2012-2018   Mark Seligman
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
   @file predictBridge.h

   @brief C++ interface to R entry for prediction.

   @author Mark Seligman
 */


#ifndef ARBORIST_PREDICT_BRIDGE_H
#define ARBORIST_PREDICT_BRIDGE_H

#include <Rcpp.h>
using namespace Rcpp;

RcppExport SEXP ValidateReg(const SEXP sPredBlock,
                            const SEXP sTrain,
                            SEXP sYTest);

RcppExport SEXP TestReg(const SEXP sPredBlock,
                        const SEXP sTrain,
                        SEXP sYTest,
                        SEXP sOOB);

RcppExport SEXP ValidateVotes(const SEXP sPredBlock,
                              const SEXP sTrain,
                              SEXP sYTest);

RcppExport SEXP ValidateProb(const SEXP sPredBlock,
                             const SEXP sTrain,
                             SEXP sYTest);

RcppExport SEXP ValidateQuant(const SEXP sPredBlock,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sQuantVec,
                              SEXP sQBin);

RcppExport SEXP TestQuant(const SEXP sPredBlock,
                          const SEXP sTrain,
                          SEXP sQuantVec,
                          SEXP sQBin,
                          SEXP sYTest,
                          SEXP sOOB);

RcppExport SEXP TestProb(const SEXP sPredBlock,
                         const SEXP sTrain,
                         SEXP sYTest,
                         SEXP sOOB);


/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sTrain contains the training summary.

   @param sVotes outputs the vote predictions.

   @return Prediction object.
 */
RcppExport SEXP TestVotes(const SEXP sPredBlock,
                          const SEXP sTrain,
                          SEXP sYTest,
                          SEXP sOOB);


namespace PredictBridge {
  static List reg(const List& sPredBlock,
                  const List& sTrain,
                  SEXP sYTest,
                  bool validate);


  static List ctg(const List& sPredBlock,
                  const List& sTrain,
                  SEXP sYTest,
                  bool validate,
                  bool doProb);


  static List quant(const List& sPredBlock,
                    const List& sTrain,
                    SEXP sQuantVec,
                    SEXP sQBin,
                    SEXP sYTest,
                    bool validate);
};

#endif
