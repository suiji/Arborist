// Copyright (C)  2012-2019   Mark Seligman
//
// This file is part of rf.
//
// rf is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rf is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file predictRf.h

   @brief C++ interface to R entry for prediction.

   @author Mark Seligman
 */


#ifndef RF_PREDICT_RF_H
#define RF_PREDICT_RF_H

#include "blockbatch.h"


RcppExport SEXP ValidateReg(const SEXP sFrame,
                            const SEXP sTrain,
                            SEXP sYTest,
                            SEXP sNThread);

RcppExport SEXP TestReg(const SEXP sFrame,
                        const SEXP sTrain,
                        SEXP sYTest,
                        SEXP sOOB,
                        SEXP sNThread);

RcppExport SEXP ValidateVotes(const SEXP sFrame,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sNThread);

RcppExport SEXP ValidateProb(const SEXP sFrame,
                             const SEXP sTrain,
                             SEXP sYTest,
                             SEXP sNThread);

RcppExport SEXP ValidateQuant(const SEXP sFrame,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sQuantVec,
                              SEXP sNThread);

RcppExport SEXP TestQuant(const SEXP sFrame,
                          const SEXP sTrain,
                          SEXP sQuantVec,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread);

/**
   @brief Predicts with class votes.

   @param sFrame contains the blocked observations.

   @param sTrain contains the trained object.

   @param sYTest is the vector of test values.

   @param sOOB indicates whether testing is out-of-bag.

   @return wrapped predict object.
 */
RcppExport SEXP TestProb(const SEXP sFrame,
                         const SEXP sTrain,
                         SEXP sYTest,
                         SEXP sOOB,
                         SEXP sNThread);


/**
   @brief Predicts with class votes.

   @param sFrame contains the blocked observations.

   @param sTrain contains the trained object.

   @param sYTest contains the test vector.

   @param sOOB indicates whether testing is out-of-bag.

   @return wrapped predict object.
 */
RcppExport SEXP TestVotes(const SEXP sFrame,
                          const SEXP sTrain,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread);

/**
   @brief Bridge-variant PredictBridge pins unwrapped front-end structures.
 */
struct PBRf {

  static SEXP checkFrame(const List& frame);
  
  /**
     @brief Obtains the number of observations.

     @param lFrame is an R-style frame summarizing the data layout.

     @return number of rows.
   */
  static size_t getNRow(const List& lFrame);

  static List predictCtg(const List& lFrame,
                  const List& lTrain,
                  SEXP sYTest,
                  bool oob,
                  bool doProb,
                  unsigned int nThread);

  /**
     @brief Prediction for regression.  Parameters as above.
   */
  static List predictReg(const List& sFrame,
                         const List& sTrain,
                         SEXP sYTest,
                         bool oob,
                         unsigned int nThread);

 /**
    @brief Prediction with quantiles.

    @param sFrame contains the blocked observations.

    @param sTrain contains the trained object.

    @param sQuantVec is a vector of quantile training data.
   
    @param sYTest is the test vector.

    @param oob is true iff testing restricted to out-of-bag.

    @return wrapped prediction list.
 */
  static List predictQuant(const List& sFrame,
                    const List& sTrain,
                    SEXP sQuantVec,
                    SEXP sYTest,
                    bool oob,
                    unsigned int nThread);

  /**
     @brief Unwraps regression data structurs and moves to box.

     @return unique pointer to bridge-variant PredictBridge. 
   */
  static unique_ptr<struct PredictBridge> unwrapReg(const List& lFrame,
                                                   const List& lTrain,
                                                   bool oob,
                                                   unsigned int nThread,
                                                   const vector<double>& quantile);

  /**
     @brief Unwraps regression data structurs and moves to box.

     @return unique pointer to bridge-variant PredictBridge. 
   */
  static unique_ptr<struct PredictBridge> unwrapReg(const List& lFrame,
                                                   const List& lTrain,
                                                   bool oob,
                                                   unsigned int nThread);


  /**
     @brief Instantiates core prediction object and predicts quantiles.

     @return wrapped predictions.
   */
  List predict(SEXP sYTest,
               const vector<double>& quantile) const;

  /**
     @brief Unwraps regression data structurs and moves to box.

     @return unique pointer to bridge-variant PredictBridge. 
   */
  static unique_ptr<struct PredictBridge> unwrapCtg(const List& sFrame,
                                                   const List& lTrain,
                                                   bool oob,
                                                   bool doProb,
                                                   unsigned int nThread);

private:
  /**
     @brief Instantiates core prediction object and predicts means.

     @return wrapped predictions.
   */
  static List predictReg(SEXP sYTest);

  /**
     @brief Instantiates core PredictRf object, driving prediction.

     @return wrapped prediction.
   */
  static List predictCtg(SEXP sYTest, const List& lTrain, const List& sFrame);


  static void predict(struct PredictBridge* pBridge,
                      BlockBatch<NumericMatrix>* blockNum,
                      BlockBatch<IntegerMatrix>* blockFac,
                      size_t nRow);

  static size_t predictBlock(PredictBridge* pBridge,
                             BlockBatch<NumericMatrix>* blockNum,
                             BlockBatch<IntegerMatrix>* blockFac,
                             size_t rowStart,
                             size_t rowCount);
};
#endif
