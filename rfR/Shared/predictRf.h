// Copyright (C)  2012-2019   Mark Seligman
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
   @file predictRf.h

   @brief C++ interface to R entry for prediction.

   @author Mark Seligman
 */


#ifndef ARBORIST_PREDICT_RF_H
#define ARBORIST_PREDICT_RF_H

#include "predict.h"

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>
using namespace std;

RcppExport SEXP ValidateReg(const SEXP sPredFrame,
                            const SEXP sTrain,
                            SEXP sYTest,
                            SEXP sNThread);

RcppExport SEXP TestReg(const SEXP sPredFrame,
                        const SEXP sTrain,
                        SEXP sYTest,
                        SEXP sOOB,
                        SEXP sNThread);

RcppExport SEXP ValidateVotes(const SEXP sPredFrame,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sNThread);

RcppExport SEXP ValidateProb(const SEXP sPredFrame,
                             const SEXP sTrain,
                             SEXP sYTest,
                             SEXP sNThread);

RcppExport SEXP ValidateQuant(const SEXP sPredFrame,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sQuantVec,
                              SEXP sNThread);

RcppExport SEXP TestQuant(const SEXP sPredFrame,
                          const SEXP sTrain,
                          SEXP sQuantVec,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread);

/**
   @brief Predicts with class votes.

   @param sPredFrame contains the blocked observations.

   @param sTrain contains the trained object.

   @param sYTest is the vector of test values.

   @param sOOB indicates whether testing is out-of-bag.

   @return wrapped predict object.
 */
RcppExport SEXP TestProb(const SEXP sPredFrame,
                         const SEXP sTrain,
                         SEXP sYTest,
                         SEXP sOOB,
                         SEXP sNThread);


/**
   @brief Predicts with class votes.

   @param sPredFrame contains the blocked observations.

   @param sTrain contains the trained object.

   @param sYTest contains the test vector.

   @param sOOB indicates whether testing is out-of-bag.

   @return wrapped predict object.
 */
RcppExport SEXP TestVotes(const SEXP sPredFrame,
                          const SEXP sTrain,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread);

/**
   @brief Bridge-variant PredictBox pins unwrapped front-end structures.
 */
struct PBRf {
  unique_ptr<class BlockFrameR> blockFrame; // Predictor layout.
  unique_ptr<class ForestRf> forest; // Trained forest.
  unique_ptr<class BagRf> bag; // Bagged row indicator.
  unique_ptr<struct PredictBox> box; // Core-level prediction frame.


  /**
     @brief Constructor.

     Paramter names mirror member names.
   */
  PBRf(unique_ptr<BlockFrameR> blockFrame_,
           unique_ptr<ForestRf> forest_,
           unique_ptr<BagRf> bag_);
};


struct PBRfReg : public PBRf {
  unique_ptr<class LeafRegRf> leaf;

  /**
     @brief Constructor.

     Parameter names mirror member names.
   */
  PBRfReg(unique_ptr<BlockFrameR> blockFrame_,
              unique_ptr<ForestRf> forest_,
              unique_ptr<BagRf> bag_,
              unique_ptr<LeafRegRf> leaf_,
              bool oob,
              unsigned int nThread);

 /**
    @brief Prediction with quantiles.

    @param sPredFrame contains the blocked observations.

    @param sTrain contains the trained object.

    @param sQuantVec is a vector of quantile training data.
   
    @param sYTest is the test vector.

    @param oob is true iff testing restricted to out-of-bag.

    @return wrapped prediction list.
 */
  static List quant(const List& sPredFrame,
                    const List& sTrain,
                    SEXP sQuantVec,
                    SEXP sYTest,
                    bool oob,
                    unsigned int nThread);

  /**
     @brief Prediction for regression.  Parameters as above.
   */
  static List reg(const List& sPredFrame,
                  const List& sTrain,
                  SEXP sYTest,
                  bool oob,
                  unsigned int nThread);


  /**
     @brief Unwraps regression data structurs and moves to box.

     @return unique pointer to bridge-variant PredictBox. 
   */
  static unique_ptr<PBRfReg> factory(const List& sPredFrame,
                                         const List& lTrain,
                                         bool oob,
                                         unsigned int nThread);

private:
  /**
     @brief Instantiates core prediction object and predicts means.

     @return wrapped predictions.
   */
  List predict(SEXP sYTest) const;

  /**
     @brief Instantiates core prediction object and predicts quantiles.

     @return wrapped predictions.
   */
  List predict(const double* quantile,
               unsigned int nQuant,
               SEXP sYTest) const;
};


struct PBRfCtg : public PBRf {
  unique_ptr<class LeafCtgRf> leaf;

  PBRfCtg(unique_ptr<BlockFrameR> blockFrame_,
              unique_ptr<ForestRf> forest_,
              unique_ptr<BagRf> bag_,
              unique_ptr<LeafCtgRf> leaf_,
              bool oob,
              unsigned int nThread);

  /**
     @brief Prediction for classification.  Paramters as above.

     @param doProb is true iff class probabilities requested.
   */
  static List ctg(const List& sPredFrame,
                  const List& sTrain,
                  SEXP sYTest,
                  bool oob,
                  bool doProb,
                  unsigned int nThread);

  /**
     @brief Unwraps regression data structurs and moves to box.

     @return unique pointer to bridge-variant PredictBox. 
   */
  static unique_ptr<PBRfCtg> factory(const List& sPredFrame,
                                         const List& lTrain,
                                         bool oob,
                                         bool doProb,
                                         unsigned int nThread);
private:
  /**
     @brief Instantiates core PredictRf object, driving prediction.

     @return wrapped prediction.
   */
  List predict(SEXP sYTest, const List& sPredFrame) const;
};

#endif
