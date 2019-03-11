// Copyright (C)  2012-2019   Mark Seligman
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

#include <memory>
using namespace std;

RcppExport SEXP ValidateReg(const SEXP sPredBlock,
                            const SEXP sTrain,
                            SEXP sYTest,
                            SEXP sNThread);

RcppExport SEXP TestReg(const SEXP sPredBlock,
                        const SEXP sTrain,
                        SEXP sYTest,
                        SEXP sOOB,
                        SEXP sNThread);

RcppExport SEXP ValidateVotes(const SEXP sPredBlock,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sNThread);

RcppExport SEXP ValidateProb(const SEXP sPredBlock,
                             const SEXP sTrain,
                             SEXP sYTest,
                             SEXP sNThread);

RcppExport SEXP ValidateQuant(const SEXP sPredBlock,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sQuantVec,
                              SEXP sQBin,
                              SEXP sNThread);

RcppExport SEXP TestQuant(const SEXP sPredBlock,
                          const SEXP sTrain,
                          SEXP sQuantVec,
                          SEXP sQBin,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread);

/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sTrain contains the trained object.

   @param sYTest is the vector of test values.

   @param sOOB indicates whether testing is out-of-bag.

   @return wrapped predict object.
 */
RcppExport SEXP TestProb(const SEXP sPredBlock,
                         const SEXP sTrain,
                         SEXP sYTest,
                         SEXP sOOB,
                         SEXP sNThread);


/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sTrain contains the trained object.

   @param sYTest contains the test vector.

   @param sOOB indicates whether testing is out-of-bag.

   @return wrapped predict object.
 */
RcppExport SEXP TestVotes(const SEXP sPredBlock,
                          const SEXP sTrain,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread);

/**
   @brief Bridge-variant PredictBox pins unwrapped front-end structures.
 */
struct PBBridge {
  unique_ptr<class FramePredictBridge> framePredict; // Predictor layout.
  unique_ptr<class ForestBridge> forest; // Trained forest.
  unique_ptr<class BagBridge> bag; // Bagged row indicator.
  unique_ptr<class PredictBox> box; // Core-level prediction frame.


  /**
     @brief Constructor.

     Paramter names mirror member names.
   */
  PBBridge(unique_ptr<FramePredictBridge> framePredict_,
           unique_ptr<ForestBridge> forest_,
           unique_ptr<BagBridge> bag_);
};


struct PBBridgeReg : public PBBridge {
  unique_ptr<class LeafRegBridge> leaf;

  /**
     @brief Constructor.

     Parameter names mirror member names.
   */
  PBBridgeReg(unique_ptr<FramePredictBridge> framePredict_,
              unique_ptr<ForestBridge> forest_,
              unique_ptr<BagBridge> bag_,
              unique_ptr<LeafRegBridge> leaf_,
              unsigned int nThread);

 /**
    @brief Prediction with quantiles.

    @param sPredBlock contains the blocked observations.

    @param sTrain contains the trained object.

    @param sQuantVec is a vector of quantile training data.
   
    @param sQBin is the bin parameter.

    @param sYTest is the test vector.

    @param oob is true iff testing restricted to out-of-bag.

    @return wrapped prediction list.
 */
  static List quant(const List& sPredBlock,
                    const List& sTrain,
                    SEXP sQuantVec,
                    SEXP sQBin,
                    SEXP sYTest,
                    bool oob,
                    unsigned int nThread);

  /**
     @brief Prediction for regression.  Parameters as above.
   */
  static List reg(const List& sPredBlock,
                  const List& sTrain,
                  SEXP sYTest,
                  bool oob,
                  unsigned int nThread);


  /**
     @brief Unwraps regression data structurs and moves to box.

     @return unique pointer to bridge-variant PredictBox. 
   */
  static unique_ptr<PBBridgeReg> factory(const List& sPredBlock,
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
               unsigned int binSize,
               SEXP sYTest) const;
};


struct PBBridgeCtg : public PBBridge {
  unique_ptr<class LeafCtgBridge> leaf;

  PBBridgeCtg(unique_ptr<FramePredictBridge> framePredict_,
              unique_ptr<ForestBridge> forest_,
              unique_ptr<BagBridge> bag_,
              unique_ptr<LeafCtgBridge> leaf_,
              unsigned int nThread);

  /**
     @brief Prediction for classification.  Paramters as above.

     @param doProb is true iff class probabilities requested.
   */
  static List ctg(const List& sPredBlock,
                  const List& sTrain,
                  SEXP sYTest,
                  bool oob,
                  bool doProb,
                  unsigned int nThread);

  /**
     @brief Unwraps regression data structurs and moves to box.

     @return unique pointer to bridge-variant PredictBox. 
   */
  static unique_ptr<PBBridgeCtg> factory(const List& sPredBlock,
                                         const List& lTrain,
                                         bool oob,
                                         bool doProb,
                                         unsigned int nThread);
private:
  /**
     @brief Instantiates core PredictBridge object, driving prediction.

     @return wrapped prediction.
   */
  List predict(SEXP sYTest, const List& sPredBlock) const;
};

#endif
