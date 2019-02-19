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
using namespace std;

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

/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sTrain contains the trained object.

   @param sYTest is the vector of test values.

   @param sOOB indicates whether testing is out-of-bag.

   @return predict object.
 */
RcppExport SEXP TestProb(const SEXP sPredBlock,
                         const SEXP sTrain,
                         SEXP sYTest,
                         SEXP sOOB);


/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sTrain contains the trained object.

   @param sYTest contains the test vector.

   @param sOOB indicates whether testing is out-of-bag.

   @return predict object.
 */
RcppExport SEXP TestVotes(const SEXP sPredBlock,
                          const SEXP sTrain,
                          SEXP sYTest,
                          SEXP sOOB);

/**
   @brief Bridge-variant PredictBox pins unwrapped front-end structures.
 */
struct PBBridge {
  unique_ptr<class FramePredictBridge> framePredict;
  unique_ptr<class ForestBridge> forest;
  unique_ptr<class BagBridge> bag;
  unique_ptr<class PredictBox> box;


  /**
     @brief Constructor.
   */
  PBBridge(unique_ptr<FramePredictBridge> framePredict_,
           unique_ptr<ForestBridge> forest_,
           unique_ptr<BagBridge> bag_);
};


struct PBBridgeReg : public PBBridge {
  unique_ptr<class LeafRegBridge> leaf;

  PBBridgeReg(unique_ptr<FramePredictBridge> framePredict_,
              unique_ptr<ForestBridge> forest_,
              unique_ptr<BagBridge> bag_,
              unique_ptr<LeafRegBridge> leaf_,
              bool validate);

 /**
    @brief Prediction with quantiles.

    @param sPredBlock contains the blocked observations.

    @param sTrain contains the trained object.

    @param sQuantVec is a vector of quantile training data.
   
    @param sQBin is the bin parameter.

    @param sYTest is the test vector.

    @param validate is true iff validating.

    @return predict list.
 */
  static List quant(const List& sPredBlock,
                    const List& sTrain,
                    SEXP sQuantVec,
                    SEXP sQBin,
                    SEXP sYTest,
                    bool validate);

  /**
     @brief Prediction for regression.  Parameters as above.
   */
  static List reg(const List& sPredBlock,
                  const List& sTrain,
                  SEXP sYTest,
                  bool validate);


  /**
     @brief Unwraps regression data structurs and moves to box.

     @return unique pointer to bridge-variant PredictBox. 
   */
  static unique_ptr<PBBridgeReg> factory(const List& sPredBlock,
                                         const List& lTrain,
                                         bool validate);

private:
  List predict(SEXP sYTest) const;
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
              bool validate);

  /**
     @brief Prediction for classification.  Paramters as above.

     @param doProb is true iff class probabilities requested.
   */
  static List ctg(const List& sPredBlock,
                  const List& sTrain,
                  SEXP sYTest,
                  bool validate,
                  bool doProb);

  /**
     @brief Unwraps regression data structurs and moves to box.

     @return unique pointer to bridge-variant PredictBox. 
   */
  static unique_ptr<PBBridgeCtg> factory(const List& sPredBlock,
                                         const List& lTrain,
                                         bool validate,
                                         bool doProb);
private:
  List predict(SEXP sYTest, const List& sPredBlock) const;
};

#endif
