// Copyright (C)  2012-2019  Mark Seligman
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
   @file predictBridge.cc

   @brief C++ interface to R entry for prediction methods.

   @author Mark Seligman
 */

#include "predictBridge.h"
#include "predict.h"
#include "quant.h"
#include "bagBridge.h"
#include "blockBridge.h"
#include "framemapBridge.h"
#include "forestBridge.h"
#include "leafBridge.h"
#include "forest.h"
#include "leaf.h"

#include <algorithm>

RcppExport SEXP ValidateReg(const SEXP sPredBlock,
                            const SEXP sTrain,
                            SEXP sYTest) {
  BEGIN_RCPP
  return PredictBridge::reg(List(sPredBlock), List(sTrain), sYTest, true);
  END_RCPP
}


RcppExport SEXP TestReg(const SEXP sPredBlock,
                        const SEXP sTrain,
                        SEXP sYTest,
                        SEXP sOOB) {
  BEGIN_RCPP
  return PredictBridge::reg(List(sPredBlock), List(sTrain), sYTest, as<bool>(sOOB));
  END_RCPP
}


/**
   @brief Predction for regression.

   @return Wrapped zero, with copy-out parameters.
 */
List PredictBridge::reg(const List& sPredBlock,
                        const List& lTrain,
                        SEXP sYTest,
                        bool validate) {
  BEGIN_RCPP

  auto frameMapBridge = FramemapBridge::factoryPredict(sPredBlock);
  auto framePredict = frameMapBridge->getFrame();
  auto forestBridge = ForestBridge::unwrap(lTrain);
  auto leafReg = LeafRegBridge::unwrap(lTrain, framePredict->getNRow());
  auto bag = BagBridge::unwrap(lTrain);

  Predict::reg(leafReg->getLeaf(), forestBridge->getForest(), bag->getRaw(), framePredict, validate);

  return move(leafReg->summary(sYTest));
  END_RCPP
}


RcppExport SEXP ValidateVotes(const SEXP sPredBlock,
                              const SEXP sTrain,
                              SEXP sYTest) {
  BEGIN_RCPP
  return PredictBridge::ctg(List(sPredBlock), List(sTrain), sYTest, true, false);
  END_RCPP
}


RcppExport SEXP ValidateProb(const SEXP sPredBlock,
                             const SEXP sTrain,
                             SEXP sYTest) {
  BEGIN_RCPP
  return PredictBridge::ctg(List(sPredBlock), List(sTrain), sYTest, true, true);
  END_RCPP
}


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
                          SEXP sOOB) {
  BEGIN_RCPP
  return PredictBridge::ctg(List(sPredBlock), List(sTrain), sYTest, as<bool>(sOOB), false);
  END_RCPP
}


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
                         SEXP sOOB) {
  BEGIN_RCPP
  return PredictBridge::ctg(List(sPredBlock), List(sTrain), sYTest, as<bool>(sOOB), true);
  END_RCPP
}


/**
   @brief predict for classification.

   @return predict list.
 */
List PredictBridge::ctg(const List& sPredBlock,
                        const List& lTrain,
                        SEXP sYTest,
                        bool validate,
                        bool doProb) {
  BEGIN_RCPP
  auto frameMapBridge = FramemapBridge::factoryPredict(sPredBlock);
  auto framePredict = frameMapBridge->getFrame();
  auto forestBridge = ForestBridge::unwrap(lTrain);
  auto leafCtg = LeafCtgBridge::unwrap(lTrain, framePredict->getNRow(), doProb);
  auto bag = BagBridge::unwrap(lTrain);

  Predict::ctg(leafCtg->getLeaf(), forestBridge->getForest(), bag->getRaw(), framePredict, validate);

  List signature = FramemapBridge::unwrapSignature(sPredBlock);
  return move(leafCtg->summary(sYTest, signature));

  END_RCPP
}


RcppExport SEXP ValidateQuant(const SEXP sPredBlock,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sQuantVec,
                              SEXP sQBin) {
  BEGIN_RCPP
  return PredictBridge::quant(sPredBlock, sTrain, sQuantVec, sQBin, sYTest, true);
  END_RCPP
}


RcppExport SEXP TestQuant(const SEXP sPredBlock,
                          const SEXP sTrain,
                          SEXP sQuantVec,
                          SEXP sQBin,
                          SEXP sYTest,
                          SEXP sOOB) {
  BEGIN_RCPP
  return PredictBridge::quant(sPredBlock, sTrain, sQuantVec, sQBin, sYTest, as<bool>(sOOB));
  END_RCPP
}


/**
   @brief predict with quantiles.

   @param sPredBlock contains the blocked observations.

   @param sTrain contains the trained object.

   @param sQuantVec is a vector of quantile training data.
   
   @param sQBin is the bin parameter.

   @param sYTest is the test vector.

   @param validate is true iff validating.

   @return predict list.
*/
List PredictBridge::quant(const List& sPredBlock,
                          const List& lTrain,
                          SEXP sQuantVec,
                          SEXP sQBin,
                          SEXP sYTest,
                          bool validate) {
  BEGIN_RCPP

  auto frameMapBridge = FramemapBridge::factoryPredict(sPredBlock);
  auto framePredict = frameMapBridge->getFrame();
  auto forestBridge = ForestBridge::unwrap(lTrain);
  auto leafReg = LeafRegBridge::unwrap(lTrain, framePredict->getNRow());

  auto bag = BagBridge::unwrap(lTrain);
  const vector<double> qVec = as<vector<double> >(NumericVector(sQuantVec));
  auto quant = make_unique<Quant>(leafReg->getLeaf(), bag->getRaw(), qVec, as<unsigned int>(sQBin));

  Predict::reg(leafReg->getLeaf(), forestBridge->getForest(), bag->getRaw(), framePredict, validate, quant.get());

  return move(leafReg->summary(sYTest, quant.get()));

  END_RCPP
}
  
