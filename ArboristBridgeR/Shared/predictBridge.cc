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

RcppExport SEXP ValidateReg(const SEXP sPredBlock,
                            const SEXP sTrain,
                            SEXP sYTest,
                            SEXP sNThread) {
  BEGIN_RCPP
    return PBBridgeReg::reg(List(sPredBlock), List(sTrain), sYTest, true, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP TestReg(const SEXP sPredBlock,
                        const SEXP sTrain,
                        SEXP sYTest,
                        SEXP sOOB,
                        SEXP sNThread) {
  BEGIN_RCPP
    return PBBridgeReg::reg(List(sPredBlock), List(sTrain), sYTest, as<bool>(sOOB), as<unsigned int>(sNThread));
  END_RCPP
}

/**
   @brief Predction for regression.

   @return Wrapped zero, with copy-out parameters.
 */
List PBBridgeReg::reg(const List& sPredBlock,
                      const List& lTrain,
                      SEXP sYTest,
                      bool oob,
                      unsigned int nThread) {
  BEGIN_RCPP

    auto pbBridge = factory(sPredBlock, lTrain, oob, nThread);
  return move(pbBridge->predict(sYTest));

  END_RCPP
}


List PBBridgeReg::predict(SEXP sYTest) const {
  BEGIN_RCPP
  Predict::predict(box.get());
  return move(leaf->summary(sYTest));
  END_RCPP
}


unique_ptr<PBBridgeReg> PBBridgeReg::factory(const List& sPredBlock,
                                             const List& lTrain,
                                             bool oob,
                                             unsigned int nThread) {
  return make_unique<PBBridgeReg>(move(FramemapBridge::factoryPredict(sPredBlock)),
                                  move(ForestBridge::unwrap(lTrain)),
                                  move(BagBridge::unwrap(lTrain, sPredBlock, oob)),
                                  move(LeafRegBridge::unwrap(lTrain, sPredBlock)),
                                  nThread);
}


PBBridgeReg::PBBridgeReg(unique_ptr<FramePredictBridge> framePredict_,
                         unique_ptr<ForestBridge> forest_,
                         unique_ptr<BagBridge> bag_,
                         unique_ptr<LeafRegBridge> leaf_,
                         unsigned int nThread) :
  PBBridge(move(framePredict_),
           move(forest_),
           move(bag_)),
  leaf(move(leaf_)) {
  box = make_unique<PredictBox>(framePredict->getFrame(), forest->getForest(), bag->getRaw(), leaf->getLeaf(), nThread);
}


PBBridge::PBBridge(unique_ptr<FramePredictBridge> framePredict_,
                   unique_ptr<ForestBridge> forest_,
                   unique_ptr<BagBridge> bag_) :
  framePredict(move(framePredict_)),
  forest(move(forest_)),
  bag(move(bag_)) {
}


RcppExport SEXP ValidateVotes(const SEXP sPredBlock,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sNThread) {
  BEGIN_RCPP
    return PBBridgeCtg::ctg(List(sPredBlock), List(sTrain), sYTest, true, false, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP ValidateProb(const SEXP sPredBlock,
                             const SEXP sTrain,
                             SEXP sYTest,
                             SEXP sNThread) {
  BEGIN_RCPP
    return PBBridgeCtg::ctg(List(sPredBlock), List(sTrain), sYTest, true, true, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP TestVotes(const SEXP sPredBlock,
                          const SEXP sTrain,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread) {
  BEGIN_RCPP
    return PBBridgeCtg::ctg(List(sPredBlock), List(sTrain), sYTest, as<bool>(sOOB), false, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP TestProb(const SEXP sPredBlock,
                         const SEXP sTrain,
                         SEXP sYTest,
                         SEXP sOOB,
                         SEXP sNThread) {
  BEGIN_RCPP
    return PBBridgeCtg::ctg(List(sPredBlock), List(sTrain), sYTest, as<bool>(sOOB), true, as<unsigned int>(sNThread));
  END_RCPP
}


/**
   @brief predict for classification.

   @return predict list.
 */
List PBBridgeCtg::ctg(const List& sPredBlock,
                      const List& lTrain,
                      SEXP sYTest,
                      bool oob,
                      bool doProb,
                      unsigned int nThread) {
  BEGIN_RCPP

    auto pbBridge = factory(sPredBlock, lTrain, oob, doProb, nThread);
  return move(pbBridge->predict(sYTest, sPredBlock));

  END_RCPP
}

List PBBridgeCtg::predict(SEXP sYTest, const List& sPredBlock) const {
  BEGIN_RCPP
  Predict::predict(box.get());
  return move(leaf->summary(sYTest, sPredBlock));
  END_RCPP
}


unique_ptr<PBBridgeCtg> PBBridgeCtg::factory(const List& sPredBlock,
                                             const List& lTrain,
                                             bool oob,
                                             bool doProb,
                                             unsigned int nThread) {
  return make_unique<PBBridgeCtg>(move(FramemapBridge::factoryPredict(sPredBlock)),
                                  move(ForestBridge::unwrap(lTrain)),
                                  move(BagBridge::unwrap(lTrain, sPredBlock, oob)),
                                  move(LeafCtgBridge::unwrap(lTrain, sPredBlock, doProb)),
                                  nThread);
}


PBBridgeCtg::PBBridgeCtg(unique_ptr<FramePredictBridge> framePredict_,
                         unique_ptr<ForestBridge> forest_,
                         unique_ptr<BagBridge> bag_,
                         unique_ptr<LeafCtgBridge> leaf_,
                         unsigned int nThread) :
  PBBridge(move(framePredict_),
           move(forest_),
           move(bag_)),
  leaf(move(leaf_)) {
  box = make_unique<PredictBox>(framePredict->getFrame(), forest->getForest(), bag->getRaw(), leaf->getLeaf(), nThread);
}


RcppExport SEXP ValidateQuant(const SEXP sPredBlock,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sQuantVec,
                              SEXP sQBin,
                              SEXP sNThread) {
  BEGIN_RCPP
    return PBBridgeReg::quant(sPredBlock, sTrain, sQuantVec, sQBin, sYTest, true, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP TestQuant(const SEXP sPredBlock,
                          const SEXP sTrain,
                          SEXP sQuantVec,
                          SEXP sQBin,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread) {
  BEGIN_RCPP
    return PBBridgeReg::quant(sPredBlock, sTrain, sQuantVec, sQBin, sYTest, as<bool>(sOOB), as<unsigned int>(sNThread));
  END_RCPP
}


List PBBridgeReg::quant(const List& sPredBlock,
                        const List& lTrain,
                        SEXP sQuantVec,
                        SEXP sQBin,
                        SEXP sYTest,
                        bool oob,
                        unsigned int nThread) {
  BEGIN_RCPP

  NumericVector quantVec(sQuantVec);
  auto pbBridge = factory(sPredBlock, lTrain, oob, nThread);
  return move(pbBridge->predict(&quantVec[0], quantVec.length(), as<unsigned int>(sQBin), sYTest));

  END_RCPP
}


List PBBridgeReg::predict(const double* quantile, unsigned int nQuant, unsigned int binSize, SEXP sYTest) const {
  BEGIN_RCPP

  auto quant = Predict::predictQuant(box.get(), quantile, nQuant, binSize);
  return move(leaf->summary(sYTest, quant.get()));

  END_RCPP
}
