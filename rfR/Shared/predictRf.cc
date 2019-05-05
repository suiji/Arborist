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

#include "predictRf.h"
#include "bagRf.h"
#include "blockRf.h"
#include "forestRf.h"
#include "leafRf.h"
#include "quant.h"

RcppExport SEXP ValidateReg(const SEXP sPredBlock,
                            const SEXP sTrain,
                            SEXP sYTest,
                            SEXP sNThread) {
  BEGIN_RCPP
    return PBRfReg::reg(List(sPredBlock), List(sTrain), sYTest, true, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP TestReg(const SEXP sPredBlock,
                        const SEXP sTrain,
                        SEXP sYTest,
                        SEXP sOOB,
                        SEXP sNThread) {
  BEGIN_RCPP
    return PBRfReg::reg(List(sPredBlock), List(sTrain), sYTest, as<bool>(sOOB), as<unsigned int>(sNThread));
  END_RCPP
}

/**
   @brief Predction for regression.

   @return Wrapped zero, with copy-out parameters.
 */
List PBRfReg::reg(const List& sPredBlock,
                      const List& lTrain,
                      SEXP sYTest,
                      bool oob,
                      unsigned int nThread) {
  BEGIN_RCPP

    auto pbRf = factory(sPredBlock, lTrain, oob, nThread);
  return pbRf->predict(sYTest);

  END_RCPP
}


List PBRfReg::predict(SEXP sYTest) const {
  BEGIN_RCPP
  Predict::predict(box.get());
  return leaf->summary(sYTest);
  END_RCPP
}


unique_ptr<PBRfReg> PBRfReg::factory(const List& sPredBlock,
                                             const List& lTrain,
                                             bool oob,
                                             unsigned int nThread) {
  return make_unique<PBRfReg>(BlockSetRf::factory(sPredBlock),
                              ForestRf::unwrap(lTrain),
                              BagRf::unwrap(lTrain, sPredBlock, oob),
                              LeafRegRf::unwrap(lTrain, sPredBlock),
                              oob,
                              nThread);
}


PBRfReg::PBRfReg(unique_ptr<BlockSetRf> blockSet_,
                         unique_ptr<ForestRf> forest_,
                         unique_ptr<BagRf> bag_,
                         unique_ptr<LeafRegRf> leaf_,
                         bool oob,
                         unsigned int nThread) :
  PBRf(move(blockSet_),
           move(forest_),
           move(bag_)),
  leaf(move(leaf_)) {
  box = make_unique<PredictBox>(oob, blockSet->getSet(), forest->getForest(), bag->getRaw(), leaf->getLeaf(), nThread);
}


PBRf::PBRf(unique_ptr<BlockSetRf> blockSet_,
                   unique_ptr<ForestRf> forest_,
                   unique_ptr<BagRf> bag_) :
  blockSet(move(blockSet_)),
  forest(move(forest_)),
  bag(move(bag_)) {
}


RcppExport SEXP ValidateVotes(const SEXP sPredBlock,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sNThread) {
  BEGIN_RCPP
    return PBRfCtg::ctg(List(sPredBlock), List(sTrain), sYTest, true, false, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP ValidateProb(const SEXP sPredBlock,
                             const SEXP sTrain,
                             SEXP sYTest,
                             SEXP sNThread) {
  BEGIN_RCPP
    return PBRfCtg::ctg(List(sPredBlock), List(sTrain), sYTest, true, true, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP TestVotes(const SEXP sPredBlock,
                          const SEXP sTrain,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread) {
  BEGIN_RCPP
    return PBRfCtg::ctg(List(sPredBlock), List(sTrain), sYTest, as<bool>(sOOB), false, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP TestProb(const SEXP sPredBlock,
                         const SEXP sTrain,
                         SEXP sYTest,
                         SEXP sOOB,
                         SEXP sNThread) {
  BEGIN_RCPP
    return PBRfCtg::ctg(List(sPredBlock), List(sTrain), sYTest, as<bool>(sOOB), true, as<unsigned int>(sNThread));
  END_RCPP
}


/**
   @brief predict for classification.

   @return predict list.
 */
List PBRfCtg::ctg(const List& sPredBlock,
                      const List& lTrain,
                      SEXP sYTest,
                      bool oob,
                      bool doProb,
                      unsigned int nThread) {
  BEGIN_RCPP

    auto pbRf = factory(sPredBlock, lTrain, oob, doProb, nThread);
  return pbRf->predict(sYTest, sPredBlock);

  END_RCPP
}

List PBRfCtg::predict(SEXP sYTest, const List& sPredBlock) const {
  BEGIN_RCPP
  Predict::predict(box.get());
  return leaf->summary(sYTest, sPredBlock);
  END_RCPP
}


unique_ptr<PBRfCtg> PBRfCtg::factory(const List& sPredBlock,
                                     const List& lTrain,
                                     bool oob,
                                     bool doProb,
                                     unsigned int nThread) {
  return make_unique<PBRfCtg>(BlockSetRf::factory(sPredBlock),
                              ForestRf::unwrap(lTrain),
                              BagRf::unwrap(lTrain, sPredBlock, oob),
                              LeafCtgRf::unwrap(lTrain, sPredBlock, doProb),
                              oob,
                              nThread);
}


PBRfCtg::PBRfCtg(unique_ptr<BlockSetRf> blockSet_,
                 unique_ptr<ForestRf> forest_,
                 unique_ptr<BagRf> bag_,
                 unique_ptr<LeafCtgRf> leaf_,
                 bool oob,
                 unsigned int nThread) :
  PBRf(move(blockSet_),
       move(forest_),
       move(bag_)),
  leaf(move(leaf_)) {
  box = make_unique<PredictBox>(oob, blockSet->getSet(), forest->getForest(), bag->getRaw(), leaf->getLeaf(), nThread);
}


RcppExport SEXP ValidateQuant(const SEXP sPredBlock,
                              const SEXP sTrain,
                              SEXP sYTest,
                              SEXP sQuantVec,
                              SEXP sNThread) {
  BEGIN_RCPP
    return PBRfReg::quant(sPredBlock, sTrain, sQuantVec, sYTest, true, as<unsigned int>(sNThread));
  END_RCPP
}


RcppExport SEXP TestQuant(const SEXP sPredBlock,
                          const SEXP sTrain,
                          SEXP sQuantVec,
                          SEXP sYTest,
                          SEXP sOOB,
                          SEXP sNThread) {
  BEGIN_RCPP
    return PBRfReg::quant(sPredBlock, sTrain, sQuantVec, sYTest, as<bool>(sOOB), as<unsigned int>(sNThread));
  END_RCPP
}


List PBRfReg::quant(const List& sPredBlock,
                        const List& lTrain,
                        SEXP sQuantVec,
                        SEXP sYTest,
                        bool oob,
                        unsigned int nThread) {
  BEGIN_RCPP
  NumericVector quantVec(sQuantVec);
  auto pbRf = factory(sPredBlock, lTrain, oob, nThread);
  return pbRf->predict(&quantVec[0], quantVec.length(), sYTest);
  END_RCPP
}


// TODO:  Quantile object always gets a bag.  Prediction box may or may not.
List PBRfReg::predict(const double* quantile, unsigned int nQuant, SEXP sYTest) const {
  BEGIN_RCPP

  auto quant = make_unique<Quant>(box.get(), quantile, nQuant);
  Predict::predict(box.get(), quant.get());
  return leaf->summary(sYTest, quant.get());

  END_RCPP
}
