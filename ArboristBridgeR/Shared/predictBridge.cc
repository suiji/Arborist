// Copyright (C)  2012-2018  Mark Seligman
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

#include "blockBridge.h"
#include "framemapBridge.h"
#include "forestBridge.h"
#include "leafBridge.h"
#include "forest.h"
#include "leaf.h"

#include <algorithm>

RcppExport SEXP ValidateReg(SEXP sPredBlock,
				  SEXP sForest,
				  SEXP sLeaf,
				  SEXP sYTest) {
  return PredictBridge::Reg(sPredBlock, sForest, sLeaf, sYTest, true);
}


RcppExport SEXP TestReg(SEXP sPredBlock,
			      SEXP sForest,
			      SEXP sLeaf,
			      SEXP sYTest) {
  return PredictBridge::Reg(sPredBlock, sForest, sLeaf, sYTest, false);
}


/**
   @brief Predction for regression.

   @return Wrapped zero, with copy-out parameters.
 */
List PredictBridge::Reg(SEXP sPredBlock,
			SEXP sForest,
			SEXP sLeaf,
			SEXP sYTest,
			bool validate) {
  BEGIN_RCPP
  
  auto frameMapBridge = FramemapBridge::FactoryPredict(sPredBlock);
  auto framePredict = frameMapBridge->GetFrame();
  
  auto forestBridge = ForestBridge::Unwrap(sForest);
  auto predict = make_unique<Predict>(framePredict, forestBridge->GetForest(),
				      validate);

  return move(LeafRegBridge::Prediction(List(sLeaf), sYTest, predict.get()));
  END_RCPP
}


RcppExport SEXP ValidateVotes(SEXP sPredBlock,
				    SEXP sForest,
				    SEXP sLeaf,
				    SEXP sYTest) {
  return PredictBridge::Ctg(sPredBlock, sForest, sLeaf, sYTest, true, false);
}


RcppExport SEXP ValidateProb(SEXP sPredBlock,
				   SEXP sForest,
				   SEXP sLeaf,
				   SEXP sYTest) {
  return PredictBridge::Ctg(sPredBlock, sForest, sLeaf, sYTest, true, true);
}


/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sForest contains the trained forest.

   @param sLeaf contains the trained leaves.

   @param sVotes outputs the vote predictions.

   @return Prediction object.
 */
RcppExport SEXP TestVotes(SEXP sPredBlock,
				SEXP sForest,
				SEXP sLeaf,
				SEXP sYTest) {
  return PredictBridge::Ctg(sPredBlock, sForest, sLeaf, sYTest, false, false);
}


/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sForest contains the trained forest.

   @param sLeaf contains the trained leaves.

   @param sVotes outputs the vote predictions.

   @return Prediction object.
 */
RcppExport SEXP TestProb(SEXP sPredBlock,
			       SEXP sForest,
			       SEXP sLeaf,
			       SEXP sYTest) {
  return PredictBridge::Ctg(sPredBlock, sForest, sLeaf, sYTest, false, true);
}


/**
   @brief Prediction for classification.

   @return Prediction list.
 */
List PredictBridge::Ctg(SEXP sPredBlock,
			SEXP sForest,
			SEXP sLeaf,
			SEXP sYTest,
			bool validate,
			bool doProb) {
  BEGIN_RCPP
  auto frameMapBridge = FramemapBridge::FactoryPredict(sPredBlock);
  auto framePredict = frameMapBridge->GetFrame();
  auto forestBridge = ForestBridge::Unwrap(sForest);
  auto predict = make_unique<Predict>(framePredict, forestBridge->GetForest(),
				      validate);
  List signature;
  List predBlock = FramemapBridge::Unwrap(sPredBlock, signature);

  return move(LeafCtgBridge::Prediction(List(sLeaf), sYTest, signature, predict.get(), doProb));
  END_RCPP
}


RcppExport SEXP ValidateQuant(SEXP sPredBlock,
				    SEXP sForest,
				    SEXP sLeaf,
				    SEXP sYTest,
				    SEXP sQuantVec,
				    SEXP sQBin) {
  return PredictBridge::Quant(sPredBlock, sForest, sLeaf, sQuantVec, sQBin, sYTest, true);
}


RcppExport SEXP TestQuant(SEXP sPredBlock,
				SEXP sForest,
				SEXP sLeaf,
				SEXP sQuantVec,
				SEXP sQBin,
				SEXP sYTest) {
  return PredictBridge::Quant(sPredBlock, sForest, sLeaf, sQuantVec, sQBin, sYTest, false);
}


/**
   @brief Prediction with quantiles.

   @param sPredBlock contains the blocked observations.

   @param sForest contains the trained forest.

   @param sLeaf contains the trained leaves.

   @param sVotes outputs the vote predictions.

   @param sQuantVec is a vector of quantile training data.
   
   @param sQBin is the bin parameter.

   @param sYTest is the test vector.

   @param bag is true iff validating.

   @return Prediction list.
*/
List PredictBridge::Quant(SEXP sPredBlock,
			  SEXP sForest,
			  SEXP sLeaf,
			  SEXP sQuantVec,
			  SEXP sQBin,
			  SEXP sYTest,
			  bool validate) {
  BEGIN_RCPP

  auto frameMapBridge = FramemapBridge::FactoryPredict(sPredBlock);
  auto framePredict = frameMapBridge->GetFrame();
  auto forestBridge = ForestBridge::Unwrap(sForest);
  auto predict = make_unique<Predict>(framePredict, forestBridge->GetForest(),
				      validate);

  return move(LeafRegBridge::Prediction(List(sLeaf), sYTest, predict.get(), NumericVector(sQuantVec), as<unsigned int>(sQBin)));
  
  END_RCPP
}
