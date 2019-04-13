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
   @file framemapBridge.h

   @brief C++ class definitions for managing flat data frames.

   @author Mark Seligman

 */


#ifndef ARBORIST_FRAMEMAP_BRIDGE_H
#define ARBORIST_FRAMEMAP_BRIDGE_H

#include <Rcpp.h>
using namespace Rcpp;

#include "block.h"
#include "framemap.h"


/**
  @brief Maps factor encodings of current observation set to those of training.

  Employs proxy values for any levels unseen during training.

  @param sXFac is a (posibly empty) zero-based integer matrix.

  @param sPredMap associates (zero-based) core and front-end predictors.

  @param sLevels contain the level strings of core-indexed factor predictors.

  @param sSigTrain holds the training signature.

  @return rewritten factor matrix.
*/
RcppExport SEXP FrameReconcile(SEXP sXFac,
                               SEXP sPredMap,
                               SEXP sLevels,
                               SEXP sSigTrain);


/**
  @brief Wraps frame components supplied by front end.

  @param sX is the raw data frame, with columns either factor or numeric.

  @param sXNum is a (possibly empty numeric matrix.

  @param sXFac is a (posibly empty) zero-based integer matrix.

  @param sPredMap associates (zero-based) core and front-end predictors.

  @param sFacCard is the cardinality of core-indexed factor predictors.

  @param sLevels contain the level strings of core-indexed factor predictors.

  @return wrapped frame containing separately-typed matrices.
*/
RcppExport SEXP WrapFrame(SEXP sX,
                          SEXP sXNum,
                          SEXP sXFac,
                          SEXP sPredMap,
                          SEXP sFacCard,
                          SEXP sLevels);

RcppExport SEXP FrameNum(SEXP sX);

RcppExport SEXP FrameSparse(SEXP sX);

/**
   @brief Captures ownership of FramePredict and component Blocks.
 */
class FramePredictBridge {
  const unique_ptr<class BlockNumBridge> blockNum;
  const unique_ptr<class BlockFacBridge> blockFac;
  const unsigned int nRow;
  const unique_ptr<FramePredict> framePredict;
 public:

  FramePredictBridge(unique_ptr<class BlockNumBridge> blockNum_,
                     unique_ptr<class BlockFacBridge> blockFac_,
                     unsigned int nRow);

  /**
     @brief Getter for core object pointer.
   */
  const auto getFrame() const {
    return framePredict.get();
  }
};


struct FramemapBridge {

  /**
     @brief Pulls signature member from a PredBlock object.

     @param sPredBlock contains the parent PredBlock.

     @return member of type Signature.
   */
  static List unwrapSignature(const List& sPredBlock);

  /**
     @brief Ensures the passed object has PredBlock type.

     @param predBlock is the object to be checked.
   */
  static SEXP checkPredblock(const List& predBlock);


  /**
     @brief Ensures passed object contains member of class Signature.

     @param sParent is the parent object.

     @return signature object. 
   */
  static SEXP checkSignature(const List& sParent);

  
  /**
     @brief Unwraps field values useful for export.

     @param[out] predMap outputs the core predictor mapping.

     @param[out] level outputs the training factor levels.
   */
  static void signatureUnwrap(const List& sTrain,
                              IntegerVector& predMap,
                              List& level);

  static SEXP wrapSignature(const IntegerVector& predMap,
                 const List& level,
                 const CharacterVector& colNames,
                 const CharacterVector& rowNames);

  /**
     @brief Singleton factory.

     @return allocated predictor map for training.
   */
  static unique_ptr<FrameTrain> factoryTrain(
                     const vector<unsigned int>& facCard,
                     unsigned int nPred,
                     unsigned int nRow);

  static unique_ptr<FramePredictBridge> factoryPredict(const List& sPredBlock);
};

#endif
