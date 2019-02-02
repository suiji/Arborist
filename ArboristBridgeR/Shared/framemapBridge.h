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
  @brief Extracts contents of a data frame into separate numeric- and integer-valued blocks.

  Potentially slow for large predictor counts, as a linked list is being walked.

  @param sX is the raw data frame, with columns either factor or numeric.

  @param sNPredNum is the number of numeric colums.

  @param sNPredFac is the number of factor-valued columns.

  @param sCardFac is the cardinality of a factor, otherwise zero.

  @param sSigTrain holds the training signature.

  @return wrapped frame containing separately-typed matrices.
*/
RcppExport SEXP FrameMixed(SEXP sX,
                           SEXP sNPredNum,
                           SEXP sNPredFac,
                           SEXP sCardFac,
                           SEXP sSigTrain);

RcppExport SEXP FrameNum(SEXP sX);

RcppExport SEXP FrameSparse(SEXP sX);

/**
   @brief Captures ownership of FramePredict and component Blocks.
 */
class FramePredictBridge {
  unique_ptr<class BlockNumBridge> blockNum;
  unique_ptr<class BlockFacBridge> blockFac;
  unique_ptr<FramePredict> framePredict;
 public:


  FramePredictBridge(unique_ptr<class BlockNumBridge> _blockNum,
                     unique_ptr<class BlockFacBridge> _blockFac,
                       unsigned int nRow);

  
  const FramePredict *getFrame() const {
    return framePredict.get();
  }
};


struct FramemapBridge {

  static void sparseIP(const NumericVector& eltsNZ,
                       const IntegerVector& i,
                       const IntegerVector& p,
                       unsigned int nRow,
                       vector<double>& valNum,
                       vector<unsigned int>& rowStart,
                       vector<unsigned int>& runLength,
                       vector<unsigned int>& predStart);

  static List unwrapSignature(const List& sPredBlock);

  static SEXP unwrap(const List& sPredBlock);

  static SEXP PredblockLegal(const List& predBlock);

  static SEXP SignatureLegal(const List& signature);

  static void SignatureUnwrap(const List& sTrain,
                              IntegerVector& _predMap,
                              List& _level);
  static SEXP wrapSignature(const IntegerVector& predMap,
                 const List& level,
                 const CharacterVector& colNames,
                 const CharacterVector& rowNames);

  static void FactorRemap(IntegerMatrix& xFac,
                          List& level,
                          List& levelTrain);

  
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
