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
   @file framemapBridge.h

   @brief C++ class definitions for managing flat data frames.

   @author Mark Seligman

 */


#ifndef ARBORIST_FRAMEMAP_BRIDGE_H
#define ARBORIST_FRAMEMAP_BRIDGE_H

#include <Rcpp.h>
using namespace Rcpp;

#include "framemap.h"

RcppExport SEXP FrameMixed(SEXP sX,
                                 SEXP sNumElt,
                                 SEXP sFacElt,
                                 SEXP sLevels,
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

  static void SparseIP(const NumericVector& eltsNZ,
                       const IntegerVector& i,
                       const IntegerVector& p,
                       unsigned int nRow,
                       vector<double>& valNum,
                       vector<unsigned int>& rowStart,
                       vector<unsigned int>& runLength,
                       vector<unsigned int>& predStart);

  static SEXP SparseJP(NumericVector& eltsNZ,
                       IntegerVector& j,
                       IntegerVector& p,
                       unsigned int nRow,
                       vector<double>& valNum,
                       vector<unsigned int>& rowStart,
                       vector<unsigned int>& runLength);

  static SEXP SparseIJ(NumericVector& eltsNZ,
                       IntegerVector& i,
                       IntegerVector& j,
                       unsigned int nRow,
                       vector<double>& valNum,
                       vector<unsigned int>& rowStart,
                       vector<unsigned int>& runLength);

  static List UnwrapSignature(const List& sPredBlock);

  static SEXP Unwrap(const List& sPredBlock);

  static SEXP PredblockLegal(const List& predBlock);

  static SEXP SignatureLegal(const List& signature);

  static void SignatureUnwrap(const List& sTrain,
                              IntegerVector& _predMap,
                              List& _level);
  static List WrapSignature(const IntegerVector& predMap,
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
  static unique_ptr<FrameTrain> FactoryTrain(
                     const vector<unsigned int>& facCard,
                     unsigned int nPred,
                     unsigned int nRow);

  static unique_ptr<FramePredictBridge> FactoryPredict(const List& sPredBlock);
};

#endif
