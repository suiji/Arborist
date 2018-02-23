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
   @file frameblockBridge.h

   @brief C++ class definitions for managing flat data frames.

   @author Mark Seligman

 */


#ifndef ARBORIST_FRAMEBLOCK_BRIDGE_H
#define ARBORIST_FRAMEBLOCK_BRIDGE_H

#include <Rcpp.h>
using namespace Rcpp;

#include "frameblock.h"

RcppExport SEXP FrameMixed(SEXP sX,
				 SEXP sNumElt,
				 SEXP sFacElt,
				 SEXP sLevels,
				 SEXP sSigTrain);
RcppExport SEXP FrameNum(SEXP sX);
RcppExport SEXP FrameSparse(SEXP sX);


class FrameblockBridge {
  static IntegerMatrix facT;
  static NumericMatrix numT;
 public:

  static void Clear();
  static void SparseIP(const NumericVector &eltsNZ,
		       const IntegerVector &i,
		       const IntegerVector &p,
		       unsigned int nRow,
		       vector<double> &valNum,
		       vector<unsigned int> &rowStart,
		       vector<unsigned int> &runLength,
		       vector<unsigned int> &predStart);

  static void SparseJP(NumericVector &eltsNZ,
		       IntegerVector &j,
		       IntegerVector &p,
		       unsigned int nRow,
		       vector<double> &valNum,
		       vector<unsigned int> &rowStart,
		       vector<unsigned int> &runLength);

  static void SparseIJ(NumericVector &eltsNZ,
		       IntegerVector &i,
		       IntegerVector &j,
		       unsigned int nRow,
		       vector<double> &valNum,
		       vector<unsigned int> &rowStart,
		       vector<unsigned int> &runLength);
  static class FramePredict *Unwrap(SEXP sPredBlock);

  static SEXP PredblockLegal(const List &predBlock);

  static SEXP SignatureLegal(const List &signature);

  static void SignatureUnwrap(SEXP sSignature,
			      IntegerVector &_predMap,
			      List &_level);

  static void FactorRemap(IntegerMatrix &xFac,
			  List &level,
			  List &levelTrain);

  /**
     @brief Singleton factory.

     @return allocated predictor map for training.
   */
  static FrameTrain *FactoryTrain(const vector<unsigned int> &facCard,
			unsigned int nPred,
			unsigned int nRow);
};


#endif
