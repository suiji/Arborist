// Copyright (C)  2012-2016  Mark Seligman
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
   @file rcppPredblock.h

   @brief C++ class definitions for managing PredBlock object.

   @author Mark Seligman

 */


#ifndef ARBORIST_RCPP_PREDBLOCK_H
#define ARBORIST_RCPP_PREDBLOCK_H

#include <Rcpp.h>
using namespace Rcpp;

class RcppPredblock {
 public:
  static void Unwrap(SEXP sPredBlock, unsigned int &_nRow, unsigned int &_nPredNum, unsigned int &_nPredFac, NumericMatrix &_blockNum, IntegerMatrix &_blockFac);
  static void SignatureUnwrap(SEXP sSignature, IntegerVector &_predMap, List &_level);
  static void FactorRemap(IntegerMatrix &xFac, List &level, List &levelTrain);
};


#endif
