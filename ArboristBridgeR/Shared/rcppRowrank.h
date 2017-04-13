// Copyright (C)  2012-2017  Mark Seligman
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
   @file rcppRowrank.h

   @brief C++ class definitions for managing RowRank object.

   @author Mark Seligman

 */


#ifndef ARBORIST_RCPP_ROWRANK_H
#define ARBORIST_RCPP_ROWRANK_H

#include <Rcpp.h>
using namespace Rcpp;

class RcppRowrank {
  static IntegerVector iv1, iv2, iv3, iv4;
  static NumericVector nv1;


 public:
  static void Unwrap(SEXP sRowRank, unsigned int *&feNumOff, double *&feNumVal, unsigned int *&feRow, unsigned int *&feRank, unsigned int *&feRLE, unsigned int &feRLELength);
  static void Clear();
};


#endif
