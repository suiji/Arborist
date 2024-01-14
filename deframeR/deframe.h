// Copyright (C)  2012-2024  Mark Seligman
//
// This file is part of deframeR.
//
// deframeR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// deframeR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with deframeR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file deframe.h

   @brief C++ class definitions for managing flat data frames.

   @author Mark Seligman

 */


#ifndef DEFRAMER_DEFRAME_H
#define DEFRAMER_DEFRAME_H


#include <Rcpp.h>
using namespace Rcpp;


/**
  @brief Wraps frame components supplied by front end.

  @param sX is a data frame.

  @param sLevels holds level strings, by predictor.

  @param sFactor holds factor values, by predictor,

  @param sSigTrain is the training signature, if any.

  @return wrapped frame containing separately-typed matrices.
*/
RcppExport SEXP deframeDF(SEXP sX,
			  SEXP sIsFactor,
                          SEXP sLevels,
			  SEXP sFactor,
			  SEXP sSigTrain);


/**
   @brief Encodes a factor-valued matrix into internal RLE format.

   @param sX is the matrix.
 */
RcppExport SEXP deframeFac(SEXP sX);


/**
   @brief Encodes a numeric-valued matrix into internal RLE format.

   @param sX is the matrix.
 */
RcppExport SEXP deframeNum(SEXP sX);


/**
   @brief Encodes a sparse matrix compressed using 'I', 'P' indices.
 */
RcppExport SEXP deframeIP(SEXP sX);

#endif
