// Copyright (C)  2012-2022  Mark Seligman
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

  @param sXNum is a (possibly empty numeric matrix.

  @param sXFac is a (posibly empty) zero-based integer matrix.

  @param sLevels contain the level strings of core-indexed factor predictors.

  @return wrapped frame containing separately-typed matrices.
*/
RcppExport SEXP deframeDF(SEXP sX,
			  SEXP sPredForm,
                          SEXP sLevels,
			  SEXP sFactor,
			  SEXP sSigTrain);


SEXP checkFrame(const List& lSigTrain,
		const CharacterVector& predForm);


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


RcppExport SEXP deframeIP(SEXP sX);

#endif
