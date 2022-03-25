// Copyright (C)  2012-2022   Mark Seligman
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
   @file signature.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/

#include "signature.h"

List Signature::wrapNum(unsigned int nPred,
			const CharacterVector& colNames,
			const CharacterVector& rowNames) {
  BEGIN_RCPP
  return wrap(nPred, rep(CharacterVector("numeric"), nPred), List::create(0), List::create(0), colNames, rowNames);
  END_RCPP
}


List Signature::wrapFac(unsigned int nPred,
			const CharacterVector& colNames,
			const CharacterVector& rowNames) {
  BEGIN_RCPP
  return wrap(nPred, rep(CharacterVector("factor"), nPred), List::create(0), List::create(0), colNames, rowNames);
  END_RCPP
}


// Signature contains front-end decorations not exposed to the
  // core.
// Column and row names stubbed to zero-length vectors if null.
List Signature::wrap(unsigned int nPred,
		     const CharacterVector& predForm,
		     const List& level,
		     const List& factor,
		     const CharacterVector& colNames,
		     const CharacterVector& rowNames) {
  BEGIN_RCPP
  List signature =
    List::create(_["nPred"] = nPred,
		 _["predForm"] = predForm,
                 _["level"] = level,
		 _["factor"] = factor,
                 _["colNames"] = colNames,
                 _["rowNames"] = rowNames
                 );
  signature.attr("class") = "Signature";

  return signature;
  END_RCPP
}


/**
   @brief Unwraps field values useful for prediction.
 */
CharacterVector Signature::unwrapRowNames(const List& lDeframe) {
  BEGIN_RCPP
  checkFrame(lDeframe);
  List signature = checkSignature(lDeframe);

  if (Rf_isNull(signature["rowNames"])) {
    return CharacterVector(0);
  }
  else {
    return CharacterVector((SEXP) signature["rowNames"]);
  }
  END_RCPP
}


CharacterVector Signature::unwrapColNames(const List& lDeframe) {
  BEGIN_RCPP
  checkFrame(lDeframe);
  List signature = checkSignature(lDeframe);

  if (Rf_isNull(signature["colNames"])) {
    return CharacterVector(0);
  }
  else {
    return CharacterVector((SEXP) signature["colNames"]);
  }
  END_RCPP
}


SEXP Signature::checkSignature(const List &lDeframe) {
  BEGIN_RCPP
  List signature((SEXP) lDeframe["signature"]);
  if (!signature.inherits("Signature")) {
    stop("Expecting Signature");
  }

  return signature;
  END_RCPP
}


List Signature::unwrapLevel(const List& sTrain) {
 List sSignature(checkSignature(sTrain));
 return as<List>(sSignature["level"]);
}


List Signature::unwrapFactor(const List& sTrain) {
  List sSignature(checkSignature(sTrain));
  return as<List>(sSignature["factor"]);
}


void Signature::unwrapExport(const List& sTrain, List& level, List& factor, StringVector& names) {
  List sSignature(checkSignature(sTrain));
  names = as<CharacterVector>(sSignature["colNames"]);
  level = as<List>(sSignature["level"]);
  factor = as<List>(sSignature["factor"]);
}


SEXP Signature::checkFrame(const List &lDeframe) {
  BEGIN_RCPP
  if (!lDeframe.inherits("Deframe")) {
    stop("Expecting Derame");
  }

  END_RCPP
}
