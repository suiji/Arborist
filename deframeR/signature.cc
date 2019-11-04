// Copyright (C)  2012-2019   Mark Seligman
//
// This file is part of framemapR.
//
// framemapR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// framemapR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with framemapR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file signature.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/

#include "signature.h"


// Signature contains front-end decorations not exposed to the
  // core.
// Column and row names stubbed to zero-length vectors if null.
SEXP Signature::wrapSignature(const IntegerVector &predMap,
                                const List &level,
                                const CharacterVector &colNames,
                                const CharacterVector &rowNames) {
  BEGIN_RCPP
  List signature =
    List::create(
                 _["predMap"] = predMap,
                 _["level"] = level,
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
CharacterVector Signature::unwrapRowNames(const List& sFrame) {
  BEGIN_RCPP
  checkFrame(sFrame);
  List signature = checkSignature(sFrame);

  if (Rf_isNull(signature["rowNames"])) {
    return CharacterVector(0);
  }
  else {
    return CharacterVector((SEXP) signature["rowNames"]);
  }
  END_RCPP
}


SEXP Signature::checkSignature(const List &sParent) {
  BEGIN_RCPP
  List signature((SEXP) sParent["signature"]);
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


void Signature::unwrapExport(const List& sTrain, IntegerVector &predMap, List &level) {
  List sSignature(checkSignature(sTrain));

  predMap = as<IntegerVector>(sSignature["predMap"]);
  level = as<List>(sSignature["level"]);
}


SEXP Signature::checkFrame(const List &frame) {
  BEGIN_RCPP
  if (!frame.inherits("Frame")) {
    stop("Expecting Frame");
  }

  if (!Rf_isNull(frame["blockFacRLE"])) {
    stop ("Sparse factors:  NYI");
  }
  END_RCPP
}
