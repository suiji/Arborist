// Copyright (C)  2012-2015   Mark Seligman
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
   @file rcppResponse.cc

   @brief C++ interface to R entry for response.

   @author Mark Seligman
 */

#include <R.h>
#include <Rcpp.h>

//#include <iostream>
using namespace std;
using namespace Rcpp;

#include "response.h"

/**
   @brief Dispatches factories for Response class, according to response type.
   Front end guarantees response is either numeric or categorical.

   @param sY is the front end's response vector.

   @return cardinality of response if classifying, otherwise zero.

 */
int FormResponse(SEXP sY) {
  int ctgWidth = 0;
  if (TYPEOF(sY) == REALSXP) {
    NumericVector y(sY);
    Response::FactoryReg(y.begin());
  }
  else if (TYPEOF(sY) == INTSXP) {
    // Class weighting constructs a proxy response from category frequency.
    // The response is then jittered to diminish the possibility of ties
    // during scoring.  The magnitude of the jitter, then, should be scaled
    // so that no combination of samples can "vote" themselves into a
    // false plurality.
    //
    bool autoWeights = false; // TODO:  Make user option.
    IntegerVector yOneBased(sY);
    IntegerVector y = yOneBased - 1;
    NumericVector classWeight;
    NumericVector tb(table(y));
    ctgWidth = tb.length();
    if (autoWeights) {
      double tbSum = sum(tb);
      NumericVector tbsInv = tbSum / tb;
      double tbsInvSum = sum(tbsInv);
      classWeight = tbsInv / tbsInvSum;
    }
    else {
      classWeight = rep(1.0, ctgWidth);
    }
    int nRow = y.length();
    double recipLen = 1.0 / nRow;
    NumericVector yWeighted = classWeight[y];
    RNGScope scope;
    NumericVector rn(runif(nRow));
    NumericVector proxy = yWeighted + (rn - 0.5) * 0.5 * (recipLen * recipLen);
    Response::FactoryCtg(y.begin(), proxy.begin(), ctgWidth);
  }
  else {
    //TODO:  flag error for unanticipated response types.
  }

  return ctgWidth;
}

/**
   @brief R-language interface to response caching.

   @parm sY is the response vector.

   @return Wrapped value of response cardinality, if applicable.
 */
RcppExport SEXP RcppResponse(SEXP sY) {
  int ctgWidth = FormResponse(sY);

  return wrap(ctgWidth);
}
