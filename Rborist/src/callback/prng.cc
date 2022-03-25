// Copyright (C)  2012-2022   Mark Seligman
//
// This file is part of rfR.
//
// rfR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rfR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file prng.cc

   @brief Implements sampling utitlities by means of calls to front end.

   @author Mark Seligman
 */

#include "prng.h"

#include <algorithm>

NumericVector PRNG::sampleUniform(size_t nObs, size_t nSamp, bool replace) {
  BEGIN_RCPP
    
  RNGScope scope;
  if (replace) {
    NumericVector rn(runif(double(nSamp)));
    return rn * nObs;
  }
  else {
    NumericVector rn(runif(double(nObs)));
    NumericVector rnOut(nSamp);
    vector<size_t> idxSeq(nObs);
    iota(idxSeq.begin(), idxSeq.end(), 0);
    size_t top = nObs;
    for (unsigned int i = 0; i < nSamp; i++) {
      size_t index = top-- * rn[i];
      rnOut[i] = exchange(idxSeq[index], idxSeq[top]);
    }
    return rnOut;
  }

  END_RCPP
}


vector<double> PRNG::rUnif(size_t len, double scale) {
  double dLen = len;
  RNGScope scope;
  NumericVector rn(runif(dLen));
  if (scale != 1.0)
    rn = rn * scale;

  vector<double> rnOut(rn.begin(), rn.end());
  return rnOut;
}
