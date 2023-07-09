// Copyright (C)  2012-2023   Mark Seligman
//
// This file is part of RboristBase.
//
// RboristBase is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RboristBase is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RboristBase.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file prng.cc

   @brief Implements random variate generation via calls to front end.

   @author Mark Seligman
 */

#include "prng.h"

#include <Rcpp.h>
using namespace Rcpp;


vector<double> PRNG::rUnif(size_t len, double scale) {
  double dLen = len; // May be necessary for values > 2^32.
  RNGScope scope;
  NumericVector rn(runif(dLen));
  if (scale != 1.0)
    rn = rn * scale;

  return vector<double>(rn.begin(), rn.end());
}


vector<size_t> PRNG::rUnifIndex(size_t len, size_t scale) {
  double dLen = len; // May be necessary for values > 2^32.
  RNGScope scope;
  NumericVector rn(runif(dLen));
  rn = rn * scale;

  return vector<size_t>(rn.begin(), rn.end());
}


vector<size_t> PRNG::rUnifIndex(const vector<size_t>& scale) {
  double dLen = scale.size(); // May be necessary for values > 2^32.
  RNGScope scope;
  NumericVector scaleCopy(scale.begin(), scale.end());
  NumericVector rn(runif(dLen));
  rn = rn * scaleCopy;

  vector<size_t> rnOut(rn.begin(), rn.end());
  return rnOut;
}
