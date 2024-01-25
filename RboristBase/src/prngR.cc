// Copyright (C)  2012-2024   Mark Seligman
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

   @brief R-language instantiation of base PRNG methods.

   @author Mark Seligman
 */

#include "prng.h"

#include <Rcpp.h>
using namespace Rcpp;


template<>
vector<size_t> PRNG::rUnif(size_t nSamp,
			   size_t scale) {
  RNGScope scope;

  // R requires type double for vector lengths >= 2^32.
  NumericVector rn(runif(double(nSamp)));

  vector<size_t> variates(nSamp);
  size_t idx = 0;
  for (const double& variate : rn) {
    variates[idx++] = variate * scale;
  }

  return variates;
}


template<>
vector<unsigned int> PRNG::rUnif(unsigned int nSamp,
				 unsigned int scale) {
  RNGScope scope;

  NumericVector rn(runif(nSamp));
  if (scale != 1)
    rn = rn * scale;

  return vector<unsigned int>(rn.begin(), rn.end());
}


template<>
vector<double> PRNG::rUnif(double nSamp,
			   double scale) {
  RNGScope scope;

  NumericVector rn(runif(nSamp));
  if (scale != 1.0)
    rn = rn * scale;

  return vector<double>(rn.begin(), rn.end());
}
