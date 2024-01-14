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


vector<double> PRNG::rUnif(size_t nSamp, double scale) {
  RNGScope scope;

  // R requires type double for vector lengths >= 2^32.
  NumericVector rn(runif(double(nSamp)));
  if (scale != 1.0)
    rn = rn * scale;

  return vector<double>(rn.begin(), rn.end());
}


template <>
vector<size_t> PRNG::rUnifIndex(size_t nSamp, size_t idxTop) {
  RNGScope scope;
  
  NumericVector rn(runif(double(nSamp)));
  rn = rn * idxTop;

  return vector<size_t>(rn.begin(), rn.end());
}


template<>
vector<size_t> PRNG::rIndexScatter(size_t nSamp,
				   const vector<size_t>& idxOmit) {
  RNGScope scope;

  vector<size_t> rnTyped = rUnifIndex(nSamp, idxOmit.size());
  vector<size_t> idxOut(nSamp);

  // Rcpp does not appear to support subscripting by numeric types, so the
  // scattering is performed by an explicit loop.
  size_t idx = 0;
  for (const size_t& rnIdx : rnTyped) {
    idxOut[idx++] = idxOmit[rnIdx];
  }
  return idxOut;
}


template <>
vector<size_t> PRNG::rUnifIndex(const vector<size_t>& idxTop) {
  RNGScope scope;

  NumericVector rn(runif(double(idxTop.size())));
  rn = rn * NumericVector(idxTop.begin(), idxTop.end());

  return vector<size_t>(rn.begin(), rn.end());
}
