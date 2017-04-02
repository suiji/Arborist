// Copyright (C)  2012-2017   Mark Seligman
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
   @file rcppSample.cc

   @brief Interface to front-end methods implementing (response) sampling.

   @author Mark Seligman
 */

#include <RcppArmadilloExtensions/sample.h>
#include "rcppSample.h"

//#include <iostream>
//using namespace std;

unsigned int RcppSample::nRow = 0;
bool RcppSample::withRepl = false;

NumericVector weightNull(0);
NumericVector &RcppSample::weight = weightNull;

/**
   @brief Caches row sampling parameters as static values.

   @param _nRow is length of the response vector.

   @param _weight is user-specified weighting of row samples.

   @param _withRepl is true iff sampling with replacement.

   @return void.
 */
void RcppSample::Init(unsigned int _nRow, const double feWeight[], bool _withRepl) {
  nRow = _nRow;
  NumericVector _weight(nRow);
  weight = _weight;
  for (unsigned int i = 0; i < nRow; i++)
    weight[i] = feWeight[i];
  withRepl = _withRepl;
}


/**
   @brief Samples row indices either with or without replacement using methods from RccpArmadillo.

   @param nSamp is the number of samples to draw.

   @param out[] is an output vector of sampled row indices.

   @return void, with output vector.
 */
void RcppSample::SampleRows(unsigned int nSamp, int out[]) {
  RNGScope scope;
  IntegerVector rowVec(seq_len(nRow)-1);
  IntegerVector samp = RcppArmadillo::sample(rowVec, nSamp, withRepl, weight);

  for (unsigned int i = 0; i < nSamp; i++)
    out[i] = samp[i];
}
