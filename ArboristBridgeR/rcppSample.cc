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
   @file rcppSample.cc

   @brief Interface to front-end methods implementing (response) sampling.

   @author Mark Seligman
 */

#include "rcppSample.h"
#include "sample.h"

int RcppSample::nRow = -1;
int RcppSample::nSamp = -1;
NumericVector RcppSample::sampleWeight = 0;
bool RcppSample::withReplacement = true;
/**
  @brief R-language entry to sampling factories.

  @param sNRow is the number of rows of observations.

  @param sNPred is the number of columns of observations.

  @param sNSamp is the number of samples requested.

  @param sSampWeight is a vector of response-element weights.

  @param sWithReplacement indicates whether sampling with replacement has been requested.

  @return Wrapped zero.
*/
RcppExport SEXP RcppSample(SEXP sNRow, SEXP sNPred, SEXP sNSamp, SEXP sSampWeight, SEXP sWithReplacement) {
  int nRow = as<int>(sNRow);
  int nPred = as<int>(sNPred);
  int nSamp = as<int>(sNSamp);
  NumericVector sampWeight(sSampWeight);
  bool withReplacement = as<bool>(sWithReplacement);

  RcppSample::Factory(nRow, nSamp, sampWeight, withReplacement);
  Sample::Factory(nRow, nSamp, nPred);

  return wrap(0);
}

/**
   @brief Lights off the initializations needed for sampling.

   @param _nRow is the number of rows of observations.

   @param _nSamp is the number of samples requested.

   @param _sampleWeight weights response vector elements.

   @param _withReplacement indicates sampling mode.

   @return void.
 */
void RcppSample::Factory(int _nRow, int _nSamp, NumericVector _sampleWeight, bool _withReplacement ) {
    nRow = _nRow;
    nSamp = _nSamp;
    sampleWeight = _sampleWeight;
    withReplacement = _withReplacement;
}


#include <RcppArmadilloExtensions/sample.h>
/**
   @brief Samples row indices either with or without replacement using methods from RccpArmadillo.

   @param out[] is an output vector of sampled row indices.

   @return Wrapped zero, with output parameter vector.
 */
void RcppSample::SampleRows(int out[]) {
  RNGScope scope;
  IntegerVector rowVec(seq_len(nRow)-1); // Sequential numbering from zero to 'tot'-1.
  IntegerVector samp = RcppArmadillo::sample(rowVec, nSamp, withReplacement, sampleWeight);

  for (int i = 0; i < nRow; i++)
    out[i] = samp[i];
}
