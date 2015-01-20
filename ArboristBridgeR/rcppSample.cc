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

#include "rcppSample.h"
#include "sample.h"

int RcppSample::nRow = -1;
int RcppSample::nSamp = -1;
NumericVector RcppSample::sampleWeight = 0;
bool RcppSample::withReplacement = true;

// Sample weights must be retained by the call-back mechansm for sampling at
// tree construction.
//
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

#include <RcppArmadilloExtensions/sample.h>

void RcppSample::SampleRows(int out[]) {
  RNGScope scope;
  IntegerVector rowVec(seq_len(nRow)-1); // Sequential numbering from zero to 'tot'-1.
  IntegerVector samp = RcppArmadillo::sample(rowVec, nSamp, withReplacement, sampleWeight);

  for (int i = 0; i < nRow; i++)
    out[i] = samp[i];
}
