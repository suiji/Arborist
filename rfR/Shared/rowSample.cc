// Copyright (C)  2012-2019   Mark Seligman
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
   @file rowSample.cc

   @brief Interface to Rcpp base methods implementing sampling.

   @author Mark Seligman
 */

#include "rowSample.h"


bool RowSample::withRepl = false;

NumericVector weightNull(0);
NumericVector &RowSample::weight = weightNull;

IntegerVector rowSeqNull(0);
IntegerVector &RowSample::rowSeq = rowSeqNull;


void RowSample::init(const NumericVector &feWeight, bool withRepl_) {
  weight = feWeight;
  rowSeq = seq(0, feWeight.length()-1);

  withRepl = withRepl_;
}


IntegerVector RowSample::sampleRows(unsigned int nSamp) {
  BEGIN_RCPP
  RNGScope scope;
  return sample(rowSeq, nSamp, withRepl, clone(weight));

  END_RCPP
}
