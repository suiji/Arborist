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

#include "callback.h"
#include <R.h>

#include "rcppSample.h"

// Call-back to Rcpp implementation of row sampling.
//
void CallBack::SampleRows(int samp[]) {
  RcppSample::SampleRows(samp);
}

// Call-back to R's integer quicksort with indices.
//
void CallBack::QSortI(int ySorted[], int rank2Row[], int one, int nRow) {
  R_qsort_int_I(ySorted, rank2Row, one, nRow);
}

// Call-back to R's double quicksort with indices.
//
void CallBack::QSortD(double ySorted[], int rank2Row[], int one, int nRow) {
  R_qsort_I(ySorted, rank2Row, one, nRow);
}

// Call-back to R's uniform random-variate generator.
//
void CallBack::RUnif(int len, double out[]) {
  RNGScope scope;
  NumericVector rn(runif(len));

  for (int i = 0; i < len; i++)
    out[i] = rn[i];
}
