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
   @file callback.cc

   @brief Implements sorting and sampling utitlities by means of calls to front end.  Employs pre-allocated copy-out parameters to avoid dependence on front end's memory allocation.

   @author Mark Seligman
 */

#include "callback.h"
#include <R.h>

#include "rcppSample.h"

/**
   @brief Call-back to Rcpp implementation of row sampling.

   @param samp[] is a copy-out/copy-out vector containing the rows
   to be sampled and overwritten with the samples.

   @return Formally void, with copy-out parameter vector.
*/

void CallBack::SampleRows(int samp[]) {
  RcppSample::SampleRows(samp);
}

/**
  @brief Call-back to R's integer quicksort with indices.

  @param ySorted[] is a copy-out vector containing the sorted integers.

  @param rank2Row[] is the vector of permuted indices.

  @param one is a hard-coded integer indicating unit stride.

  @param nRow is the number of rows to sort.

  @return Formally void, with copy-out parameter vectors.
*/

void CallBack::QSortI(int ySorted[], int rank2Row[], int one, int nRow) {
  R_qsort_int_I(ySorted, rank2Row, one, nRow);
}

/**
   @brief Call-back to R's double quicksort with indices.

   @param ySorted[] is the copy-out vector of sorted values.

   @param rank2Row[] is the copy-out vector of permuted indices.

   @param one is a hard-coded integer indicating unit stride.

   @param nRow is the number of rows to sort.

   @return Formally void, with copy-out parameter vectors.
*/

void CallBack::QSortD(double ySorted[], int rank2Row[], int one, int nRow) {
  R_qsort_I(ySorted, rank2Row, one, nRow);
}

/**
   @brief Call-back to R's uniform random-variate generator.

   @param len is number of variates to generate.

   @param out[] is the copy-out vector of generated variates.

   @return Formally void, with copy-out parameter vector.
    
 */
void CallBack::RUnif(int len, double out[]) {
  RNGScope scope;
  NumericVector rn(runif(len));

  for (int i = 0; i < len; i++)
    out[i] = rn[i];
}
