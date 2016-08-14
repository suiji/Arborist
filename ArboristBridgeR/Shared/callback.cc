// Copyright (C)  2012-2016   Mark Seligman
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


#include "rcppSample.h"
#include "callback.h"

/**
   @brief Initializes static state parameters for row sampling.

   @param _nRow is the (fixed) number of response rows.

   @param _weight is the user-specified weighting of row samples.

   @param _repl is true iff sampling with replacement.

   @return void.
 */
void CallBack::SampleInit(unsigned int _nRow, const double _weight[], bool _repl) {
  RcppSample::Init(_nRow, _weight, _repl);
}


/**
   @brief Call-back to Rcpp implementation of row sampling.

   @param nSamp is the number of samples to draw.

   @param out[] outputs the sampled row indices.

   @return Formally void, with copy-out parameter vector.
*/
void CallBack::SampleRows(unsigned int nSamp, int out[]) {
  RcppSample::SampleRows(nSamp, out);
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


// Leaving front-end sorts in place for now, as std:: versions
// not offering a performance advantage, even when run in parallel:
//

/**
  @brief Call-back to R's integer quicksort with indices.

  @param ySorted[] is a copy-out vector containing the sorted integers.

  @param rank2Row[] is the vector of permuted indices.

  @param one is a hard-coded integer indicating unit stride.

  @param nRow is the number of rows to sort.

  @return Formally void, with copy-out parameter vectors.
*/
void CallBack::QSortI(int ySorted[],  unsigned int rank2Row[], int one,  int nRow) {
  R_qsort_int_I(ySorted, (int *) rank2Row, one, (int) nRow);
}


/**
   @brief Call-back to R's double quicksort with indices.

   @param ySorted[] is the copy-out vector of sorted values.

   @param rank2Row[] is the copy-out vector of permuted indices.

   @param one is a hard-coded integer indicating unit stride.

   @param nRow is the number of rows to sort.

   @return Formally void, with copy-out parameter vectors.
*/
void CallBack::QSortD(double ySorted[],  unsigned int rank2Row[], int one,  int nRow) {
  R_qsort_I(ySorted, (int*) rank2Row, one, (int) nRow);
}


