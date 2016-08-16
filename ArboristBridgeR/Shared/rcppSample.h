// Copyright (C)  2012-2016  Mark Seligman
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
   @file rcppSample.h

   @brief C++ class definitions for invocation of R methods implementing response sampling.   Can be extended for other instances of sampling.

   @author Mark Seligman

 */


#ifndef ARBORIST_RCPP_SAMPLE_H
#define ARBORIST_RCPP_SAMPLE_H

#include <Rcpp.h>
using namespace Rcpp;

/**
   @brief Row-sampling parameters supplied by the front end are invariant, so can be cached as static.
 */
class RcppSample {
  static unsigned int nRow;
  static bool withRepl;
  static NumericVector &weight;
public:
  static void Init(unsigned int _nRow, const double feWeight[], bool _withRepl);
  static void SampleRows(unsigned int nSamp, int out[]);
};

#endif
