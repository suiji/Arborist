// Copyright (C)  2012-2015  Mark Seligman
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

   @brief C++ class definitions for invocation of R methods implementing response sampling.   Suitable for other uses of sampling, as implemented.

   @author Mark Seligman

 */


#ifndef ARBORIST_RCPP_SAMPLE_H
#define ARBORIST_RCPP_SAMPLE_H

#include <RcppArmadillo.h>
using namespace Rcpp;

class RcppSample {
  static int nRow;
  static int nSamp;
  static NumericVector sampleWeight;
  static bool withReplacement;
public:
  static void Factory(int _nRow, int _nSamp, NumericVector _sampleWeight, bool _withReplacement);
  static void SampleRows(int samp[]);
};

#endif
