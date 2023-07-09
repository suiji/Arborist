// Copyright (C)  2012-2023  Mark Seligman
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
   @file weightingR.h

   @brief C++ interface to R entry for Meinhausen's (2006) weights.

   @author Mark Seligman
 */

#ifndef RBORIST_BASE_FORESTWEIGHT_R_H
#define RBORIST_BASE_FORESTWEIGHT_R_H

#include <Rcpp.h>
using namespace Rcpp;

/**
   @brief Entry from R.
 */
RcppExport SEXP forestWeightRcpp(const SEXP sTrain,
				 const SEXP sSampler,
				 const SEXP sPredict,
				 const SEXP sArgs);


struct ForestWeightR {
  /**
     @brief Meinshausen's forest weights for multiple predictions.

     @return matrix with rows of per-observation weights.
   */
  static NumericMatrix forestWeight(const List& lTrain,
				    const List& lSampler,
				    const NumericMatrix& indices,
				    const List& lArgs);
};

#endif
