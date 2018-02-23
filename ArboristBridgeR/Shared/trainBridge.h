// Copyright (C)  2012-2018   Mark Seligman
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
   @file bridgeTrain.h

   @brief C++ interface to R entry for training.

   @author Mark Seligman
 */

#ifndef ARBORIST_TRAIN_BRIDGE_H
#define ARBORIST_TRAIN_BRIDGE_H

#include <Rcpp.h>
using namespace Rcpp;

#include "train.h"


RcppExport SEXP Train(const SEXP sArgList);


class TrainBridge {
  /**
     @brief Constructs classification forest.

     @param sNTree is the number of trees requested.

     @return Wrapped length of forest vector, with output parameters.
  */
  static List Classification(const IntegerVector &y,
			     const NumericVector &classWeight,
			     const class FrameTrain *frameTrain,
			     const class RowRankBridge *rowRank,
			     const IntegerVector &predMap,
			     unsigned int nTree,
			     vector<string> &diag);

  /**
      @brief R-language interface to response caching.

      @parm sY is the response vector.

      @return Wrapped value of response cardinality, if applicable.
  */
  static NumericVector CtgProxy(const IntegerVector &y,
				 const NumericVector &classWeight);

  static NumericVector PredInfo(const vector<double> &predInfo,
				const IntegerVector &predMap,
				unsigned int nTree);
  

  static List Regression(const NumericVector &y,
			 const class FrameTrain *frameTrain,
			 const class RowRankBridge *rowRank,
			 const IntegerVector &predMap,
			 unsigned int nTree,
			 vector<string> &diag);

 public:
  static List Train(const List &argList,
		    const class FrameTrain *frameTrain,
		    const class RowRankBridge *rowRank,
		    const IntegerVector &predMap,
		    unsigned int nTree,
		    vector<string> &diag);
  
  /**
     @return implicit R_NilValue.
   */
  static SEXP Init(const List &argList,
		   const IntegerVector &predMap);

};

#endif
