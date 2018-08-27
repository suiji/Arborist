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

#include <string>
#include <vector>
using namespace std;

RcppExport SEXP Train(const SEXP sArgList);


class TrainBridge {

  /**
     @brief Constructs classification forest.

     @param sNTree is the number of trees requested.

     @return Wrapped length of forest vector, with output parameters.
  */
  static List classification(const IntegerVector &y,
                             const NumericVector &classWeight,
                             const class FrameTrain *frameTrain,
                             const class RankedSet *rankedPair,
                             const IntegerVector &predMap,
                             unsigned int nTree,
                             class BagBridge *bag,
                             vector<string> &diag);
  
  static List regression(const NumericVector &y,
                         const class FrameTrain *frameTrain,
                         const class RankedSet *rankedPair,
                         const IntegerVector &predMap,
                         unsigned int nTree,
                         class BagBridge *bag,
                         vector<string> &diag);

  /**
      @brief R-language interface to response caching.

      @parm sY is the response vector.

      @return Wrapped value of response cardinality, if applicable.
  */
  static NumericVector ctgProxy(const IntegerVector &y,
                                 const NumericVector &classWeight);


  static NumericVector predInfo(const vector<double> &predInfo,
                                const IntegerVector &predMap,
                                unsigned int nTree);

  static List summarize(const class TrainCtg *trainCtg,
                        class BagBridge *bag,
                        const IntegerVector &predMap,
                        unsigned int nTree,
                        const IntegerVector &y,
                        const vector<string> &diag);
  
  static List summarize(const class TrainReg *trainReg,
                        class BagBridge *bag,
                        const IntegerVector &predMap,
                        unsigned int nTree,
                        const NumericVector &y,
                        const vector<string> &diag);
  /**
     @return implicit R_NilValue.
   */
  static SEXP init(const List &argList,
                   const class FrameTrain* frameTrain,
                   const IntegerVector &predMap);


public:  
  static List train(const List &argList,
                    const IntegerVector &predMap,
                    const vector<unsigned int> &facCard,
                    unsigned int nRow);
};

#endif
