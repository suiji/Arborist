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

RcppExport SEXP TrainForest(const SEXP sArgList);


struct TrainBridge {

  // Training granularity.  Values guesstimated to minimize footprint of
  // Core-to-Bridge copies while also not over-allocating:
  static const unsigned int treeChunk = 20;
  static constexpr double allocSlop = 1.15d;
  
  static bool verbose; // Whether to report progress while training.

  class BagBridge *bag;
  const unsigned int nTree;
  unique_ptr<class FBTrain> forest;
  NumericVector predInfo;
  unique_ptr<class LBTrain> leaf;

  TrainBridge(class BagBridge* bag_,
              unsigned int nTree_,
              const IntegerVector& predMap,
              const NumericVector& yTrain);
  
  TrainBridge(class BagBridge* bag_,
              unsigned int nTree_,
              const IntegerVector& predMap,
              const IntegerVector& yTrain);

  /**
     @brief Estimates scale factor for allocating forest-wide vector.

     @param treesTot is the total number of trees trained so far.

     @return scale factor estimation for accommodating entire forest.
   */
  double safeScale(unsigned int treesTot) {
    return (treesTot == nTree ? 1 : allocSlop) * double(nTree) / treesTot;
  }


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

  NumericVector scalePredInfo(const IntegerVector &predMap);

  /**
     @return implicit R_NilValue.
   */
  static SEXP init(const List &argList,
                   const class FrameTrain* frameTrain,
                   const IntegerVector &predMap);


  static List train(const List &argList,
                    const IntegerVector &predMap,
                    const vector<unsigned int> &facCard,
                    unsigned int nRow);

  void consumeReg(const class TrainReg* train,
                  unsigned int treeOff,
                  double scale);

  void consumeCtg(const class TrainCtg* train,
                  unsigned int treeOff,
                  double scale);

  void consume(const class Train* train,
               unsigned int treeOff,
               double scale);

  List summarize(const IntegerVector& predMap,
                 const vector<string>& diag);
};
#endif
