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
   @file rcppTrain.cc

   @brief C++ interface to R entry for training.

   @author Mark Seligman
 */

#include <R.h>
#include <Rcpp.h>

using namespace Rcpp;

#include "train.h"
//#include <iostream>

/**
   @brief R-language interface to response caching.

   @parm sY is the response vector.

   @return Wrapped value of response cardinality, if applicable.
 */
RcppExport SEXP RcppResponseCtg(IntegerVector y, unsigned int &ctgWidth) {
  // Class weighting constructs a proxy response from category frequency.
  // The response is then jittered to diminish the possibility of ties
  // during scoring.  The magnitude of the jitter, then, should be scaled
  // so that no combination of samples can "vote" themselves into a
  // false plurality.
  //
  bool autoWeights = false; // TODO:  Make user option.
  NumericVector classWeight;
  NumericVector tb(table(y));
  ctgWidth = tb.length();
  if (autoWeights) {
    double tbSum = sum(tb);
    NumericVector tbsInv = tbSum / tb;
    double tbsInvSum = sum(tbsInv);
    classWeight = tbsInv / tbsInvSum;
  }
  else {
    classWeight = rep(1.0, ctgWidth);
  }
  int nRow = y.length();
  double recipLen = 1.0 / nRow;
  NumericVector yWeighted = classWeight[y];
  RNGScope scope;
  NumericVector rn(runif(nRow));
  NumericVector proxy = yWeighted + (rn - 0.5) * 0.5 * (recipLen * recipLen);

  return wrap(proxy);
}


/**
   @brief Constructs classification forest.

   @param sNTree is the number of trees requested.

   @param sMinNode is the smallest index node width allowed for splitting.

   @param sMinRatio is a threshold ratio of information measures between an index node and its offspring, below which the node does not split.

   @param sTotLevels is an upper bound on the number of levels to construct for each tree.

   @param sTotLevels is an upper bound on the number of levels to construct for each tree.

   @return Wrapped length of forest vector, with output parameters.
 */
RcppExport SEXP RcppTrainCtg(SEXP sYOneBased, SEXP sNTree, SEXP sNPred, SEXP sNSamp, SEXP sTrainBlock, SEXP sMinNode, SEXP sMinRatio, SEXP sTotLevels) {
  IntegerVector yOneBased(sYOneBased);
  int nTree = as<int>(sNTree);
  int nPred = as<int>(sNPred);
  int nSamp = as<int>(sNSamp);
  IntegerVector y = yOneBased - 1;
  unsigned int ctgWidth;
  NumericVector proxy = RcppResponseCtg(y, ctgWidth);
  int nRow = y.length();

  Train::Init(nTree, nRow, nPred, nSamp, as<int>(sTrainBlock), as<int>(sMinNode), as<double>(sMinRatio), as<int>(sTotLevels), ctgWidth);

  IntegerVector origin(nTree);
  IntegerVector facOrig(nTree);
  NumericVector predInfo(nPred);
  std::vector<int> pred;
  std::vector<double> split;
  std::vector<int> bump;
  std::vector<unsigned int> facSplit;
  std::vector<double> weight;

  //  Maintains forest-wide in-bag set as bits.  Achieves high compression, but
  //  may not scale to multi-gigarow sets.
  //
  int inBagSize = ((nTree * nRow) + 8 * sizeof(unsigned int) - 1) / (8 * sizeof(unsigned int));
  IntegerVector inBag(inBagSize, 0);

  Train::ForestCtg(y.begin(), proxy.begin(), (unsigned int*) inBag.begin(), origin.begin(), facOrig.begin(), predInfo.begin(), pred, split, bump, facSplit, weight);

  return List::create(
		      _["bag"] = inBag,
		      _["origin"] = origin,
		      _["pred"] = pred,
		      _["split"] = split,
		      _["bump"] = bump,
		      _["facOrig"] = facOrig,
		      _["facSplit"] = facSplit,
		      _["predInfo"] = predInfo,
		      _["weight"] = weight
  );
}

using namespace std;
RcppExport SEXP RcppTrainReg(SEXP sY, SEXP sNTree, SEXP sNPred, SEXP sNSamp, SEXP sTrainBlock, SEXP sMinNode, SEXP sMinRatio, SEXP sTotLevels) {
  NumericVector y(sY);
  int nTree = as<int>(sNTree);
  int nPred = as<int>(sNPred);
  int nRow = y.length();
  int nSamp = as<int>(sNSamp);
  Train::Init(nTree, nRow, nPred, nSamp, as<int>(sTrainBlock), as<int>(sMinNode), as<double>(sMinRatio), as<int>(sTotLevels));

  NumericVector yRanked(nRow);
  IntegerVector origin(nTree);
  IntegerVector facOrig(nTree);
  NumericVector predInfo(nPred);

  // Variable-length vectors.
  //
  std::vector<int> pred;
  std::vector<double> split;
  std::vector<int> bump;
  std::vector<unsigned int> facSplit;
  std::vector<unsigned int> rank;
  std::vector<unsigned int> sCount;

  //  Maintains forest-wide in-bag set as bits.  Achieves high compression, but
  //  may not scale to multi-gigarow sets.
  //  Inititalized to zeroes.
  //
  int inBagSize = ((nTree * nRow) + 8 * sizeof(unsigned int) - 1) / (8 * sizeof(unsigned int));
  IntegerVector inBag(inBagSize, 0);

  Train::ForestReg(y.begin(), yRanked.begin(), (unsigned int*) inBag.begin(), origin.begin(), facOrig.begin(), predInfo.begin(), pred, split, bump, facSplit, rank, sCount);

  return List::create(
     _["bag"] = inBag,
     _["origin"] = origin,
     _["pred"] = pred,
     _["split"] = split,
     _["bump"] = bump,
     _["facOrig"] = facOrig,
     _["facSplit"] = facSplit,
     _["predInfo"] = predInfo,
     _["rank"] = rank,
     _["sCount"] = sCount,
     _["yRanked"] = yRanked
    );
}
