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

using namespace std;
using namespace Rcpp;
#include "train.h"

/**
   @brief Dispatches factories for Train class, according to response type.

   @param sY is the response vector.

   @return Number of categories if classifying, otherwise zero.

 */
int FormResponse(SEXP sY) {
  int ctgWidth = 0;
  if (TYPEOF(sY) == REALSXP) {
    NumericVector y(sY);
    Train::ResponseReg(y.begin());
  }
  else if (TYPEOF(sY) == INTSXP) {
    IntegerVector yOneBased(sY);
    IntegerVector y = yOneBased - 1;
    RNGScope scope;
    NumericVector rn(runif(y.length()));
    ctgWidth = Train::ResponseCtg(y.begin(), rn.begin());
  }
  else {
    //TODO:  flag error for unanticipated response types.
  }

  return ctgWidth;
}

/**
   @brief R-language interface to response caching.

   @parm sY is the response vector

   @return Wrapped count of category levels, if applicable.
 */
RcppExport SEXP RcppTrainResponse(SEXP sY) {
  int ctgWidth = FormResponse(sY);

  return wrap(ctgWidth);
}

/**
   @brief Lights off intializations for Train class, which drives training.

   @param sNTree is the number of trees requested.

   @param sQuantiles indicates whether quantiles are requested.

   @param sMinRatio is a threshold ratio of information measures between an index node and its offspring, below which the node does not split.

   @param sBlockSize is a block size, tuned for performance.

   @return Wrapped zero.
 */
RcppExport SEXP RcppTrainInit(SEXP sNTree, SEXP sQuantiles, SEXP sMinRatio, SEXP sBlockSize) {
  Train::Factory(as<int>(sNTree), as<bool>(sQuantiles), as<double>(sMinRatio), as<int>(sBlockSize));

  return wrap(0);
}

/**
   @brief Builds the forest.

   @param sMinH is the smallest index node width allowed for splitting.

   @param sFacWidth records the cardinalities of factor-valued predictors.

   @param sTotBagCount is an output scalar giving the sum of in-bag sizes.

   @param sTotQLeafWidth is an output scalar recording the sum of quantile-vectors sizes.

   @param sTotLevels is an upper bound on the number of levels to construct for each tree.

   @return Wrapped length of forest vector, with output parameters.
 */
RcppExport SEXP RcppTrain(SEXP sMinH, SEXP sFacWidth, SEXP sTotBagCount, SEXP sTotQLeafWidth, SEXP sTotLevels) {
  IntegerVector facWidth(sFacWidth);
  IntegerVector totBagCount(sTotBagCount);
  IntegerVector totQLeafWidth(sTotQLeafWidth);

  int forestHeight = Train::Training(as<int>(sMinH), facWidth.begin(), totBagCount.begin(), totQLeafWidth.begin(), as<int>(sTotLevels));

  return wrap(forestHeight);
}

/**
   @brief Writes forest into storage provided by R.

   @param sPreds are the predictors splitting each nonterminal.

   @param sSplits are the splitting values for each nonterminal.

   @param sScores are the score values for the terminals.

   @param sBump are the left-hand index increments for each nonterminal.

   @param sOrigins is a vector recording the beginning offsets of each tree.

   @param sFacOff are offests into a bit vector recording splitting subsets.

   @param sFacSplits are the bit values of left-hand subsets.

   @return Wrapped zero, with output parameter vectors.
 */
RcppExport SEXP RcppWriteForest(SEXP sPreds, SEXP sSplits, SEXP sScores, SEXP sBump, SEXP sOrigins, SEXP sFacOff, SEXP sFacSplits) { // SEXP sSplitGini)
  IntegerVector rPreds(sPreds);
  NumericVector rSplits(sSplits);
  NumericVector rScores(sScores);
  IntegerVector rBump(sBump);
  IntegerVector rOrigins(sOrigins); // Per-tree offsets of table origins.
  IntegerVector rFacOff(sFacOff); // Per-tree offsets of split bits.
  IntegerVector rFacSplits(sFacSplits);

  //  NumericMatrix rSplitGini(sSplitGini);
  Train::WriteForest(rPreds.begin(), rSplits.begin(), rScores.begin(), rBump.begin(), rOrigins.begin(), rFacOff.begin(), rFacSplits.begin());

  return wrap(0);
}

/**
   @brief Writes forest quantile information into storage provided by R. 

   @param sQYRanked is an output vector giving the response in rank order.

   @param sQRankOrigin is an output vector giving the origin of each tree's ranked information.

   @param sQRank is an output vector of quantile ranks.

   @param sQRankCount is an output vector of quantile rank counts.

   @param sQLeafPos is an output vector recording the offset of each quantile leaf.

   @param sQLeafExtent is an output vector recording the length of each quantile leaf.

   @return Wrapped zero, with output parameters.
 */
RcppExport SEXP RcppWriteQuantile(SEXP sQYRanked, SEXP sQRankOrigin, SEXP sQRank, SEXP sQRankCount, SEXP sQLeafPos, SEXP sQLeafExtent) {
  NumericVector rQYRanked(sQYRanked);
  IntegerVector rQRankOrigin(sQRankOrigin);
  IntegerVector rQRank(sQRank);
  IntegerVector rQRankCount(sQRankCount);
  IntegerVector rQLeafPos(sQLeafPos);
  IntegerVector rQLeafExtent(sQLeafExtent);

  Train::WriteQuantile(rQYRanked.begin(), rQRankOrigin.begin(), rQRank.begin(), rQRankCount.begin(), rQLeafPos.begin(), rQLeafExtent.begin());

  return wrap(0);
}
