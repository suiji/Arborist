# Copyright (C)  2012-2014   Mark Seligman
##
## This file is part of ArboristBridgeR.
##
## ArboristBridgeR is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## ArboristBridgeR is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ArboristBridgeR.  If not, see <http://www.gnu.org/licenses/>.


#include <R.h>
#include <Rcpp.h>

#include <iostream>

using namespace Rcpp;
using namespace std;

#include "train.h"

// Caller ensures 'sQ' is false for categorical response.
// Returns 'ctgWidth', if categorical, else zero.
//
int FormResponse(SEXP sy) {
  int ctgWidth;
  if (TYPEOF(sy) == REALSXP) {
    NumericVector y(sy);
    Train::ResponseReg(y.begin());
    ctgWidth = 0;
  }
  else if (TYPEOF(sy) == INTSXP) {
    IntegerVector yOneBased(sy);
    IntegerVector y = yOneBased - 1;
    RNGScope scope;
    NumericVector rn(runif(y.length()));
    ctgWidth = Train::ResponseCtg(y.begin(), rn.begin());
  }
  return ctgWidth;
  //else   TODO:  flag error for unanticipated response types.
}

// Predictors must be partitioned, regardless whether training, predicting
// or testing.
//
// For training, blocks must be cloned to provide space for sorting.  Other modes
// can use R's allocations, as passed.  Cloning is never necessary for data frames,
// however, as submatrices are built from data frames on the fly.
//


// Training variant clones implicitly.
//
RcppExport SEXP RcppTrainInt(SEXP sx) {
  IntegerMatrix xi(sx);

  Train::IntBlock(xi.begin(), xi.nrow(), xi.ncol());

  return wrap(0);
}


RcppExport SEXP RcppTrainResponse(SEXP sy) {
  int ctgWidth = FormResponse(sy);

  return wrap(ctgWidth);
}

RcppExport SEXP RcppSampWeight(SEXP sSampWeight) {
  NumericVector sampWeight(sSampWeight);

  Train::SampleWeights(sampWeight.begin());

  return wrap(0);
}

RcppExport SEXP RcppTrainInit(SEXP sPredWeight, SEXP sPredProb, SEXP sNTree, SEXP sNSamp, SEXP sSmpReplace, SEXP sQuantiles, SEXP sMinRatio, SEXP sBlockSize) {
  NumericVector predWeight(sPredWeight);
  Train::TrainInit(predWeight.begin(), as<double>(sPredProb), as<int>(sNTree), as<int>(sNSamp), as<bool>(sSmpReplace), as<bool>(sQuantiles), as<double>(sMinRatio), as<int>(sBlockSize));
}

//
RcppExport SEXP RcppTrain(SEXP sMinH, SEXP sFacWidth, SEXP sTotBagCount, SEXP sTotQLeafWidth, SEXP sTotLevels) {
  IntegerVector facWidth(sFacWidth);
  IntegerVector totBagCount(sTotBagCount);
  IntegerVector totQLeafWidth(sTotQLeafWidth);

  int forestHeight = Train::Training(as<int>(sMinH), facWidth.begin(), totBagCount.begin(), totQLeafWidth.begin(), as<int>(sTotLevels));

  return wrap(forestHeight);
}

// Writes back trees.  Tree write-back requires memory allocated from the R caller, hence
// is performed separately from training, which establishes the requisite array sizes.
//
// Calls destructors belonging to remaining objects needed for the write.
// Returns value of mean-square error, wrapped.
//
RcppExport SEXP RcppWriteForest(SEXP sPreds, SEXP sSplits, SEXP sScores, SEXP sBumpL, SEXP sBumpR, SEXP sOrigins, SEXP sFacOff, SEXP sFacSplits) { // SEXP sSplitGini)
  IntegerVector rPreds(sPreds);
  NumericVector rSplits(sSplits);
  NumericVector rScores(sScores);
  IntegerVector rBumpL(sBumpL);
  IntegerVector rBumpR(sBumpR);
  IntegerVector rOrigins(sOrigins); // Per-tree offsets of table origins.
  IntegerVector rFacOff(sFacOff); // Per-tree offsets of split bits.
  IntegerVector rFacSplits(sFacSplits);

  //  NumericMatrix rSplitGini(sSplitGini);
  Train::WriteForest(rPreds.begin(), rSplits.begin(), rScores.begin(), rBumpL.begin(), rBumpR.begin(), rOrigins.begin(), rFacOff.begin(), rFacSplits.begin());

  return wrap(0);
}

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
