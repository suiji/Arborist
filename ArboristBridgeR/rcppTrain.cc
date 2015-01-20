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


#include <R.h>
#include <Rcpp.h>
#include <iostream>

using namespace std;
using namespace Rcpp;
#include "train.h"

// Caller ensures 'sQ' is false for categorical response.
// Returns 'ctgWidth', if categorical, else zero.
//
int FormResponse(SEXP sy) {
  int ctgWidth = 0;
  if (TYPEOF(sy) == REALSXP) {
    NumericVector y(sy);
    Train::ResponseReg(y.begin());
  }
  else if (TYPEOF(sy) == INTSXP) {
    IntegerVector yOneBased(sy);
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

RcppExport SEXP RcppTrainInit(SEXP sNTree, SEXP sQuantiles, SEXP sMinRatio, SEXP sBlockSize) {
  Train::Factory(as<int>(sNTree), as<bool>(sQuantiles), as<double>(sMinRatio), as<int>(sBlockSize));

  return wrap(0);
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
