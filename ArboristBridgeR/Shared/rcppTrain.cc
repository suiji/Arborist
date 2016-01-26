// Copyright (C)  2012-2016   Mark Seligman
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

#include "rcppForest.h"
#include "rcppLeaf.h"
#include "train.h"

//using namespace std;
//#include <iostream>

/**
   @brief R-language interface to response caching.

   @parm sY is the response vector.

   @return Wrapped value of response cardinality, if applicable.
 */
RcppExport SEXP RcppResponseCtg(IntegerVector y) {
  // Class weighting constructs a proxy response from category frequency.
  // The response is then jittered to diminish the possibility of ties
  // during scoring.  The magnitude of the jitter, then, should be scaled
  // so that no combination of samples can "vote" themselves into a
  // false plurality.
  //
  bool autoWeights = false; // TODO:  Make user option.
  NumericVector classWeight;
  NumericVector tb(table(y));
  double tbSum = sum(tb);
  if (autoWeights) {
    NumericVector tbsInv = tbSum / tb;
    double tbsInvSum = sum(tbsInv);
    classWeight = tbsInv / tbsInvSum;
  }
  else {
    classWeight = rep(1.0, tb.length());
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
RcppExport SEXP RcppTrainCtg(SEXP sPredBlock, SEXP sRowRank, SEXP sYOneBased, SEXP sNTree, SEXP sNSamp, SEXP sSampleWeight, SEXP sWithRepl, SEXP sTrainBlock, SEXP sMinNode, SEXP sMinRatio, SEXP sTotLevels, SEXP sPredFixed, SEXP sProbVec) {
  List predBlock(sPredBlock);
  if (!predBlock.inherits("PredBlock"))
    stop("Expecting PredBlock");

  List rowRank(sRowRank);
  if (!rowRank.inherits("RowRank"))
    stop("Expecting RowRank");

  int nRow = as<int>(predBlock["nRow"]);
  int nPredNum = as<int>(predBlock["nPredNum"]);
  int nPredFac = as<int>(predBlock["nPredFac"]);
  double *feNum = 0;
  int *feInvNum = 0;
  if (nPredNum > 0) {
    NumericMatrix xNum(as<NumericMatrix>(predBlock["blockNum"]));
    feNum = xNum.begin();
    IntegerVector invNum(as<IntegerVector>(rowRank["invNum"]));
    feInvNum = invNum.begin();
  }
  int *feFacCard = 0;
  int cardMax = 0;
  if (nPredFac > 0) {
    IntegerVector facCard(as<IntegerVector>(predBlock["facCard"]));
    feFacCard = facCard.begin();
    cardMax = max(facCard);
  }
  IntegerVector predMap(as<IntegerVector>(predBlock["predMap"]));

  IntegerVector feRow(as<IntegerVector>(rowRank["row"]));
  IntegerVector feRank(as<IntegerVector>(rowRank["rank"]));

  IntegerVector yOneBased(sYOneBased);
  CharacterVector levels(yOneBased.attr("levels"));
  unsigned int ctgWidth = levels.length();

  IntegerVector y = yOneBased - 1;
  NumericVector proxy = RcppResponseCtg(y);

  int nTree = as<int>(sNTree);
  NumericVector sampleWeight(as<NumericVector>(sSampleWeight));

  int nPred = nPredNum + nPredFac;
  // Probability vector reindexed by core ordering.
  NumericVector probVec(sProbVec);
  IntegerVector invMap(nPred);
  IntegerVector seq = seq_len(nPred) - 1;
  invMap[predMap] = seq;
  NumericVector predProb = probVec[invMap];

  Train::Init(feNum, feFacCard, predMap.begin(), cardMax, nPredNum, nPredFac, nRow, nTree, as<int>(sNSamp), sampleWeight.begin(), as<bool>(sWithRepl), as<int>(sTrainBlock), as<int>(sMinNode), as<double>(sMinRatio), as<int>(sTotLevels), ctgWidth, as<int>(sPredFixed), predProb.begin());

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
  std::vector<int> inBag;

  Train::Classification(feRow.begin(), feRank.begin(), feInvNum, y.begin(), ctgWidth, proxy.begin(), inBag, origin.begin(), facOrig.begin(), predInfo.begin(), pred, split, bump, facSplit, weight);

  return List::create(
      _["forest"] = RcppForestWrap(pred, split, bump, origin, facOrig, facSplit),
      _["leaf"] = RcppLeafWrapCtg(weight, CharacterVector(yOneBased.attr("levels"))),
      _["bag"] = inBag,
      _["predInfo"] = predInfo[predMap] // Maps back from core order.
  );
}

using namespace std;

RcppExport SEXP RcppTrainReg(SEXP sPredBlock, SEXP sRowRank, SEXP sY, SEXP sNTree, SEXP sNSamp, SEXP sSampleWeight, SEXP sWithRepl, SEXP sTrainBlock, SEXP sMinNode, SEXP sMinRatio, SEXP sTotLevels, SEXP sPredFixed, SEXP sProbVec) {
  List predBlock(sPredBlock);
  if (!predBlock.inherits("PredBlock"))
    stop("Expecting PredBlock");

  List rowRank(sRowRank);
  if (!rowRank.inherits("RowRank"))
    stop("Expecting RowRank");

  int nRow = as<int>(predBlock["nRow"]);
  int nPredNum = as<int>(predBlock["nPredNum"]);
  int nPredFac = as<int>(predBlock["nPredFac"]);
  double *feNum = 0;
  int *feInvNum = 0;
  if (nPredNum > 0) {
    NumericMatrix xNum(as<NumericMatrix>(predBlock["blockNum"]));
    feNum = xNum.begin();
    IntegerVector invNum(as<IntegerVector>(rowRank["invNum"]));
    feInvNum = invNum.begin();
  }
  int *feFacCard = 0;
  int cardMax = 0;
  if (nPredFac > 0) {
    IntegerVector facCard(as<IntegerVector>(predBlock["facCard"]));
    feFacCard = facCard.begin();
    cardMax = max(facCard);
  }
  IntegerVector predMap(as<IntegerVector>(predBlock["predMap"]));
  
  NumericVector y(sY);
  int nTree = as<int>(sNTree);
  NumericVector sampleWeight(as<NumericVector>(sSampleWeight));

  int nPred = nPredNum + nPredFac;
  // Probability vector reindexed by core ordering.
  NumericVector probVec(sProbVec);
  IntegerVector invMap(nPred);
  IntegerVector seq = seq_len(nPred) - 1;
  invMap[predMap] = seq;
  NumericVector predProb = probVec[invMap];

  
  Train::Init(feNum, feFacCard, predMap.begin(), cardMax, nPredNum, nPredFac, nRow, nTree, as<int>(sNSamp), sampleWeight.begin(), as<bool>(sWithRepl), as<int>(sTrainBlock), as<int>(sMinNode), as<double>(sMinRatio), as<int>(sTotLevels), 0, as<int>(sPredFixed), predProb.begin());

  IntegerVector feRow(as<IntegerVector>(rowRank["row"]));
  IntegerVector feRank(as<IntegerVector>(rowRank["rank"]));

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
  std::vector<int> inBag;

  Train::Regression(feRow.begin(), feRank.begin(), feInvNum, y.begin(), yRanked.begin(), inBag, origin.begin(), facOrig.begin(), predInfo.begin(), pred, split, bump, facSplit, rank, sCount);

  return List::create(
      _["forest"] = RcppForestWrap(pred, split, bump, origin, facOrig, facSplit),
      _["leaf"] = RcppLeafWrapReg(rank, sCount, yRanked),
      _["bag"] = inBag,
      _["predInfo"] = predInfo[predMap] // Maps back from core order.
    );
}
