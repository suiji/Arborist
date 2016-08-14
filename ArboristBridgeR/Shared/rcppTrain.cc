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

#include <Rcpp.h>
using namespace Rcpp;

#include "rcppForest.h"
#include "rcppLeaf.h"
#include "train.h"
#include "forest.h"
#include "leaf.h"

//#include <iostream>
using namespace std;


/**
   @brief R-language interface to response caching.

   @parm sY is the response vector.

   @return Wrapped value of response cardinality, if applicable.
 */
void RcppProxyCtg(IntegerVector y, NumericVector classWeight, std::vector<double> &proxy) {
  // Class weighting constructs a proxy response from category frequency.
  // The response is then jittered to diminish the possibility of ties
  // during scoring.  The magnitude of the jitter, then, should be scaled
  // so that no combination of samples can "vote" themselves into a
  // false plurality.
  //
  if (is_true(all(classWeight == 0.0))) { // Place-holder for balancing.
    NumericVector tb(table(y));
    for (unsigned int i = 0; i < classWeight.length(); i++) {
      classWeight[i] = tb[i] == 0.0 ? 0.0 : 1.0 / tb[i];
    }
  }
  classWeight = classWeight / sum(classWeight);

  unsigned int nRow = y.length();
  double recipLen = 1.0 / nRow;
  NumericVector yWeighted = classWeight[y];
  RNGScope scope;
  NumericVector rn(runif(nRow));
  for (unsigned int i = 0; i < nRow; i++)
    proxy[i] = yWeighted[i] + (rn[i] - 0.5) * 0.5 * (recipLen * recipLen);
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
RcppExport SEXP RcppTrainCtg(SEXP sPredBlock, SEXP sRowRank, SEXP sYOneBased, SEXP sNTree, SEXP sNSamp, SEXP sSampleWeight, SEXP sWithRepl, SEXP sTrainBlock, SEXP sMinNode, SEXP sMinRatio, SEXP sTotLevels, SEXP sPredFixed, SEXP sProbVec, SEXP sClassWeight) {
  List predBlock(sPredBlock);
  if (!predBlock.inherits("PredBlock"))
    stop("Expecting PredBlock");

  List rowRank(sRowRank);
  if (!rowRank.inherits("RowRank"))
    stop("Expecting RowRank");

  unsigned int nRow = as<unsigned int>(predBlock["nRow"]);
  unsigned int nPredNum = as<unsigned int>(predBlock["nPredNum"]);
  unsigned int nPredFac = as<unsigned int>(predBlock["nPredFac"]);
  NumericMatrix xNum;
  IntegerMatrix feInvNum;
  if (nPredNum > 0) {
    xNum = as<NumericMatrix>(predBlock["blockNum"]);
    feInvNum = as<IntegerMatrix>(rowRank["invNum"]);
  }
  else {
    xNum = NumericMatrix(0,0);
    feInvNum = IntegerMatrix(0,0);
  }

  IntegerVector facCard(0);
  unsigned int cardMax = 0;
  if (nPredFac > 0) {
    facCard = as<IntegerVector>(predBlock["facCard"]);
    cardMax = max(facCard);
  }
  List signature(as<List>(predBlock["signature"]));
  IntegerVector predMap(as<IntegerVector>(signature["predMap"]));

  IntegerMatrix feRow = as<IntegerMatrix>(rowRank["row"]);
  IntegerMatrix feRank = as<IntegerMatrix>(rowRank["rank"]);

  IntegerVector yOneBased(sYOneBased);
  CharacterVector levels(yOneBased.attr("levels"));
  unsigned int ctgWidth = levels.length();

  IntegerVector y = yOneBased - 1;
  std::vector<double> proxy(y.length());
  NumericVector classWeight(as<NumericVector>(sClassWeight));
  RcppProxyCtg(y, classWeight, proxy);

  unsigned int nTree = as<unsigned int>(sNTree);
  NumericVector sampleWeight(as<NumericVector>(sSampleWeight));

  unsigned int nPred = nPredNum + nPredFac;
  NumericVector predProb = NumericVector(sProbVec)[predMap];

  Train::Init(xNum.begin(), (unsigned int*) facCard.begin(), cardMax, nPredNum, nPredFac, nRow, nTree, as<unsigned int>(sNSamp), sampleWeight.begin(), as<bool>(sWithRepl), as<unsigned int>(sTrainBlock), as<unsigned int>(sMinNode), as<double>(sMinRatio), as<unsigned int>(sTotLevels), ctgWidth, as<unsigned int>(sPredFixed), predProb.begin());

  std::vector<unsigned int> origin(nTree);
  std::vector<unsigned int> facOrig(nTree);
  std::vector<unsigned int> leafOrigin(nTree);
  NumericVector predInfo(nPred);

  std::vector<ForestNode> forestNode;
  std::vector<unsigned int> facSplit;
  std::vector<LeafNode> leafNode;
  std::vector<BagRow> bagRow;
  std::vector<double> weight;

  Train::Classification((unsigned int*) feRow.begin(), (unsigned int*) feRank.begin(), (unsigned int*) feInvNum.begin(), as<std::vector<unsigned int> >(y), ctgWidth, proxy, origin, facOrig, predInfo.begin(), forestNode, facSplit, leafOrigin, leafNode, bagRow, weight);


  return List::create(
      _["forest"] = RcppForest::Wrap(origin, facOrig, facSplit, forestNode),
      _["leaf"] = RcppLeaf::WrapCtg(leafOrigin, leafNode, bagRow, nRow, weight, CharacterVector(yOneBased.attr("levels"))),
      _["predInfo"] = predInfo[predMap] // Maps back from core order.
  );
}


RcppExport SEXP RcppTrainReg(SEXP sPredBlock, SEXP sRowRank, SEXP sY, SEXP sNTree, SEXP sNSamp, SEXP sSampleWeight, SEXP sWithRepl, SEXP sTrainBlock, SEXP sMinNode, SEXP sMinRatio, SEXP sTotLevels, SEXP sPredFixed, SEXP sProbVec, SEXP sRegMono) {
  List predBlock(sPredBlock);
  if (!predBlock.inherits("PredBlock"))
    stop("Expecting PredBlock");

  List rowRank(sRowRank);
  if (!rowRank.inherits("RowRank"))
    stop("Expecting RowRank");

  unsigned int nRow = as<unsigned int>(predBlock["nRow"]);
  unsigned int nPredNum = as<unsigned int>(predBlock["nPredNum"]);
  unsigned int nPredFac = as<unsigned int>(predBlock["nPredFac"]);
  NumericMatrix xNum;
  IntegerMatrix feInvNum;
  if (nPredNum > 0) {
    xNum = as<NumericMatrix>(predBlock["blockNum"]);
    feInvNum = as<IntegerMatrix>(rowRank["invNum"]);
  }
  else {
    xNum = NumericMatrix(0,0);
    feInvNum = IntegerMatrix(0,0);
  }

  IntegerVector facCard(0);
  unsigned int cardMax = 0;
  if (nPredFac > 0) {
    facCard = as<IntegerVector>(predBlock["facCard"]);
    cardMax = max(facCard);
  }
  List signature(as<List>(predBlock["signature"]));
  IntegerVector predMap(as<IntegerVector>(signature["predMap"]));
  
  unsigned int nTree = as<unsigned int>(sNTree);
  NumericVector sampleWeight(as<NumericVector>(sSampleWeight));

  unsigned int nPred = nPredNum + nPredFac;
  NumericVector predProb = NumericVector(sProbVec)[predMap];
  NumericVector regMono = NumericVector(sRegMono)[predMap];
  
  Train::Init(xNum.begin(), (unsigned int*) facCard.begin(), cardMax, nPredNum, nPredFac, nRow, nTree, as<unsigned int>(sNSamp), sampleWeight.begin(), as<bool>(sWithRepl), as<unsigned int>(sTrainBlock), as<unsigned int>(sMinNode), as<double>(sMinRatio), as<unsigned int>(sTotLevels), 0, as<unsigned int>(sPredFixed), predProb.begin(), regMono.begin());

  IntegerMatrix feRow = as<IntegerMatrix>(rowRank["row"]);
  IntegerMatrix feRank = as<IntegerMatrix>(rowRank["rank"]);

  NumericVector y(sY);
  NumericVector yRanked = clone(y).sort();
  IntegerVector row2Rank = match(y, yRanked) - 1;

  std::vector<unsigned int> origin(nTree);
  std::vector<unsigned int> facOrig(nTree);
  std::vector<unsigned int> leafOrigin(nTree);
  NumericVector predInfo(nPred);

  std::vector<ForestNode> forestNode;
  std::vector<LeafNode> leafNode;
  std::vector<BagRow> bagRow;
  std::vector<unsigned int> rank;
  std::vector<unsigned int> facSplit;

  Train::Regression((unsigned int*) feRow.begin(), (unsigned int*) feRank.begin(), (unsigned int*) feInvNum.begin(), as<std::vector<double> >(y), as<std::vector<unsigned int> >(row2Rank), origin, facOrig, predInfo.begin(), forestNode, facSplit, leafOrigin, leafNode, bagRow, rank);

  return List::create(
      _["forest"] = RcppForest::Wrap(origin, facOrig, facSplit, forestNode),
      _["leaf"] = RcppLeaf::WrapReg(leafOrigin, leafNode, bagRow, nRow, rank, as<std::vector<double> >(yRanked)),
      _["predInfo"] = predInfo[predMap] // Maps back from core order.
    );
}
