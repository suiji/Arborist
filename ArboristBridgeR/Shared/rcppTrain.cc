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


//using namespace std;
//#include <iostream>

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
  List signature(as<List>(predBlock["signature"]));
  IntegerVector predMap(as<IntegerVector>(signature["predMap"]));

  IntegerVector feRow(as<IntegerVector>(rowRank["row"]));
  IntegerVector feRank(as<IntegerVector>(rowRank["rank"]));

  IntegerVector yOneBased(sYOneBased);
  CharacterVector levels(yOneBased.attr("levels"));
  unsigned int ctgWidth = levels.length();

  IntegerVector y = yOneBased - 1;
  std::vector<double> proxy(y.length());
  NumericVector classWeight(as<NumericVector>(sClassWeight));
  RcppProxyCtg(y, classWeight, proxy);

  int nTree = as<int>(sNTree);
  NumericVector sampleWeight(as<NumericVector>(sSampleWeight));

  int nPred = nPredNum + nPredFac;
  NumericVector predProb = NumericVector(sProbVec)[predMap];

  Train::Init(feNum, feFacCard, cardMax, nPredNum, nPredFac, nRow, nTree, as<int>(sNSamp), sampleWeight.begin(), as<bool>(sWithRepl), as<int>(sTrainBlock), as<int>(sMinNode), as<double>(sMinRatio), as<int>(sTotLevels), ctgWidth, as<int>(sPredFixed), predProb.begin());

  std::vector<unsigned int> origin(nTree);
  std::vector<unsigned int> facOrig(nTree);
  std::vector<unsigned int> leafOrigin(nTree);
  NumericVector predInfo(nPred);

  std::vector<ForestNode> forestNode;

  std::vector<unsigned int> facSplit;
  std::vector<LeafNode> leafNode;
  std::vector<double> leafInfo;

  //  Maintains forest-wide in-bag set as bits.  Achieves high compression, but
  //  may not scale to multi-gigarow sets.
  std::vector<unsigned int> inBag;

  Train::Classification(feRow.begin(), feRank.begin(), feInvNum, as<std::vector<unsigned int> >(y), ctgWidth, proxy, inBag, origin, facOrig, predInfo.begin(), forestNode, facSplit, leafOrigin, leafNode, leafInfo);

  return List::create(
      _["forest"] = ForestWrap(origin, facOrig, facSplit, forestNode),
      _["leaf"] = LeafWrapCtg(leafOrigin, leafNode, leafInfo, CharacterVector(yOneBased.attr("levels"))),
      _["bag"] = inBag,
      _["predInfo"] = predInfo[predMap] // Maps back from core order.
  );
}

using namespace std;

RcppExport SEXP RcppTrainReg(SEXP sPredBlock, SEXP sRowRank, SEXP sY, SEXP sNTree, SEXP sNSamp, SEXP sSampleWeight, SEXP sWithRepl, SEXP sTrainBlock, SEXP sMinNode, SEXP sMinRatio, SEXP sTotLevels, SEXP sPredFixed, SEXP sProbVec, SEXP sRegMono) {
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
  List signature(as<List>(predBlock["signature"]));
  IntegerVector predMap(as<IntegerVector>(signature["predMap"]));
  
  int nTree = as<int>(sNTree);
  NumericVector sampleWeight(as<NumericVector>(sSampleWeight));

  int nPred = nPredNum + nPredFac;
  NumericVector predProb = NumericVector(sProbVec)[predMap];
  NumericVector regMono = NumericVector(sRegMono)[predMap];
  
  Train::Init(feNum, feFacCard, cardMax, nPredNum, nPredFac, nRow, nTree, as<int>(sNSamp), sampleWeight.begin(), as<bool>(sWithRepl), as<int>(sTrainBlock), as<int>(sMinNode), as<double>(sMinRatio), as<int>(sTotLevels), 0, as<int>(sPredFixed), predProb.begin(), regMono.begin());

  IntegerVector feRow(as<IntegerVector>(rowRank["row"]));
  IntegerVector feRank(as<IntegerVector>(rowRank["rank"]));

  NumericVector y(sY);
  NumericVector yRanked = clone(y).sort();
  IntegerVector row2Rank = match(y, yRanked) - 1;

  std::vector<unsigned int> origin(nTree);
  std::vector<unsigned int> facOrig(nTree);
  std::vector<unsigned int> leafOrigin(nTree);
  NumericVector predInfo(nPred);

  std::vector<ForestNode> forestNode;
  std::vector<LeafNode> leafNode;
  std::vector<RankCount> leafInfo;
  std::vector<unsigned int> facSplit;

  //  Maintains forest-wide in-bag set as bits.  Achieves high compression, but
  //  may not scale to multi-gigarow sets.
  //  Inititalized to zeroes.
  //
  std::vector<unsigned int> inBag;

  Train::Regression(feRow.begin(), feRank.begin(), feInvNum, as<std::vector<double> >(y), as<std::vector<unsigned int> >(row2Rank), inBag, origin, facOrig, predInfo.begin(), forestNode, facSplit, leafOrigin, leafNode, leafInfo);

  return List::create(
      _["forest"] = ForestWrap(origin, facOrig, facSplit, forestNode),
      _["leaf"] = LeafWrapReg(leafOrigin, leafNode, leafInfo, as<std::vector<double> >(yRanked)),
      _["bag"] = inBag,
      _["predInfo"] = predInfo[predMap] // Maps back from core order.
    );
}
