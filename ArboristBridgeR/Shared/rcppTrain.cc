// Copyright (C)  2012-2017   Mark Seligman
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

#include "rcppRowrank.h"
#include "rcppForest.h"
#include "rcppLeaf.h"
#include "train.h"
#include "forest.h"
#include "leaf.h"

//#include <iostream>
//using namespace std;


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
    for (R_len_t i = 0; i < classWeight.length(); i++) {
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
RcppExport SEXP RcppTrainCtg(SEXP sPredBlock, SEXP sRowRank, SEXP sYOneBased, SEXP sNTree, SEXP sNSamp, SEXP sSampleWeight, SEXP sWithRepl, SEXP sTrainBlock, SEXP sMinNode, SEXP sMinRatio, SEXP sTotLevels, SEXP sLeafMax, SEXP sPredFixed, SEXP sSplitQuant, SEXP sProbVec, SEXP sAutoCompress, SEXP sThinLeaves, SEXP sEnableCoproc, SEXP sClassWeight) {
  BEGIN_RCPP
  List predBlock(sPredBlock);

  if (!predBlock.inherits("PredBlock")) {
    stop("Expecting PredBlock");
  }

  IntegerVector yOneBased(sYOneBased);
  CharacterVector levels(yOneBased.attr("levels"));
  unsigned int ctgWidth = levels.length();

  IntegerVector y = yOneBased - 1;
  std::vector<double> proxy(y.length());
  NumericVector classWeight(as<NumericVector>(sClassWeight));
  RcppProxyCtg(y, classWeight, proxy);

  unsigned int nTree = as<unsigned int>(sNTree);
  std::vector<double> sampleWeight(as<std::vector<double> >(sSampleWeight));

  unsigned int nPredNum = as<unsigned int>(predBlock["nPredNum"]);
  unsigned int nPredFac = as<unsigned int>(predBlock["nPredFac"]);
  unsigned int nPred = nPredNum + nPredFac;

  List signature(as<List>(predBlock["signature"]));
  IntegerVector predMap(as<IntegerVector>(signature["predMap"]));

  NumericVector predProb = NumericVector(sProbVec)[predMap];
  NumericVector splitQuant = NumericVector(sSplitQuant)[predMap];

  Train::Init(nPred, nTree, as<unsigned int>(sNSamp), sampleWeight, as<bool>(sWithRepl), as<unsigned int>(sTrainBlock), as<unsigned int>(sMinNode), as<double>(sMinRatio), as<unsigned int>(sTotLevels), as<unsigned int>(sLeafMax), ctgWidth, as<unsigned int>(sPredFixed), splitQuant.begin(), predProb.begin(), as<bool>(sThinLeaves));

  std::vector<unsigned int> facCard(as<std::vector<unsigned int> >(predBlock["facCard"]));
  std::vector<unsigned int> origin(nTree);
  std::vector<unsigned int> facOrig(nTree);
  std::vector<unsigned int> leafOrigin(nTree);
  std::vector<double> predInfo(nPred);

  std::vector<ForestNode> forestNode;
  std::vector<unsigned int> facSplit;
  std::vector<LeafNode> leafNode;
  std::vector<BagLeaf> bagLeaf;
  std::vector<unsigned int> bagBits;
  std::vector<double> weight;

  double *feNumVal;
  unsigned int *feNumOff, *feRow, *feRank, *feRLE, rleLength;
  RcppRowrank::Unwrap(sRowRank, feNumOff, feNumVal, feRow, feRank, feRLE, rleLength);

  std::vector<std::string> diag;
  std::string diagOut;
  Train::Classification(feRow, feRank, feNumOff, feNumVal, feRLE, rleLength, as<std::vector<unsigned int> >(y), ctgWidth, proxy, origin, facOrig, predInfo, facCard, forestNode, facSplit, leafOrigin, leafNode, as<double>(sAutoCompress), bagLeaf, bagBits, weight, as<bool>(sEnableCoproc), diagOut);
  diag.push_back(diagOut);
  
  RcppRowrank::Clear();
  
  NumericVector infoOut(predInfo.begin(), predInfo.end());
  return List::create(
      _["forest"] = RcppForest::Wrap(origin, facOrig, facSplit, forestNode),
      _["leaf"] = RcppLeaf::WrapCtg(leafOrigin, leafNode, bagLeaf, bagBits, weight, yOneBased.length(), CharacterVector(yOneBased.attr("levels"))),
      _["predInfo"] = infoOut[predMap], // Maps back from core order.
      _["diag"] = diag
  );
  END_RCPP
}


RcppExport SEXP RcppTrainReg(SEXP sPredBlock, SEXP sRowRank, SEXP sY, SEXP sNTree, SEXP sNSamp, SEXP sSampleWeight, SEXP sWithRepl, SEXP sTrainBlock, SEXP sMinNode, SEXP sMinRatio, SEXP sTotLevels, SEXP sLeafMax, SEXP sPredFixed, SEXP sSplitQuant, SEXP sProbVec, SEXP sAutoCompress, SEXP sThinLeaves, SEXP sEnableCoproc, SEXP sRegMono) {
  BEGIN_RCPP
  List predBlock(sPredBlock);

  if (!predBlock.inherits("PredBlock")) {
    stop("Expecting PredBlock");
  }


  List signature(as<List>(predBlock["signature"]));
  IntegerVector predMap(as<IntegerVector>(signature["predMap"]));
  
  unsigned int nTree = as<unsigned int>(sNTree);
  std::vector<double> sampleWeight(as<std::vector<double> >(sSampleWeight));

  unsigned int nPredNum = as<unsigned int>(predBlock["nPredNum"]);
  unsigned int nPredFac = as<unsigned int>(predBlock["nPredFac"]);
  unsigned int nPred = nPredNum + nPredFac;

  NumericVector predProb = NumericVector(sProbVec)[predMap];
  NumericVector regMono = NumericVector(sRegMono)[predMap];
  NumericVector splitQuant = NumericVector(sSplitQuant)[predMap];

  Train::Init(nPred, nTree, as<unsigned int>(sNSamp), sampleWeight, as<bool>(sWithRepl), as<unsigned int>(sTrainBlock), as<unsigned int>(sMinNode), as<double>(sMinRatio), as<unsigned int>(sTotLevels), as<unsigned int>(sLeafMax), 0, as<unsigned int>(sPredFixed), splitQuant.begin(), predProb.begin(), as<bool>(sThinLeaves), regMono.begin());

  double *feNumVal;
  unsigned int *feRow, *feNumOff, *feRank, *feRLE, rleLength;
  RcppRowrank::Unwrap(sRowRank, feNumOff, feNumVal, feRow, feRank, feRLE, rleLength);

  NumericVector y(sY);
  NumericVector yOrdered = clone(y).sort();
  IntegerVector row2Rank = match(y, yOrdered) - 1;

  std::vector<unsigned int> origin(nTree);
  std::vector<unsigned int> facOrig(nTree);
  std::vector<unsigned int> leafOrigin(nTree);
  std::vector<double> predInfo(nPred);

  std::vector<ForestNode> forestNode;
  std::vector<LeafNode> leafNode;
  std::vector<BagLeaf> bagLeaf;
  std::vector<unsigned int> bagBits;
  std::vector<unsigned int> facSplit;

  std::vector<std::string> diag;
  std::string diagOut;
  const std::vector<unsigned int> facCard(as<std::vector<unsigned int> >(predBlock["facCard"]));
  Train::Regression(feRow, feRank, feNumOff, feNumVal, feRLE, rleLength, as<std::vector<double> >(y), as<std::vector<unsigned int> >(row2Rank), origin, facOrig, predInfo, facCard, forestNode, facSplit, leafOrigin, leafNode, as<double>(sAutoCompress), bagLeaf, bagBits, as<bool>(sEnableCoproc), diagOut);
  diag.push_back(diagOut);

  RcppRowrank::Clear();

  // Temporary copy for subscripted access by IntegerVector.
  NumericVector infoOut(predInfo.begin(), predInfo.end()); 
  return List::create(
      _["forest"] = RcppForest::Wrap(origin, facOrig, facSplit, forestNode),
      _["leaf"] = RcppLeaf::WrapReg(leafOrigin, leafNode, bagLeaf, bagBits, as<std::vector<double> >(y)),
      _["predInfo"] = infoOut[predMap], // Maps back from core order.
      _["diag"] = diag
    );

  END_RCPP
}
