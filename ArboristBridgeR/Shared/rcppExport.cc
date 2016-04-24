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
   @file rcppExport.cc

   @brief C++ interface to R entry for export methods.

   @author Mark Seligman
 */

#include <Rcpp.h>
using namespace Rcpp;

using namespace std;
//#include <iostream>

#include "rcppPredblock.h"
#include "rcppForest.h"
#include "rcppLeaf.h"
#include "forest.h"
#include "bv.h"
#include "leaf.h"


/**
   @brief Recasts 'pred' field of nonterminals to front-end facing values.

   @return void.
 */
void PredTree(const int predMap[], std::vector<unsigned int> &pred, std::vector<unsigned int> &bump) {
  for (unsigned int i = 0; i < pred.size(); i++) {
    if (bump[i] > 0) { // terminal 'pred' values do not reference predictors.
      unsigned int predCore = pred[i];
      pred[i] = predMap[predCore];
    }
  }
}


/**
   @brief Prepares predictor field for export by remapping to front-end indices.
 */
void PredExport(const int predMap[], std::vector<std::vector<unsigned int> > &predTree, std::vector<std::vector<unsigned int> > &bumpTree) {
  for (unsigned int tIdx = 0; tIdx < predTree.size(); tIdx++) {
    PredTree(predMap, predTree[tIdx], bumpTree[tIdx]);
  }
}


/**
   @brief Exports core data structures as vector of per-tree vectors.

   @return List with common and regression-specific members.
 */
RcppExport SEXP ExportReg(SEXP sForest, SEXP sLeaf, IntegerVector predMap) {

  // Instantiates the forest-wide data structures as long vectors, then
  // distributes per tree.
  //
  std::vector<unsigned int> nodeOrigin, facOrigin, splitBV;
  std::vector<ForestNode> *forestNode;
  ForestUnwrap(sForest, nodeOrigin, facOrigin, splitBV, forestNode);

  unsigned int nTree = nodeOrigin.size();
  std::vector<std::vector<unsigned int> > predTree(nTree), bumpTree(nTree);
  std::vector<std::vector<double > > splitTree(nTree);
  ForestNode::Export(nodeOrigin, *forestNode, predTree, bumpTree, splitTree);
  PredExport(predMap.begin(), predTree, bumpTree);
  
  std::vector<std::vector<unsigned int> > facSplitTree(nTree);
  BVJagged::Export(facOrigin, splitBV, facSplitTree);

  std::vector<double> yRanked;
  std::vector<unsigned int> leafOrigin;
  std::vector<LeafNode> *leafNode;
  std::vector<BagRow> *bagRow;
  unsigned int rowTrain;
  std::vector<unsigned int> rank;
  LeafUnwrapReg(sLeaf, yRanked, leafOrigin, leafNode, bagRow, rowTrain, rank);

  std::vector<std::vector<unsigned int> > rowTree(nTree), sCountTree(nTree);
  std::vector<std::vector<double> > scoreTree(nTree);
  std::vector<std::vector<unsigned int> > extentTree(nTree);
  std::vector<std::vector<unsigned int> > rankTree(nTree);
  LeafReg::Export(leafOrigin, *leafNode, *bagRow, rank, rowTree, sCountTree, scoreTree, extentTree, rankTree);

  List outBundle = List::create(
				_["rowTrain"] = rowTrain,
				_["pred"] = predTree,
				_["bump"] = bumpTree,
				_["split"] = splitTree,
				_["facSplit"] = facSplitTree,
				_["row"] = rowTree,
				_["sCount"] = sCountTree,
				_["score"] = scoreTree,
				_["extent"] = extentTree,
				_["rank"] = rankTree
				);
  outBundle.attr("class") = "ExportReg";

  return outBundle;
}


/**
   @brief Exports core data structures as vector of per-tree vectors.

   @return List with common and classification-specific members.
 */
RcppExport SEXP ExportCtg(SEXP sForest, SEXP sLeaf, IntegerVector predMap) {
  std::vector<unsigned int> nodeOrigin, facOrigin, splitBV;
  std::vector<ForestNode> *forestNode;
  ForestUnwrap(sForest, nodeOrigin, facOrigin, splitBV, forestNode);

  unsigned int nTree = nodeOrigin.size();
  std::vector<std::vector<unsigned int> > predTree(nTree), bumpTree(nTree);
  std::vector<std::vector<double > > splitTree(nTree);
  ForestNode::Export(nodeOrigin, *forestNode, predTree, bumpTree, splitTree);
  PredExport(predMap.begin(), predTree, bumpTree);
  
  std::vector<std::vector<unsigned int> > facSplitTree(nTree);
  BVJagged::Export(facOrigin, splitBV, facSplitTree);

  std::vector<unsigned int> leafOrigin;
  std::vector<LeafNode> *leafNode;
  std::vector<BagRow> *bagRow;
  unsigned int rowTrain;
  std::vector<double> weight;
  CharacterVector yLevel;
  LeafUnwrapCtg(sLeaf, leafOrigin, leafNode, bagRow, rowTrain, weight, yLevel);

  std::vector<std::vector<unsigned int> > rowTree(nTree), sCountTree(nTree);
  std::vector<std::vector<double> > scoreTree(nTree);
  std::vector<std::vector<unsigned int> > extentTree(nTree);
  std::vector<std::vector<double> > weightTree(nTree);
  LeafCtg::Export(leafOrigin, *leafNode, *bagRow, weight, yLevel.length(), rowTree, sCountTree, scoreTree, extentTree, weightTree);

  List outBundle = List::create(
				_["rowTrain"] = rowTrain,
				_["pred"] = predTree,
				_["bump"] = bumpTree,
				_["split"] = splitTree,
				_["facSplit"] = facSplitTree,
				_["row"] = rowTree,
				_["sCount"] = sCountTree,
				_["score"] = scoreTree,
				_["extent"] = extentTree,
				_["yLevel"] = yLevel,
				_["weight"] = weightTree
				);
  outBundle.attr("class") = "ExportCtg";

  return outBundle;
}


unsigned int NTree(SEXP sExp) {
  List exp(sExp);
  if (!exp.inherits("ExportCtg") && !exp.inherits("ExportReg"))
    stop("Unrecognized export object");

  std::vector<std::vector<unsigned int> > pred = exp["pred"];

  return pred.size();
}


/**
   @brief Only the scores are of interest to ForestFloor.

   @param forestCore is the exported core image.

   @param tIdx is the tree index.

   @return Vector of score values.
 */
RcppExport SEXP FFloorLeafReg(SEXP sForestCore, unsigned int tIdx) {
  List forestCore(sForestCore);
  std::vector<std::vector<double> > score = forestCore["score"];
  List ffLeaf = List::create(
     _["score"] = score[tIdx]
    );
  
  ffLeaf.attr("class") = "FFloorLeafReg";
  return ffLeaf;
}


/**
   @brief Only the scores and weights are of interest to ForestFloor.

   @param forestCore is the exported core image.

   @param tIdx is the tree index.

   @return Vector of score values.
 */
RcppExport SEXP FFloorLeafCtg(SEXP sForestCore, unsigned int tIdx) {
  List forestCore(sForestCore);
  std::vector<std::vector<double> > score = forestCore["score"];
  std::vector<std::vector<double> > weight = forestCore["weight"];
  unsigned int leafCount = score[tIdx].size();
  NumericMatrix weightOut = NumericMatrix(weight[tIdx].size() / leafCount, leafCount, weight[tIdx].begin());
  List ffLeaf = List::create(
     _["score"] = score[tIdx],
     _["weight"] = transpose(weightOut)
     );

  ffLeaf.attr("class") = "FFloorLeafCtg";
  return ffLeaf;
}


/**
 */
RcppExport SEXP FFloorInternal(SEXP sForestCore, unsigned int tIdx) {
  List forestCore(sForestCore);
  std::vector<std::vector<unsigned int> > predTree = forestCore["pred"];
  std::vector<std::vector<unsigned int> > bumpTree = forestCore["bump"];
  std::vector<std::vector<double > > splitTree = forestCore["split"];
  std::vector<std::vector<unsigned int> > facSplitTree = forestCore["facSplit"];
  IntegerVector incrL(bumpTree[tIdx].begin(), bumpTree[tIdx].end());
  IntegerVector predIdx(predTree[tIdx].begin(), predTree[tIdx].end());
  List ffTree = List::create(
     _["pred"] = ifelse(incrL == 0, -(predIdx + 1), predIdx),
     _["daughterL"] = incrL,
     _["daughterR"] = ifelse(incrL == 0, 0, incrL + 1),
     _["split"] = splitTree[tIdx],
     _["facSplit"] = facSplitTree[tIdx]
     );

  ffTree.attr("class") = "FFloorTree";
  return ffTree;
}


RcppExport SEXP FFloorBag(SEXP sForestCore, int tIdx) {
  List forestCore(sForestCore);
  unsigned int rowTrain = as<unsigned int>(forestCore["rowTrain"]);
  std::vector<std::vector<unsigned int> > rowTree = forestCore["row"];
  std::vector<std::vector<unsigned int> > sCountTree = forestCore["sCount"];
  IntegerVector bag = IntegerVector(rowTrain);
  IntegerVector row(rowTree[tIdx].begin(), rowTree[tIdx].end());
  IntegerVector sCount(sCountTree[tIdx].begin(), sCountTree[tIdx].end());
  bag[row] = sCount;

  return bag;
}


/**
 */
RcppExport SEXP FFloorTreeReg(SEXP sCoreReg, unsigned int tIdx) {
  List ffReg = List::create(
    _["internal"] = FFloorInternal(sCoreReg, tIdx),
    _["leaf"] = FFloorLeafReg(sCoreReg, tIdx),
    _["bag"] = FFloorBag(sCoreReg, tIdx)
  );

  ffReg.attr("class") = "FFloorTreeReg";
  return ffReg;
}


/**
 */
RcppExport SEXP FFloorTreeCtg(SEXP sCoreCtg, unsigned int tIdx) {
  List ffCtg = List::create(
    _["internal"] = FFloorInternal(sCoreCtg, tIdx),
    _["leaf"] = FFloorLeafCtg(sCoreCtg, tIdx),
    _["bag"] = FFloorBag(sCoreCtg, tIdx)
  );

  ffCtg.attr("class") = "FFloorTreeCtg";
  return ffCtg;
}


/**
 */
RcppExport SEXP FFloorReg(SEXP sForest, SEXP sLeaf, IntegerVector predMap, List predLevel) {
  SEXP sCoreReg = ExportReg(sForest, sLeaf, predMap);
  unsigned int nTree = NTree(sCoreReg);
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
     trees[tIdx] = FFloorTreeReg(sCoreReg, tIdx);
  }

  int facCount = predLevel.length();
  IntegerVector facMap(predMap.end() - facCount, predMap.end());
  List ffe = List::create(
    _["facMap"] = facMap,
    _["predLevel"] = predLevel,
    _["tree"] = trees
  );
  ffe.attr("class") = "ForestFloorReg";
  return ffe;
}


/**
 */
RcppExport SEXP FFloorCtg(SEXP sForest, SEXP sLeaf, IntegerVector predMap, List predLevel) {
  SEXP sCoreCtg = ExportCtg(sForest, sLeaf, predMap);
  unsigned int nTree = NTree(sCoreCtg);
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    trees[tIdx] = FFloorTreeCtg(sCoreCtg, tIdx);
  }

  int facCount = predLevel.length();
  IntegerVector facMap(predMap.end() - facCount, predMap.end());
  List coreCtg(sCoreCtg);
  List ffe = List::create(
   _["facMap"] = facMap,
   _["predLevel"] = predLevel,
   _["yLevel"] = as<CharacterVector>(coreCtg["yLevel"]),
   _["tree"] = trees
  );
  ffe.attr("class") = "ForestFloorCtg";
  return ffe;
}


/**
   @brief Structures forest summary for analysis by ForestFloor package.

   @param sForest is the Forest summary.

   @return ForestFloorExport as List.
 */
RcppExport SEXP RcppForestFloorExport(SEXP sArbOut) {
  List arbOut(sArbOut);
  if (!arbOut.inherits("Rborist")) {
    warning("Expecting an Rborist object");
    return List::create(0);
  }

  IntegerVector predMap;
  List predLevel;
  SignatureUnwrap(arbOut["signature"], predMap, predLevel);

  List leaf((SEXP) arbOut["leaf"]);
  if (leaf.inherits("LeafReg"))  {
    return FFloorReg(arbOut["forest"], arbOut["leaf"], predMap, predLevel);
  }
  else if (leaf.inherits("LeafCtg")) {
    return FFloorCtg(arbOut["forest"], arbOut["leaf"], predMap, predLevel);
  }
  else {
    warning("Unrecognized forest type.");
    return List::create(0);
  }
}
