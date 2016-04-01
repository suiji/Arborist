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

using namespace std;
using namespace Rcpp;

#include "rcppPredblock.h"
#include "rcppForest.h"
#include "rcppLeaf.h"
#include "forest.h"
#include "bv.h"
#include "leaf.h"

//#include <iostream>

/**
 */
void PredTree(const int predMap[], std::vector<unsigned int> &pred) {
  for (unsigned int i = 0; i < pred.size(); i++) {
    unsigned int predCore = pred[i];
    pred[i] = predMap[predCore];
  }
}


/**
   @brief Prepares predictor field for export by remapping to front-end indices.
 */
void PredExport(const int predMap[], std::vector<std::vector<unsigned int> > &pred) {
  for (unsigned int tIdx = 0; tIdx < pred.size(); tIdx++) {
    PredTree(predMap, pred[tIdx]);
  }
}


/**
   @brief Exports core data structures as vector of per-tree vectors.

   @return List with common and regression-specific members.
 */
RcppExport SEXP ExportReg(SEXP sSignature, SEXP sForest, SEXP sLeaf, SEXP sBag) {
  unsigned int nRow;
  IntegerVector predMap;
  SignatureUnwrap(sSignature, nRow, predMap);

  // Instantiates the forest-wide data structures as long vectors, then
  // distributes per tree.
  //
  std::vector<unsigned int> nodeOrigin, facOrigin, splitBV;
  std::vector<ForestNode> *forestNode;
  ForestUnwrap(sForest, nodeOrigin, facOrigin, splitBV, forestNode);

  std::vector<double> yRanked;
  std::vector<unsigned int> leafOrigin;
  std::vector<LeafNode> *leafNode;
  std::vector<RankCount> *leafInfo;
  LeafUnwrapReg(sLeaf, yRanked, leafOrigin, leafNode, leafInfo);

  unsigned int nTree = nodeOrigin.size();
  std::vector<std::vector<unsigned int> > pred(nTree), bump(nTree);
  std::vector<std::vector<double > > split(nTree);
  ForestNode::Export(nodeOrigin, *forestNode, pred, bump, split);
  PredExport(predMap.begin(), pred);
  
  std::vector<std::vector<unsigned int> > facSplit(nTree);
  BVJagged::Export(facOrigin, splitBV, facSplit);

  std::vector<std::vector<unsigned int> > bag(nTree);
  BitMatrix::Export(as<std::vector<unsigned int> >(sBag), nRow, bag);

  std::vector<std::vector<double> > score(nTree);
  std::vector<std::vector<unsigned int> > extent(nTree);
  LeafNode::Export(leafOrigin, *leafNode, score, extent);

  std::vector<std::vector<unsigned int> > rank(nTree), sCount(nTree);
  LeafReg::Export(leafOrigin, *leafInfo, rank, sCount);

  List outBundle = List::create(
				_["pred"] = pred,
				_["bump"] = bump,
				_["split"] = split,
				_["facSplit"] = facSplit,
				_["bag"] = bag,
				_["score"] = score,
				_["extent"] = extent,
				_["rank"] = rank,
				_["sCount"] = sCount
				);
  outBundle.attr("class") = "ExportReg";

  return outBundle;
}


/**
   @brief Exports core data structures as vector of per-tree vectors.

   @return List with common and classification-specific members.
 */
RcppExport SEXP ExportCtg(SEXP sSignature, SEXP sForest, SEXP sLeaf, SEXP sBag) {
  unsigned int nRow;
  IntegerVector predMap;
  SignatureUnwrap(sSignature, nRow, predMap);

  std::vector<unsigned int> nodeOrigin, facOrigin, splitBV;
  std::vector<ForestNode> *forestNode;
  ForestUnwrap(sForest, nodeOrigin, facOrigin, splitBV, forestNode);

  std::vector<unsigned int> leafOrigin;
  std::vector<LeafNode> *leafNode;
  std::vector<double> leafInfo;
  CharacterVector levels;
  LeafUnwrapCtg(sLeaf, leafOrigin, leafNode, leafInfo, levels);

  unsigned int nTree = nodeOrigin.size();
  std::vector<std::vector<unsigned int> > pred(nTree), bump(nTree);
  std::vector<std::vector<double > > split(nTree);
  ForestNode::Export(nodeOrigin, *forestNode, pred, bump, split);
  PredExport(predMap.begin(), pred);
  
  std::vector<std::vector<unsigned int> > facSplit(nTree);
  BVJagged::Export(facOrigin, splitBV, facSplit);

  std::vector<std::vector<unsigned int> > bag(nTree);
  BitMatrix::Export(as<std::vector<unsigned int> >(sBag), nRow, bag);

  std::vector<std::vector<double> > score(nTree);
  std::vector<std::vector<unsigned int> > extent(nTree);
  LeafNode::Export(leafOrigin, *leafNode, score, extent);

  std::vector<std::vector<double> > weight(nTree);
  LeafCtg::Export(leafOrigin, leafInfo, levels.length(), weight);

  List outBundle = List::create(
				_["pred"] = pred,
				_["bump"] = bump,
				_["split"] = split,
				_["facSplit"] = facSplit,
				_["bag"] = bag,
				_["score"] = score,
				_["extent"] = extent,
				_["weight"] = weight
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
  std::vector<std::vector<unsigned int> > pred = forestCore["pred"];
  std::vector<std::vector<unsigned int> > bump = forestCore["bump"];
  std::vector<std::vector<double > > split = forestCore["split"];
  std::vector<std::vector<unsigned int> > facSplit = forestCore["facSplit"];
  std::vector<std::vector<unsigned int> > bag = forestCore["bag"];
  IntegerVector incrL(bump[tIdx].begin(), bump[tIdx].end());
  List ffTree = List::create(
     _["pred"] = pred[tIdx],
     _["daugherL"] = incrL,
     _["daughterR"] = incrL + 1,
     _["split"] = split[tIdx],
     _["facSplit"] = facSplit[tIdx],
     _["bag"] = bag[tIdx]
     );

  ffTree.attr("class") = "FFloorTree";
  return ffTree;
}


/**
 */
RcppExport SEXP FFloorTreeReg(SEXP sCoreReg, unsigned int tIdx) {
  List ffReg = List::create(
    _["internal"] = FFloorInternal(sCoreReg, tIdx),
    _["leaf"] = FFloorLeafReg(sCoreReg, tIdx)
  );

  ffReg.attr("class") = "FFloorTreeReg";
  return ffReg;
}


/**
 */
RcppExport SEXP FFloorTreeCtg(SEXP sCoreCtg, unsigned int tIdx) {
  List ffCtg = List::create(
    _["internal"] = FFloorInternal(sCoreCtg, tIdx),
    _["leaf"] = FFloorLeafCtg(sCoreCtg, tIdx)
  );

  ffCtg.attr("class") = "FFloorTreeCtg";
  return ffCtg;
}


/**
 */
RcppExport SEXP FFloorReg(SEXP sSignature, SEXP sForest, SEXP sLeaf, SEXP sBag) {
  SEXP coreReg = ExportReg(sSignature, sForest, sLeaf, sBag);
  unsigned int nTree = NTree(coreReg);
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
     trees[tIdx] = FFloorTreeReg(coreReg, tIdx);
  }

  List ffe = List::create(
			  _["tree"] = trees
			  );
  ffe.attr("class") = "ForestFloorReg";
  return ffe;
}


/**
 */
RcppExport SEXP FFloorCtg(SEXP sSignature, SEXP sForest, SEXP sLeaf, SEXP sBag) {
  SEXP coreCtg = ExportCtg(sSignature, sForest, sLeaf, sBag);
  unsigned int nTree = NTree(coreCtg);
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    trees[tIdx] = FFloorTreeCtg(coreCtg, tIdx);
  }

  List ffe = List::create(
			  _["tree"] = trees
			  );
  ffe.attr("class") = "ForestFloorCtg";
  return ffe;
}


/**
   @brief Structures forest summary for analysis by ForestFloor package.

   @param sForest is the Forest summary.

   @return wrapped ForestFloorExport object.
 */
RcppExport SEXP RcppForestFloorExport(SEXP sArbOut) {
  List arbOut(sArbOut);
  if (!arbOut.inherits("Rborist")) {
    warning("Expecting an Rborist object");
    return List::create(0);
  }

  List training((SEXP) arbOut["training"]);
  List leaf((SEXP) arbOut["leaf"]);
  if (leaf.inherits("LeafReg"))  {
    return FFloorReg(arbOut["signature"], arbOut["forest"], arbOut["leaf"], training["bag"]);
  }
  else if (leaf.inherits("LeafCtg")) {
    return FFloorCtg(arbOut["signature"], arbOut["forest"], arbOut["leaf"], training["bag"]);
  }
  else {
    warning("Unrecognized forest type.");
    return List::create(0);
  }
}
