// Copyright (C)  2012-2018   Mark Seligman
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
   @file exportBridge.cc

   @brief C++ interface to R entry for export methods.

   @author Mark Seligman
 */

#include "exportBridge.h"
#include "framemapBridge.h"
#include "forestBridge.h"
#include "leafBridge.h"

#include <vector>

/**
   @brief Structures forest summary for analysis by ForestFloor package.

   @param sForest is the Forest summary.

   @return ForestFloorExport as List.
 */
RcppExport SEXP ForestFloorExport(SEXP sArbOut) {
  BEGIN_RCPP
    
  List arbOut(sArbOut);
  if (!arbOut.inherits("Rborist")) {
    warning("Expecting an Rborist object");
    return List::create(0);
  }

  IntegerVector predMap;
  List predLevel;
  FramemapBridge::SignatureUnwrap(arbOut["signature"], predMap, predLevel);

  List leaf((SEXP) arbOut["leaf"]);
  if (leaf.inherits("LeafReg"))  {
    return ExportBridge::FFloorReg(arbOut["forest"], leaf, predMap, predLevel);
  }
  else if (leaf.inherits("LeafCtg")) {
    return ExportBridge::FFloorCtg(arbOut["forest"], leaf, predMap, predLevel);
  }
  else {
    warning("Unrecognized forest type.");
    return List::create(0);
  }

  END_RCPP
}


/**
   @brief Exports core data structures as vector of per-tree vectors.

   @return List with common and regression-specific members.
 */
List ExportBridge::ExportReg(const SEXP sForest,
			     const List &leaf,
			     IntegerVector &predMap,
			     unsigned int &nTree) {
  BEGIN_RCPP

  auto forest = make_unique<ForestExport>(sForest, predMap);
  nTree = forest->NTree();

  auto leafReg = make_unique<LeafExportReg>(leaf, true);
  List outBundle = List::create(
				_["pred"] = forest->PredTree(),
				_["bump"] = forest->BumpTree(),
				_["split"] = forest->SplitTree(),
				_["facSplit"] = forest->FacSplitTree(),
				_["rowTrain"] = leafReg->RowTrain(),
				_["row"] = leafReg->RowTree(),
				_["sCount"] = leafReg->SCountTree(),
				_["score"] = leafReg->ScoreTree(),
				_["extent"] = leafReg->ExtentTree()
				);
  outBundle.attr("class") = "ExportReg";
  return outBundle;

  END_RCPP
}


/**
   @brief Exports core data structures as vector of per-tree vectors.

   @return List with common and classification-specific members.
 */
List ExportBridge::ExportCtg(const SEXP sForest,
			     const List &leaf,
			     IntegerVector &predMap,
			     unsigned int &nTree) {
  BEGIN_RCPP

  auto forest = make_unique<ForestExport>(sForest, predMap);
  nTree = forest->NTree();

  auto leafCtg = make_unique<LeafExportCtg>(leaf, true);
  List outBundle = List::create(
				_["pred"] = forest->PredTree(),
				_["bump"] = forest->BumpTree(),
				_["split"] = forest->SplitTree(),
				_["facSplit"] = forest->FacSplitTree(),
				_["rowTrain"] = leafCtg->RowTrain(),
				_["row"] = leafCtg->RowTree(),
				_["sCount"] = leafCtg->SCountTree(),
				_["score"] = leafCtg->ScoreTree(),
				_["extent"] = leafCtg->ExtentTree(),
				_["yLevel"] = leafCtg->YLevel(),
				_["weight"] = leafCtg->WeightTree()
				);
  outBundle.attr("class") = "ExportCtg";
  return outBundle;

  END_RCPP
}


/**
   @brief Only the scores are of interest to ForestFloor.

   @param forestCore is the exported core image.

   @param tIdx is the tree index.

   @return Vector of score values.
 */
SEXP ExportBridge::FFloorLeafReg(List &forestCore,
				 unsigned int tIdx) {
  BEGIN_RCPP
  const vector<vector<double> > &score = forestCore["score"];
  List ffLeaf = List::create(
     _["score"] = score[tIdx]
    );

  ffLeaf.attr("class") = "FFloorLeafReg";
  return ffLeaf;
  END_RCPP
}


/**
   @brief Only the scores and weights are of interest to ForestFloor.

   @param forestCore is the exported core image.

   @param tIdx is the tree index.

   @return Vector of score values.
 */
SEXP ExportBridge::FFloorLeafCtg(List &forestCore,
				 unsigned int tIdx) {
  BEGIN_RCPP
  const vector<vector<double> > &score = forestCore["score"];
  const vector<vector<double> > &weight = forestCore["weight"];
  unsigned int leafCount = score[tIdx].size();
  NumericMatrix weightOut = NumericMatrix(weight[tIdx].size() / leafCount, leafCount, weight[tIdx].begin());
  List ffLeaf = List::create(
     _["score"] = score[tIdx],
     _["weight"] = transpose(weightOut)
     );

  ffLeaf.attr("class") = "FFloorLeafCtg";
  return ffLeaf;
  END_RCPP
}


/**
 */
SEXP ExportBridge::FFloorInternal(List &forestCore,
				 unsigned int tIdx) {
  BEGIN_RCPP
  const vector<vector<unsigned int> > &predTree = forestCore["pred"];
  const vector<vector<unsigned int> > &bumpTree = forestCore["lhDel"];
  const vector<vector<double > > &splitTree = forestCore["split"];
  const vector<vector<unsigned int> > &facSplitTree = forestCore["facSplit"];
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
  END_RCPP
}


SEXP ExportBridge::FFloorBag(List &forestCore,
			    int tIdx) {
  BEGIN_RCPP
  const vector<vector<unsigned int> > &rowTree = forestCore["row"];
  const vector<vector<unsigned int> > &sCountTree = forestCore["sCount"];
  IntegerVector row(rowTree[tIdx].begin(), rowTree[tIdx].end());
  IntegerVector sCount(sCountTree[tIdx].begin(), sCountTree[tIdx].end());
  IntegerVector bag = IntegerVector(as<unsigned int>(forestCore["rowTrain"]), 0);
  bag[row] = sCount;

  return bag;
  END_RCPP
}


/**
 */
List ExportBridge::FFloorTreeReg(SEXP sForest,
				 const List &leaf,
				 IntegerVector &predMap) {
  BEGIN_RCPP

  unsigned int nTree;
  List coreReg = ExportReg(sForest, leaf, predMap, nTree);
    
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    List ffReg = List::create(
			      _["internal"] = FFloorInternal(coreReg, tIdx),
			      _["leaf"] = FFloorLeafReg(coreReg, tIdx),
			      _["bag"] = FFloorBag(coreReg, tIdx)
			      );
      ffReg.attr("class") = "FFloorTreeReg";
      trees[tIdx] = ffReg;
  }
  return trees;

  END_RCPP
}


/**
 */
List ExportBridge::FFloorTreeCtg(List &coreCtg,
				 unsigned int nTree) {
  BEGIN_RCPP

    List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    List ffCtg = List::create(
			      _["internal"] = FFloorInternal(coreCtg, tIdx),
			      _["leaf"] = FFloorLeafCtg(coreCtg, tIdx),
			      _["bag"] = FFloorBag(coreCtg, tIdx)
			      );
    ffCtg.attr("class") = "FFloorTreeCtg";
    trees[tIdx] = ffCtg;
  }
  return trees;

  END_RCPP
}


/**
 */
SEXP ExportBridge::FFloorReg(SEXP sForest,
			     List &leaf,
			     IntegerVector &predMap,
			     List &predLevel) {
  BEGIN_RCPP

  int facCount = predLevel.length();
  IntegerVector facMap(predMap.end() - facCount, predMap.end());
  List ffe = List::create(
			  _["facMap"] = facMap,
			  _["predLevel"] = predLevel,
			  _["tree"] = FFloorTreeReg(sForest, leaf, predMap)
			  );
  ffe.attr("class") = "ForestFloorReg";
  return ffe;
  END_RCPP
}


/**
 */
SEXP ExportBridge::FFloorCtg(SEXP sForest,
			     List &leaf,
			     IntegerVector &predMap,
			     List &predLevel) {
  BEGIN_RCPP

  unsigned int nTree;
  List coreCtg = ExportCtg(sForest, leaf, predMap, nTree);

  int facCount = predLevel.length();
  IntegerVector facMap(predMap.end() - facCount, predMap.end());
  List ffe = List::create(
   _["facMap"] = facMap,
   _["predLevel"] = predLevel,
   _["yLevel"] = as<CharacterVector>(coreCtg["yLevel"]),
   _["tree"] = FFloorTreeCtg(coreCtg, nTree)
  );
  ffe.attr("class") = "ForestFloorCtg";
  return ffe;

  END_RCPP
}


