// Copyright (C)  2012-2019   Mark Seligman
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
#include "bagBridge.h"
#include "framemapBridge.h"
#include "forestBridge.h"
#include "forest.h"
#include "leafBridge.h"
#include "leaf.h"
#include "bv.h"
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
  FramemapBridge::signatureUnwrap(arbOut, predMap, predLevel);

  List leaf((SEXP) arbOut["leaf"]);
  if (leaf.inherits("LeafReg"))  {
    return ExportBridge::fFloorReg(arbOut, predMap, predLevel);
  }
  else if (leaf.inherits("LeafCtg")) {
    return ExportBridge::fFloorCtg(arbOut, predMap, predLevel);
  }
  else {
    warning("Unrecognized forest type.");
    return List::create(0);
  }

  END_RCPP
}


/**
 */
List ExportBridge::fFloorForest(const ForestExport *forest,
                                unsigned int tIdx) {
  BEGIN_RCPP

  const vector<unsigned int> &predTree = forest->getPredTree(tIdx);
  const vector<unsigned int> &bumpTree = forest->getBumpTree(tIdx);
  IntegerVector incrL(bumpTree.begin(), bumpTree.end());
  IntegerVector predIdx(predTree.begin(), predTree.end());
  List ffTree = List::create(
     _["pred"] = ifelse(incrL == 0, -(predIdx + 1), predIdx),
     _["daughterL"] = incrL,
     _["daughterR"] = ifelse(incrL == 0, 0, incrL + 1),
     _["split"] = forest->getSplitTree(tIdx),
     _["facSplit"] = forest->getFacSplitTree(tIdx)
     );

  ffTree.attr("class") = "fFloorTree";
  return ffTree;
  END_RCPP
}


IntegerVector ExportBridge::fFloorBag(const LeafBridge *leaf,
                                      unsigned int tIdx,
                                      unsigned int rowTrain) {
  BEGIN_RCPP

  vector<unsigned int> rowTree = leaf->getRowTree(tIdx);
  vector<unsigned int> sCountTree = leaf->getSCountTree(tIdx);

  IntegerVector row(rowTree.begin(), rowTree.end());
  IntegerVector sCount(sCountTree.begin(), sCountTree.end());
  IntegerVector bag = IntegerVector(rowTrain, 0);
  bag[row] = sCount;

  return bag;
  END_RCPP
}


/**
   @brief Only the scores are of interest to ForestFloor.

   @param forestCore is the exported core image.

   @param tIdx is the tree index.

   @return Vector of score values.
 */
List ExportBridge::fFloorLeafReg(const LeafRegBridge *leaf, unsigned int tIdx) {
  BEGIN_RCPP

  const vector<double> &score = leaf->getScoreTree(tIdx);
  List ffLeaf = List::create(
     _["score"] = score[tIdx]
    );

  ffLeaf.attr("class") = "fFloorLeafReg";
  return ffLeaf;

  END_RCPP
}


/**
 */
List ExportBridge::fFloorTreeCtg(const ForestExport *forest,
                                 const LeafCtgBridge *leaf,
                                 unsigned int rowTrain) {
  BEGIN_RCPP

  auto nTree = forest->getNTree();
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    List ffCtg = List::create(
                              _["internal"] = fFloorForest(forest, tIdx),
                              _["leaf"] = fFloorLeafCtg(leaf, tIdx),
                              _["bag"] = fFloorBag(leaf, tIdx, rowTrain)
                              );
    ffCtg.attr("class") = "fFloorTreeCtg";
    trees[tIdx] = move(ffCtg);
  }
  return trees;

  END_RCPP
}


/**
   @brief Only the scores and weights are of interest to ForestFloor.

   @param forestCore is the exported core image.

   @param tIdx is the tree index.

   @return Vector of score values.
 */
List ExportBridge::fFloorLeafCtg(const LeafCtgBridge *leaf,
                                 unsigned int tIdx) {
  BEGIN_RCPP

  const vector<double> &score = leaf->getScoreTree(tIdx);
  const vector<double> &weight = leaf->getWeightTree(tIdx);
  unsigned int leafCount = score.size();
  NumericMatrix weightOut = NumericMatrix(weight.size() / leafCount, leafCount, weight.begin());
  List ffLeaf = List::create(
                           _["score"] = score,
                           _["weight"] = transpose(weightOut)
                             );

  ffLeaf.attr("class") = "fFloorLeafCtg";
  return ffLeaf;
  END_RCPP
}


/**
 */
List ExportBridge::fFloorReg(const List &lTrain,
                             IntegerVector &predMap,
                             List &predLevel) {
  BEGIN_RCPP

  int facCount = predLevel.length();
  IntegerVector facMap(predMap.end() - facCount, predMap.end());
  List ffe = List::create(
                          _["facMap"] = facMap,
                          _["predLevel"] = predLevel,
                          _["tree"] = fFloorTreeReg(lTrain, predMap)
                          );
  ffe.attr("class") = "ForestFloorReg";
  return ffe;

  END_RCPP
}


/**
 */
List ExportBridge::fFloorTreeReg(const List &lTrain,
                                 IntegerVector &predMap) {
  BEGIN_RCPP

    auto bag = BagBridge::unwrap(lTrain);
  auto leaf = LeafRegBridge::unwrap(lTrain, bag->getRaw());
  auto forest = ForestExport::unwrap(lTrain, predMap);

  auto nTree = bag->getNTree();
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    List ffReg = List::create(
                              _["internal"] = fFloorForest(forest.get(), tIdx),
                              _["leaf"] = fFloorLeafReg(leaf.get(), tIdx),
                              _["bag"] = fFloorBag(leaf.get(), tIdx, bag->getNRow())
                              );
      ffReg.attr("class") = "fFloorTreeReg";
      trees[tIdx] = move(ffReg);
  }
  return trees;

  END_RCPP
}


/**
 */
List ExportBridge::fFloorCtg(const List &lTrain,
                             IntegerVector &predMap,
                             List &predLevel) {
  BEGIN_RCPP
    auto bag = BagBridge::unwrap(lTrain);
  auto leaf = LeafCtgBridge::unwrap(lTrain, bag->getRaw());
  auto forest = ForestExport::unwrap(lTrain, predMap);
  int facCount = predLevel.length();
  IntegerVector facMap(predMap.end() - facCount, predMap.end());
  List ffe = List::create(
   _["facMap"] = facMap,
   _["predLevel"] = predLevel,
   _["yLevel"] = leaf->getLevelsTrain(),
   _["tree"] = fFloorTreeCtg(forest.get(), leaf.get(), bag->getNRow())
  );
  ffe.attr("class") = "ForestFloorCtg";
  return ffe;

  END_RCPP
}


