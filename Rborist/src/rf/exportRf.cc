// Copyright (C)  2012-2019   Mark Seligman
//
// This file is part of rfR.
//
// rfR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rfR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file exportRf.cc

   @brief C++ interface to R entry for export methods.

   @author Mark Seligman
 */

#include "exportRf.h"
#include "bagRf.h"
#include "bagbridge.h"
#include "signature.h"
#include "forestRf.h"
#include "forestbridge.h"
#include "leafRf.h"
#include "leafbridge.h"

#include <vector>


/**
   @brief Structures forest summary for analysis by Export package.

   @param sForest is the Forest summary.

   @return RboristExport as List.
 */
RcppExport SEXP Export(SEXP sArbOut) {
  BEGIN_RCPP
    
  List arbOut(sArbOut);
  if (!arbOut.inherits("Rborist")) {
    warning("Expecting an Rborist object");
    return List::create(0);
  }

  IntegerVector predMap;
  List predLevel;
  Signature::unwrapExport(arbOut, predMap, predLevel);

  List leaf((SEXP) arbOut["leaf"]);
  if (leaf.inherits("LeafReg"))  {
    return ExportRf::exportReg(arbOut, predMap, predLevel);
  }
  else if (leaf.inherits("LeafCtg")) {
    return ExportRf::exportCtg(arbOut, predMap, predLevel);
  }
  else {
    warning("Unrecognized forest type.");
    return List::create(0);
  }

  END_RCPP
}


/**
 */
List ExportRf::exportForest(const ForestExport *forest,
                            unsigned int tIdx) {
  BEGIN_RCPP

  auto predTree(forest->getPredTree(tIdx));
  auto bumpTree(forest->getBumpTree(tIdx));
  IntegerVector incrL(bumpTree.begin(), bumpTree.end());
  IntegerVector predIdx(predTree.begin(), predTree.end());
  List ffTree = List::create(
     _["pred"] = ifelse(incrL == 0, -(predIdx + 1), predIdx),
     _["daughterL"] = incrL,
     _["daughterR"] = ifelse(incrL == 0, 0, incrL + 1),
     _["split"] = forest->getSplitTree(tIdx),
     _["facSplit"] = forest->getFacSplitTree(tIdx)
     );

  ffTree.attr("class") = "exportTree";
  return ffTree;
  END_RCPP
}


IntegerVector ExportRf::exportBag(const LeafExport* leaf,
                                  unsigned int tIdx,
                                  unsigned int rowTrain) {
  BEGIN_RCPP

  auto rowTree(leaf->getRowTree(tIdx));
  auto sCountTree(leaf->getSCountTree(tIdx));

  IntegerVector row(rowTree.begin(), rowTree.end());
  IntegerVector sCount(sCountTree.begin(), sCountTree.end());
  IntegerVector bag(rowTrain);

  bag[row] = sCount;

  return bag;
  END_RCPP
}


/**
   @brief Only the scores are of interest to Export.

   @param forestCore is the exported core image.

   @param tIdx is the tree index.

   @return Vector of score values.
 */
List ExportRf::exportLeafReg(const LeafExportReg* leaf, unsigned int tIdx) {
  BEGIN_RCPP

  auto score(leaf->getScoreTree(tIdx));
  List ffLeaf = List::create(
                             _["score"] = score
                             );

  ffLeaf.attr("class") = "exportLeafReg";
  return ffLeaf;

  END_RCPP
}


/**
 */
List ExportRf::exportTreeCtg(const ForestExport* forest,
                             const LeafExportCtg* leaf,
                             unsigned int rowTrain) {
  BEGIN_RCPP

  auto nTree = forest->getNTree();
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    List ffCtg =
      List::create(
                   _["internal"] = exportForest(forest, tIdx),
                   _["leaf"] = exportLeafCtg(leaf, tIdx),
                   _["bag"] = exportBag(leaf, tIdx, rowTrain)
                   );
    ffCtg.attr("class") = "exportTreeCtg";
    trees[tIdx] = move(ffCtg);
  }
  return trees;

  END_RCPP
}


/**
   @brief Only the scores and weights are of interest to Export.

   @param forestCore is the exported core image.

   @param tIdx is the tree index.

   @return Vector of score values.
 */
List ExportRf::exportLeafCtg(const LeafExportCtg* leaf,
                             unsigned int tIdx) {
  BEGIN_RCPP

  auto score(leaf->getScoreTree(tIdx));
  auto weight(leaf->getWeightTree(tIdx));
  unsigned int leafCount = score.size();
  NumericMatrix weightOut = NumericMatrix(weight.size() / leafCount, leafCount, weight.begin());
  List ffLeaf =
    List::create(
                 _["score"] = score,
                 _["weight"] = transpose(weightOut)
                 );

  ffLeaf.attr("class") = "exportLeafCtg";
  return ffLeaf;
  END_RCPP
}


/**
 */
List ExportRf::exportReg(const List& lArb,
                         const IntegerVector& predMap,
                         const List& predLevel) {
  BEGIN_RCPP

  int facCount = predLevel.length();
  List ffe =
    List::create(
                 _["facMap"] = IntegerVector(predMap.end() - facCount, predMap.end()),
                 _["predLevel"] = predLevel,
                 _["tree"] = exportTreeReg(lArb, predMap)
                 );
  ffe.attr("class") = "ExportReg";
  return ffe;

  END_RCPP
}


/**
 */
List ExportRf::exportTreeReg(const List& lArb,
                             const IntegerVector& predMap) {
  BEGIN_RCPP

  auto bag(BagRf::unwrap(lArb));
  auto leaf(LeafExportReg::unwrap(lArb, bag.get()));
  auto forest(ForestExport::unwrap(lArb, predMap));

  auto nTree = bag->getNTree();
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    List ffReg =
      List::create(
                   _["internal"] = exportForest(forest.get(), tIdx),
                   _["leaf"] = exportLeafReg(leaf.get(), tIdx),
                   _["bag"] = exportBag(leaf.get(), tIdx, bag->getNObs())
                   );
      ffReg.attr("class") = "exportTreeReg";
      trees[tIdx] = move(ffReg);
  }
  return trees;

  END_RCPP
}


/**
 */
List ExportRf::exportCtg(const List& lArb,
                         const IntegerVector& predMap,
                         const List& predLevel) {
  BEGIN_RCPP

  auto bag(BagRf::unwrap(lArb));
  auto leaf(LeafExportCtg::unwrap(lArb, bag.get()));
  auto forest(ForestExport::unwrap(lArb, predMap));
  int facCount = predLevel.length();
  List ffe =
    List::create(
                 _["facMap"] = IntegerVector(predMap.end() - facCount, predMap.end()),
                 _["predLevel"] = predLevel,
                 _["yLevel"] = leaf->getLevelsTrain(),
                 _["tree"] = exportTreeCtg(forest.get(), leaf.get(), bag->getNObs())
  );
  ffe.attr("class") = "ExportCtg";
  return ffe;

  END_RCPP
}


unique_ptr<LeafExportCtg> LeafExportCtg::unwrap(const List &lTrain,
                                                const BagBridge* bag) {
  List lLeaf(LeafCtgRf::checkLeaf(lTrain));
  return make_unique<LeafExportCtg>(lLeaf, bag);
}

LeafExport::LeafExport(unsigned int nTree_) :
  nTree(nTree_),
  rowTree(vector<vector<size_t> >(nTree)),
  sCountTree(vector<vector<unsigned int> >(nTree)),
  extentTree(vector<vector<unsigned int> >(nTree)) {
}


/**
   @brief Constructor caches front-end vectors and instantiates a Leaf member.
 */
LeafExportCtg::LeafExportCtg(const List& lLeaf,
                             const BagBridge* bagBridge) :
  LeafExport((unsigned int) IntegerVector((SEXP) lLeaf["nodeHeight"]).length()),
  levelsTrain(CharacterVector((SEXP) lLeaf["levels"])),
  scoreTree(vector<vector<double > >(nTree)),
  weightTree(vector<vector<double> >(nTree)) {
  unique_ptr<LeafCtgBridge>  leaf =
    make_unique<LeafCtgBridge>((unsigned int*) IntegerVector((SEXP) lLeaf["nodeHeight"]).begin(),
                               nTree,
                               (unsigned char*) RawVector((SEXP) lLeaf["node"]).begin(),
                               (unsigned int*) IntegerVector((SEXP) lLeaf["bagHeight"]).begin(),
                               (unsigned char*) RawVector((SEXP) lLeaf["bagSample"]).begin(),
                               (double*) NumericVector((SEXP) lLeaf["weight"]).begin(),
                               (unsigned int) CharacterVector((SEXP) lLeaf["levels"]).length(),
                               0,
                               false);
  leaf->dump(bagBridge, rowTree, sCountTree, scoreTree, extentTree, weightTree);
}


unique_ptr<LeafExportReg> LeafExportReg::unwrap(const List& lTrain,
                                                const BagBridge *bag) {
  List lLeaf(LeafRegRf::checkLeaf(lTrain));
  return make_unique<LeafExportReg>(lLeaf, bag);
}
 

/**
   @brief Constructor instantiates leaves for export only:
   no prediction.
 */
LeafExportReg::LeafExportReg(const List& lLeaf,
                             const BagBridge* bagBridge) :
  LeafExport((unsigned int) IntegerVector((SEXP) lLeaf["nodeHeight"]).length()),
  yTrain(NumericVector((SEXP) lLeaf["yTrain"])),
  scoreTree(vector<vector<double > >(nTree)) {
  unique_ptr<LeafRegBridge> leaf =
    make_unique<LeafRegBridge>((unsigned int*) IntegerVector((SEXP) lLeaf["nodeHeight"]).begin(),
                               (unsigned int) IntegerVector((SEXP) lLeaf["nodeHeight"]).length(),
                               (unsigned char*) RawVector((SEXP) lLeaf["node"]).begin(),
                               (unsigned int*) IntegerVector((SEXP) lLeaf["bagHeight"]).begin(),
                               (unsigned char*) RawVector((SEXP) lLeaf["bagSample"]).begin(),
                               (double*) yTrain.begin(),
                               (size_t) yTrain.length(),
                               mean(yTrain),
                               0);
  leaf->dump(bagBridge, rowTree, sCountTree, scoreTree, extentTree);
}

