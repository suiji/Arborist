// Copyright (C)  2012-2021   Mark Seligman
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
   @file forestR.cc

   @brief C++ interface to R entry for Forest methods.

   @author Mark Seligman
 */

#include "forestR.h"
#include "forestbridge.h"


FBTrain::FBTrain(unsigned int nTree_) :
  nTree(nTree_),
  rawTop(0),
  nodeRaw(RawVector(0)),
  facTop(0),
  facRaw(RawVector(0)) {
}


void FBTrain::consume(const ForestBridge* fb,
                      unsigned int tIdx,
                      double scale) {
  size_t nodeBytes = fb->getNodeBytes();  // # bytes in node chunk.
  if (rawTop + nodeBytes > static_cast<size_t>(nodeRaw.length())) {
    nodeRaw = move(resizeRaw(&nodeRaw[0], rawTop, nodeBytes, scale));
  }
  fb->dumpTreeRaw(&nodeRaw[rawTop]);
  rawTop += nodeBytes;

  size_t facBytes = fb->getFactorBytes();
  if (facTop + facBytes > static_cast<size_t>(facRaw.length())) {
    facRaw = move(resizeRaw(&facRaw[0], facTop, facBytes, scale));
  }
  fb->dumpFactorRaw(&facRaw[facTop]);
  facTop += facBytes;
}


RawVector FBTrain::resizeRaw(const unsigned char* raw, size_t offset, size_t bytes, double scale) { // Assumes scale >= 1.0.
  RawVector temp(scale * (offset + bytes));
  for (size_t i = 0; i < offset; i++)
    temp[i] = raw[i];

  return temp;
}


List FBTrain::wrap() {
  BEGIN_RCPP
  List forest =
    List::create(_["nTree"] = nTree,
                 _["forestNode"] = move(nodeRaw),
                 _["facSplit"] = move(facRaw)
                 );
  nodeRaw = RawVector(0);
  facRaw = RawVector(0);
  forest.attr("class") = "Forest";

  return forest;
  END_RCPP
}


unique_ptr<ForestBridge> ForestRf::unwrap(const List& lTrain) {
  List lForest(checkForest(lTrain));
  return make_unique<ForestBridge>(as<unsigned int>(lForest["nTree"]),
				   as<RawVector>(lForest["forestNode"]).begin(),
				   as<RawVector>(lForest["facSplit"]).begin());
}


List ForestRf::checkForest(const List& lTrain) {
  BEGIN_RCPP

  List lForest((SEXP) lTrain["forest"]);
  if (!lForest.inherits("Forest")) {
    stop("Expecting Forest");
  }
  return lForest;
  
  END_RCPP
}


unique_ptr<ForestExport> ForestExport::unwrap(const List& lTrain,
                                              const IntegerVector& predMap) {
  (void) ForestRf::checkForest(lTrain);
  return make_unique<ForestExport>(lTrain, predMap);
}


ForestExport::ForestExport(const List &lTrain,
                           const IntegerVector &predMap) :
  forestBridge(ForestRf::unwrap(lTrain)),
  predTree(vector<vector<unsigned int> >(forestBridge->getNTree())),
  bumpTree(vector<vector<unsigned int> >(forestBridge->getNTree())),
  splitTree(vector<vector<double > >(forestBridge->getNTree())),
  facSplitTree(vector<vector<unsigned int> >(forestBridge->getNTree())) {
  forestBridge->dump(predTree, splitTree, bumpTree, facSplitTree);
  predExport(predMap.begin());
}


unsigned int ForestExport::getNTree() const {
  return forestBridge->getNTree();
}


/**
   @brief Recasts 'pred' field of nonterminals to front-end facing values.

   @return void.
 */
void ForestExport::treeExport(const int predMap[],
                            vector<unsigned int> &pred,
                            const vector<unsigned int> &bump) {
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
void ForestExport::predExport(const int predMap[]) {
  for (unsigned int tIdx = 0; tIdx < predTree.size(); tIdx++) {
    treeExport(predMap, predTree[tIdx], bumpTree[tIdx]);
  }
}
