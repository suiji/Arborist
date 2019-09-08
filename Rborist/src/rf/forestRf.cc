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
   @file forestRf.cc

   @brief C++ interface to R entry for Forest methods.

   @author Mark Seligman
 */

#include "forestRf.h"
#include "forestbridge.h"
#include "trainbridge.h"


FBTrain::FBTrain(unsigned int nTree) :
  nodeRaw(RawVector(0)),
  height(IntegerVector(nTree)),
  facRaw(RawVector(0)),
  facHeight(IntegerVector(nTree)) {
}


void FBTrain::consume(const TrainChunk* train,
                      unsigned int tIdx,
                      double scale) {
  unsigned int i = tIdx;
  for (auto th : train->getForestHeight()) {
    height[i++] = th + (tIdx == 0 ? 0 : height[tIdx-1]);
  }

  i = tIdx;
  for (auto fo : train->getFactorHeight()) {
    facHeight[i++] = fo + (tIdx == 0 ? 0 : facHeight[tIdx-1]);
  }

  size_t nodeOff = tIdx == 0 ? 0 : height[tIdx-1] * ForestBridge::nodeSize();
  size_t nodeBytes = train->getForestHeight().back() * ForestBridge::nodeSize();
  if (nodeOff + nodeBytes > static_cast<size_t>(nodeRaw.length())) {
    RawVector temp(scale * (nodeOff + nodeBytes));
    for (size_t i = 0; i < nodeOff; i++)
      temp[i] = nodeRaw[i];
    nodeRaw = move(temp);
  }
  train->dumpTreeRaw(&nodeRaw[nodeOff]);

  size_t facOff = tIdx == 0 ? 0 : facHeight[tIdx-1] * sizeof(unsigned int);
  size_t facBytes = train->getFactorHeight().back() * sizeof(unsigned int);
  if (facOff + facBytes > static_cast<size_t>(facRaw.length())) {
    RawVector temp(scale * (facOff + facBytes));
    for (size_t i = 0; i < facOff; i++)
      temp[i] = facRaw[i];
    facRaw = move(temp);
  }
  train->dumpFactorRaw(&facRaw[facOff]);
}


List FBTrain::wrap() {
  BEGIN_RCPP
  List forest =
    List::create(
                 _["forestNode"] = move(nodeRaw),
                 _["height"] = move(height),
                 _["facHeight"] = move(facHeight),
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
  return make_unique<ForestBridge>((unsigned int*) IntegerVector((SEXP) lForest["height"]).begin(),
                                   (size_t) IntegerVector((SEXP) lForest["height"]).length(),
                                   RawVector((SEXP) lForest["forestNode"]).begin(),
                                   (unsigned int*) RawVector((SEXP) lForest["facSplit"]).begin(),
                                   (unsigned int*) IntegerVector((SEXP) lForest["facHeight"]).begin());
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
