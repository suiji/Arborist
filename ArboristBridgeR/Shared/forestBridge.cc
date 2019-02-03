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
   @file forestBridge.cc

   @brief C++ interface to R entry for Forest methods.

   @author Mark Seligman
 */

#include "forestBridge.h"
#include "forest.h"

void FBTrain::consume(const ForestTrain* forestTrain,
                      unsigned int tIdx,
                      double scale) {
  unsigned int i = tIdx;
  for (auto th : forestTrain->getNodeHeight()) {
    height[i++] = th + (tIdx == 0 ? 0 : height[tIdx-1]);
  }

  i = tIdx;
  for (auto fo : forestTrain->getFacHeight()) {
    facHeight[i++] = fo + (tIdx == 0 ? 0 : facHeight[tIdx-1]);
  }

  size_t nodeOff = tIdx == 0 ? 0 : height[tIdx-1] * sizeof(TreeNode);
  size_t nodeBytes = forestTrain->getNodeHeight().back() * sizeof(TreeNode);
  if (nodeOff + nodeBytes > static_cast<size_t>(nodeRaw.length())) {
    RawVector temp(scale * (nodeOff + nodeBytes));
    for (size_t i = 0; i < nodeOff; i++)
      temp[i] = nodeRaw[i];
    nodeRaw = move(temp);
  }
  forestTrain->cacheNodeRaw(&nodeRaw[nodeOff]);

  size_t facOff = tIdx == 0 ? 0 : facHeight[tIdx-1] * sizeof(unsigned int);
  size_t facBytes = forestTrain->getFacHeight().back() * sizeof(unsigned int);
  if (facOff + facBytes > static_cast<size_t>(facRaw.length())) {
    RawVector temp(scale * (facOff + facBytes));
    for (size_t i = 0; i < facOff; i++)
      temp[i] = facRaw[i];
    facRaw = move(temp);
  }
  forestTrain->cacheFacRaw(&facRaw[facOff]);
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


unique_ptr<ForestBridge> ForestBridge::unwrap(const List& lTrain) {
  List lForest = checkForest(lTrain);
  return make_unique<ForestBridge>(IntegerVector((SEXP) lForest["height"]),
                                   RawVector((SEXP) lForest["facSplit"]),
                                   IntegerVector((SEXP) lForest["facHeight"]),
                                   RawVector((SEXP) lForest["forestNode"]));
}


SEXP ForestBridge::checkForest(const List &lTrain) {
  BEGIN_RCPP

  List lForest = List((SEXP) lTrain["forest"]);
  if (!lForest.inherits("Forest")) {
    stop("Expecting Forest");
  }
  return lForest;
  
  END_RCPP
}


unique_ptr<ForestExport> ForestExport::unwrap(const List &lTrain,
                                              IntegerVector &predMap) {
  List lForest = checkForest(lTrain);
  return make_unique<ForestExport>(lForest, predMap);
}


ForestExport::ForestExport(List &lForest,
                           IntegerVector &predMap) :
  ForestBridge(IntegerVector((SEXP) lForest["height"]),
               RawVector((SEXP) lForest["facSplit"]),
               IntegerVector((SEXP) lForest["facHeight"]),
               RawVector((SEXP) lForest["forestNode"])),
  predTree(vector<vector<unsigned int> >(getNTree())),
  bumpTree(vector<vector<unsigned int> >(getNTree())),
  splitTree(vector<vector<double > >(getNTree())),
  facSplitTree(vector<vector<unsigned int> >(getNTree())) {
  forest->dump(predTree, splitTree, bumpTree, facSplitTree);
  predExport(predMap.begin());
}


// Alignment should be sufficient to guarantee safety of
// the casted loads.
ForestBridge::ForestBridge(const IntegerVector& feHeight_,
                           const RawVector &feFacSplit_,
                           const IntegerVector &feFacHeight_,
                           const RawVector &feNode_) :
  feHeight(feHeight_),
  feNode(feNode_),
  feFacHeight(feFacHeight_),
  feFacSplit(feFacSplit_),
  forest(move(make_unique<Forest>((unsigned int*) &feHeight[0],
                                  feHeight.length(),
                                  (TreeNode*) &feNode[0],
                                  (unsigned int *) &feFacSplit[0],
                                  (unsigned int*) &feFacHeight[0]))) {
}


FBTrain::FBTrain(unsigned int nTree) :
  nodeRaw(RawVector(0)),
  facRaw(RawVector(0)),
  height(IntegerVector(nTree)),
  facHeight(IntegerVector(nTree)) {
}
