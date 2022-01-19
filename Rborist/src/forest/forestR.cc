// Copyright (C)  2012-2022   Mark Seligman
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

#include "resizeR.h"
#include "forestR.h"
#include "forestbridge.h"


FBTrain::FBTrain(unsigned int nTree_) :
  nTree(nTree_),
  nodeExtent(NumericVector(nTree)),
  nodeTop(0),
  nodeRaw(RawVector(0)),
  scoreTop(0),
  scores(NumericVector(0)),
  facExtent(NumericVector(nTree)),
  facTop(0),
  facRaw(RawVector(0)) {
}


void FBTrain::bridgeConsume(const ForestBridge* bridge,
			    unsigned int tIdx,
			    double scale) {
  const vector<size_t>&nExtents = bridge->getNodeExtents();
  unsigned int fromIdx = 0;
  for (unsigned int toIdx = tIdx; toIdx < tIdx + nExtents.size(); toIdx++) {
    nodeExtent[toIdx] = nExtents[fromIdx++];
  }

  size_t nodeBytes = bridge->getNodeBytes();  // # bytes in node chunk.
  if (nodeTop + nodeBytes > static_cast<size_t>(nodeRaw.length())) {
    nodeRaw = move(ResizeR::resizeRaw(nodeRaw, nodeTop, nodeBytes, scale));
  }
  bridge->dumpTreeRaw(&nodeRaw[nodeTop]);
  nodeTop += nodeBytes;

  size_t scoreSize = bridge->getScoreSize();
  if (scoreTop + scoreSize > static_cast<size_t>(scores.length())) {
    scores = move(ResizeR::resizeNum(scores, scoreTop, scoreSize, scale));
  }
  bridge->dumpScore(&scores[scoreTop]);
  scoreTop += scoreSize;

  const vector<size_t>&fExtents = bridge->getFacExtents();
  fromIdx = 0;
  for (unsigned int toIdx = tIdx; toIdx < tIdx + fExtents.size(); toIdx++) {
    facExtent[toIdx] = fExtents[fromIdx++];
  }

  size_t facBytes = bridge->getFactorBytes();
  if (facTop + facBytes > static_cast<size_t>(facRaw.length())) {
    facRaw = move(ResizeR::resizeRaw(facRaw, facTop, facBytes, scale));
  }
  bridge->dumpFactorRaw(&facRaw[facTop]);
  facTop += facBytes;
}



List FBTrain::wrap() {
  BEGIN_RCPP
  List forest =
    List::create(_["nTree"] = nTree,
		 _["nodeExtent"] = move(nodeExtent),
                 _["forestNode"] = move(nodeRaw),
		 _["scores"] = move(scores),
		 _["facExtent"] = move(facExtent),
                 _["facSplit"] = move(facRaw)
                 );
  nodeRaw = RawVector(0);
  scores = NumericVector(0);
  facRaw = RawVector(0);
  forest.attr("class") = "Forest";

  return forest;
  END_RCPP
}


unique_ptr<ForestBridge> ForestRf::unwrap(const List& lTrain) {
  List lForest(checkForest(lTrain));
  return make_unique<ForestBridge>(as<unsigned int>(lForest["nTree"]),
				   as<NumericVector>(lForest["nodeExtent"]).begin(),
				   as<RawVector>(lForest["forestNode"]).begin(),
				   as<NumericVector>(lForest["scores"]).begin(),
				   as<NumericVector>(lForest["facExtent"]).begin(),
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
