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


const string FBTrain::strNTree = "nTree";
const string FBTrain::strNodeExtent = "nodeExtent";
const string FBTrain::strForestNode = "forestNode";
const string FBTrain::strScores= "scores";
const string FBTrain::strFacExtent = "facExtent";
const string FBTrain::strFacSplit = "facSplit";


FBTrain::FBTrain(unsigned int nTree_) :
  nTree(nTree_),
  nodeExtent(NumericVector(nTree)),
  nodeTop(0),
  scores(NumericVector(0)),
  facExtent(NumericVector(nTree)),
  facTop(0),
  facRaw(RawVector(0)) {
}


void FBTrain::bridgeConsume(const ForestBridge& bridge,
			    unsigned int tIdx,
			    double scale) {
  const vector<size_t>&nExtents = bridge.getNodeExtents();
  unsigned int fromIdx = 0;
  for (unsigned int toIdx = tIdx; toIdx < tIdx + nExtents.size(); toIdx++) {
    nodeExtent[toIdx] = nExtents[fromIdx++];
  }

  size_t nodeCount = bridge.getNodeCount();
  if (nodeTop + nodeCount > static_cast<size_t>(cNode.length())) {
    cNode = move(ResizeR::resizeComplex(cNode, nodeTop, nodeCount, scale));
    scores = move(ResizeR::resizeNum(scores, nodeTop, nodeCount, scale));
  }
  bridge.dumpTree((complex<double>*)&cNode[nodeTop]);
  bridge.dumpScore(&scores[nodeTop]);
  nodeTop += nodeCount;

  const vector<size_t>& fExtents = bridge.getFacExtents();
  fromIdx = 0;
  for (unsigned int toIdx = tIdx; toIdx < tIdx + fExtents.size(); toIdx++) {
    facExtent[toIdx] = fExtents[fromIdx++];
  }
 
  size_t facBytes = bridge.getFactorBytes();
  if (facTop + facBytes > static_cast<size_t>(facRaw.length())) {
    facRaw = move(ResizeR::resizeRaw(facRaw, facTop, facBytes, scale));
  }
  bridge.dumpFactorRaw(&facRaw[facTop]);
  facTop += facBytes;
}



List FBTrain::wrap() {
  BEGIN_RCPP
  List forest =
    List::create(_[strNTree] = nTree,
		 _[strNodeExtent] = move(nodeExtent),
		 _[strForestNode] = move(cNode),
		 _[strScores] = move(scores),
		 _[strFacExtent] = move(facExtent),
                 _[strFacSplit] = move(facRaw)
                 );
  cNode = ComplexVector(0);
  scores = NumericVector(0);
  facRaw = RawVector(0);
  forest.attr("class") = "Forest";

  return forest;
  END_RCPP
}


unique_ptr<ForestBridge> ForestRf::unwrap(const List& lTrain) {
  List lForest(checkForest(lTrain));
  return make_unique<ForestBridge>(as<unsigned int>(lForest[FBTrain::strNTree]),
				   as<NumericVector>(lForest[FBTrain::strNodeExtent]).begin(),
				   (complex<double>*) as<ComplexVector>(lForest[FBTrain::strForestNode]).begin(),
				   as<NumericVector>(lForest[FBTrain::strScores]).begin(),
				   as<NumericVector>(lForest[FBTrain::strFacExtent]).begin(),
				   as<RawVector>(lForest[FBTrain::strFacSplit]).begin());
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
