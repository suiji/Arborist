// Copyright (C)  2012-2023   Mark Seligman
//
// This file is part of RboristBase.
//
// RboristBase is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RboristBase is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RboristBase.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file forestR.cc

   @brief C++ interface to R entry for Forest methods.

   @author Mark Seligman
 */

#include "resizeR.h"
#include "forestR.h"
#include "grovebridge.h"
#include "forestbridge.h"
#include "trainbridge.h"
#include "trainR.h"
#include "samplerR.h"
#include "leafR.h"


const string FBTrain::strNTree = "nTree";
const string FBTrain::strNode = "node";
const string FBTrain::strExtent = "extent";
const string FBTrain::strTreeNode = "treeNode";
const string FBTrain::strScores= "scores";
const string FBTrain::strFactor = "factor";
const string FBTrain::strFacSplit = "facSplit";
const string FBTrain::strObserved = "observed";

const string FBTrain::strScoreDesc = "scoreDesc";
const string FBTrain::strNu = "nu";
const string FBTrain::strBaseScore = "baseScore";
const string FBTrain::strForestScorer = "scorer";


FBTrain::FBTrain(unsigned int nTree_) :
  nTree(nTree_),
  nodeExtent(NumericVector(nTree)),
  nodeTop(0),
  scores(NumericVector(0)),
  facExtent(NumericVector(nTree)),
  facTop(0),
  facRaw(RawVector(0)) {
}


void FBTrain::groveConsume(const GroveBridge* grove,
			   unsigned int tIdx,
			   double scale) {
  nodeConsume(grove, tIdx, scale);
  factorConsume(grove, tIdx, scale);
}


void FBTrain::nodeConsume(const GroveBridge* bridge,
			  unsigned int tIdx,
			  double scale) {
  const vector<size_t>&nExtents = bridge->getNodeExtents();
  unsigned int fromIdx = 0;
  for (unsigned int toIdx = tIdx; toIdx < tIdx + nExtents.size(); toIdx++) {
    nodeExtent[toIdx] = nExtents[fromIdx++];
  }

  size_t nodeCount = bridge->getNodeCount();
  if (nodeTop + nodeCount > static_cast<size_t>(cNode.length())) {
    cNode = std::move(ResizeR::resize<ComplexVector>(cNode, nodeTop, nodeCount, scale));
    scores = std::move(ResizeR::resize<NumericVector>(scores, nodeTop, nodeCount, scale));
  }
  bridge->dumpTree((complex<double>*)&cNode[nodeTop]);
  bridge->dumpScore(&scores[nodeTop]);
  nodeTop += nodeCount;
}


void FBTrain::factorConsume(const GroveBridge* bridge,
			    unsigned int tIdx,
			    double scale) {
  const vector<size_t>& fExtents = bridge->getFacExtents();
  unsigned int fromIdx = 0;
  for (unsigned int toIdx = tIdx; toIdx < tIdx + fExtents.size(); toIdx++) {
    facExtent[toIdx] = fExtents[fromIdx++];
  }
 
  size_t facBytes = bridge->getFactorBytes();
  if (facTop + facBytes > static_cast<size_t>(facRaw.length())) {
    facRaw = std::move(ResizeR::resize<RawVector>(facRaw, facTop, facBytes, scale));
    facObserved = std::move(ResizeR::resize<RawVector>(facObserved, facTop, facBytes, scale));
  }
  bridge->dumpFactorRaw(&facRaw[facTop]);
  bridge->dumpFactorObserved(&facObserved[facTop]);
  facTop += facBytes;
}


void FBTrain::scoreDescConsume(const TrainBridge& trainBridge) {
  trainBridge.getScoreDesc(nu, baseScore, forestScorer);
}


List FBTrain::wrapNode() {
  BEGIN_RCPP
  List wrappedNode = List::create(_[strTreeNode] = std::move(cNode),
				  _[strExtent] = std::move(nodeExtent)
				  );
  wrappedNode.attr("class") = "Node";
  return wrappedNode;
  END_RCPP
}


List FBTrain::wrapFactor() {
  BEGIN_RCPP
    List wrappedFactor = List::create(_[strFacSplit] = std::move(facRaw),
				      _[strExtent] = std::move(facExtent),
				      _[strObserved] = std::move(facObserved)
				      );
  wrappedFactor.attr("class") = "Factor";

  return wrappedFactor;
  END_RCPP
}


List FBTrain::wrap() {
  BEGIN_RCPP
  List forest =
    List::create(_[strNTree] = nTree,
		 _[strScoreDesc] = std::move(summarizeScoreDesc()),
		 _[strNode] = std::move(wrapNode()),
		 _[strScores] = std::move(scores),
		 _[strFactor] = std::move(wrapFactor())
                 );
  cNode = ComplexVector(0);
  scores = NumericVector(0);
  facRaw = RawVector(0);
  facObserved = RawVector(0);
  forest.attr("class") = "Forest";

  return forest;
  END_RCPP
}


List FBTrain::summarizeScoreDesc() {
  return List::create(
		      _[strNu] = nu,
		      _[strBaseScore] = baseScore,
		      _[strForestScorer] = forestScorer
		      );
}


ForestBridge ForestR::unwrap(const List& lTrain,
			     bool categorical) {
  List lForest(checkForest(lTrain));
  List lNode((SEXP) lForest[FBTrain::strNode]);
  List lFactor((SEXP) lForest[FBTrain::strFactor]);
  return ForestBridge(as<unsigned int>(lForest[FBTrain::strNTree]),
		      as<NumericVector>(lNode[FBTrain::strExtent]).begin(),
		      (complex<double>*) as<ComplexVector>(lNode[FBTrain::strTreeNode]).begin(),
		      as<NumericVector>(lForest[FBTrain::strScores]).begin(),
		      as<NumericVector>(lFactor[FBTrain::strExtent]).begin(),
		      as<RawVector>(lFactor[FBTrain::strFacSplit]).begin(),
		      as<RawVector>(lFactor[FBTrain::strObserved]).begin(),
		      unwrapScoreDesc(lForest, categorical));
}


ForestBridge ForestR::unwrap(const List& lTrain,
			     const SamplerBridge& samplerBridge) {
  List lForest(checkForest(lTrain));
  List lNode((SEXP) lForest[FBTrain::strNode]);
  List lFactor((SEXP) lForest[FBTrain::strFactor]);
  List lLeaf((SEXP) lTrain[TrainR::strLeaf]);
  bool emptyLeaf = (Rf_isNull(lLeaf[LeafR::strIndex]) || Rf_isNull(lLeaf[LeafR::strExtent]));
  bool thinLeaf = emptyLeaf || as<NumericVector>(lLeaf[LeafR::strExtent]).length() == 0;
  return ForestBridge(as<unsigned int>(lForest[FBTrain::strNTree]),
		      as<NumericVector>(lNode[FBTrain::strExtent]).begin(),
		      (complex<double>*) as<ComplexVector>(lNode[FBTrain::strTreeNode]).begin(),
		      as<NumericVector>(lForest[FBTrain::strScores]).begin(),
		      as<NumericVector>(lFactor[FBTrain::strExtent]).begin(),
		      as<RawVector>(lFactor[FBTrain::strFacSplit]).begin(),
		      as<RawVector>(lFactor[FBTrain::strObserved]).begin(),
		      unwrapScoreDesc(lForest, samplerBridge.categorical()),
		      samplerBridge,
		      thinLeaf ? nullptr : as<NumericVector>(lLeaf[LeafR::strExtent]).begin(),
		      thinLeaf ? nullptr : as<NumericVector>(lLeaf[LeafR::strIndex]).begin());
}


tuple<double, double, string> ForestR::unwrapScoreDesc(const List& lForest,
						       bool categorical) {
  // Legacy RF implementations did not record a score descriptor,
  // so one is created on-the-fly:
  if (!lForest.containsElementNamed("scoreDesc")) {
    if (categorical)
      return make_tuple<double, double, string>(0.0, 0.0, "plurality");
    else
      return make_tuple<double, double, string>(0.0, 0.0, "mean");
  }
  
  List lScoreDesc(as<List>(lForest[FBTrain::strScoreDesc]));
  return make_tuple<double, double>(as<double>(lScoreDesc[FBTrain::strNu]), as<double>(lScoreDesc[FBTrain::strBaseScore]), as<string>(lScoreDesc[FBTrain::strForestScorer]));
}


List ForestR::checkForest(const List& lTrain) {
  BEGIN_RCPP

  List lForest((SEXP) lTrain["forest"]);
  if (!lForest.inherits("Forest")) {
    stop("Expecting Forest");
  }
  return lForest;
  
  END_RCPP
}


ForestExpand ForestExpand::unwrap(const List& lTrain,
				  const IntegerVector& predMap) {
  (void) ForestR::checkForest(lTrain);
  return ForestExpand(lTrain, predMap);
}


ForestExpand::ForestExpand(const List &lTrain,
                           const IntegerVector& predMap) {
  // Leaving legacy categorical flag turned off:  not quite correct.
  ForestBridge forestBridge = ForestR::unwrap(lTrain);

  predTree = vector<vector<unsigned int>>(forestBridge.getNTree());
  bumpTree = vector<vector<size_t> >(forestBridge.getNTree());
  splitTree = vector<vector<double > >(forestBridge.getNTree());
  facSplitTree = vector<vector<unsigned char> >(forestBridge.getNTree());
  scoreTree = vector<vector<double>>(forestBridge.getNTree());
  forestBridge.dump(predTree, splitTree, bumpTree, facSplitTree, scoreTree);
  predExport(predMap.begin());
}


/**
   @brief Prepares predictor field for export by remapping to front-end indices.
 */
void ForestExpand::predExport(const int predMap[]) {
  for (unsigned int tIdx = 0; tIdx < predTree.size(); tIdx++) {
    treeExport(predMap, predTree[tIdx], bumpTree[tIdx]);
  }
}


/**
   @brief Recasts 'pred' field of nonterminals to front-end facing values.
 */
void ForestExpand::treeExport(const int predMap[],
			      vector<unsigned int>& pred,
			      const vector<size_t>& bump) {
  for (unsigned int i = 0; i < pred.size(); i++) {
    if (bump[i] > 0) { // terminal 'pred' values do not reference predictors.
      unsigned int predCore = pred[i];
      pred[i] = predMap[predCore];
    }
  }
}


List ForestExpand::expand(const List& lTrain,
			  const IntegerVector& predMap) {
  BEGIN_RCPP

  ForestExpand forest(ForestExpand::unwrap(lTrain, predMap));
  unsigned int nTree = forest.predTree.size();
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    List ffReg =
      List::create(
                   _["tree"] = expandTree(forest, tIdx)
                   );
    ffReg.attr("class") = "expandForest";
    trees[tIdx] = std::move(ffReg);
  }
  return trees;

  END_RCPP
}


List ForestExpand::expandTree(const ForestExpand& forest,
			 unsigned int tIdx) {
  BEGIN_RCPP

  auto predTree(forest.getPredTree(tIdx));
  auto bumpTree(forest.getBumpTree(tIdx));
  IntegerVector incrL(bumpTree.begin(), bumpTree.end());
  IntegerVector predIdx(predTree.begin(), predTree.end());
  List ffTree = List::create(
     _["pred"] = ifelse(incrL == 0, -(predIdx + 1), predIdx),
     _["childL"] = incrL,
     _["childR"] = ifelse(incrL == 0, 0, incrL + 1),
     _["split"] = forest.getSplitTree(tIdx),
     _["facSplit"] = forest.getFacSplitTree(tIdx),
     _["score"] = forest.getScoreTree(tIdx)
     );

  ffTree.attr("class") = "expandTree";
  return ffTree;
  END_RCPP
}
