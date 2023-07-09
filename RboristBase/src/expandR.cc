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
   @file expandR.cc

   @brief C++ interface to R entry for expansion methods.

   @author Mark Seligman
 */

#include "expandR.h"
#include "leafR.h"
#include "samplerR.h"
#include "forestR.h"
#include "forestbridge.h"
#include "signatureR.h"

#include <vector>


RcppExport SEXP expandR(SEXP sTrain) {
  BEGIN_RCPP
    
  List lTrain(sTrain);
  if (!lTrain.inherits("rfArb")) {
    warning("Expecting an rfArb object");
    return List::create(0);
  }

  List lForest(lTrain["forest"]);
  List leaf((SEXP) lTrain["leaf"]);
  if (leaf.inherits("Leaf")) {
    List lSampler((SEXP) lTrain["sampler"]);
    SEXP yTrain = lSampler[SamplerR::strYTrain];
    if (Rf_isFactor(yTrain)) {
      return ExpandR::expandCtg(lTrain);
    }
    else {
      return ExpandR::expandReg(lTrain);
    }
  }
  else {
    warning("Unrecognized leaf type.");
    return List::create(0);
  }
  ForestBridge::deInit();

  END_RCPP
}


List ExpandR::expandForest(const ForestExpand& forest,
                            unsigned int tIdx) {
  BEGIN_RCPP

  auto predTree(forest.getPredTree(tIdx));
  auto bumpTree(forest.getBumpTree(tIdx));
  IntegerVector incrL(bumpTree.begin(), bumpTree.end());
  IntegerVector predIdx(predTree.begin(), predTree.end());
  List ffTree = List::create(
     _["pred"] = ifelse(incrL == 0, -(predIdx + 1), predIdx),
     _["daughterL"] = incrL,
     _["daughterR"] = ifelse(incrL == 0, 0, incrL + 1),
     _["split"] = forest.getSplitTree(tIdx),
     _["facSplit"] = forest.getFacSplitTree(tIdx)
     );

  ffTree.attr("class") = "expandTree";
  return ffTree;
  END_RCPP
}


IntegerVector ExpandR::expandBag(const SamplerExpand& sampler,
				  const LeafExpand& leaf,
                                  unsigned int tIdx) {
  BEGIN_RCPP

  auto rowTree(leaf.getRowTree(tIdx));
  auto sCountTree(leaf.getSCountTree(tIdx));

  IntegerVector row(rowTree.begin(), rowTree.end());
  IntegerVector sCount(sCountTree.begin(), sCountTree.end());
  IntegerVector bag(sampler.nObs);

  bag[row] = sCount;

  return bag;
  END_RCPP
}


List ExpandR::expandLeafReg(const LeafExpandReg& leaf, unsigned int tIdx) {
  BEGIN_RCPP

  List ffLeaf = List::create(
                             _["score"] = leaf.getScoreTree(tIdx)
                             );

  ffLeaf.attr("class") = "expandLeafReg";
  return ffLeaf;

  END_RCPP
}


List ExpandR::expandTreeCtg(const List& lTrain,
			     const IntegerVector& predMap) {
  BEGIN_RCPP

  LeafExpandCtg leaf(LeafExpandCtg::unwrap(lTrain));
  ForestExpand forest(ForestExpand::unwrap(lTrain, predMap));
  SamplerExpand sampler(SamplerR::unwrapExpand(lTrain));

  unsigned int nTree = sampler.nTree;
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    List ffCtg =
      List::create(
                   _["internal"] = expandForest(forest, tIdx),
                   _["leaf"] = expandLeafCtg(leaf, tIdx),
                   _["bag"] = expandBag(sampler, leaf, tIdx)
                   );
    ffCtg.attr("class") = "expandTreeCtg";
    trees[tIdx] = std::move(ffCtg);
  }
  return trees;

  END_RCPP
}


List ExpandR::expandLeafCtg(const LeafExpandCtg& leaf,
                             unsigned int tIdx) {
  BEGIN_RCPP

  List ffLeaf =
    List::create(
                 _["score"] = leaf.getScoreTree(tIdx)
                 );
  ffLeaf.attr("class") = "expandLeafCtg";
  return ffLeaf;
  END_RCPP
}


List ExpandR::expandReg(const List& lTrain) {
  BEGIN_RCPP

  IntegerVector predMap((SEXP) lTrain["predMap"]);
  SignatureExpand signature = SignatureExpand::unwrap(lTrain);
  unsigned int facCount = signature.level.length();
  List ffe =
    List::create(_["predMap"] = IntegerVector(predMap),
                 _["factorMap"] = IntegerVector(predMap.end() - facCount, predMap.end()),
                 _["predLevel"] = signature.level,
		 _["predFactor"] = signature.factor,
                 _["tree"] = expandTreeReg(lTrain, predMap)
                 );

  ffe.attr("class") = "ExpandReg";
  return ffe;

  END_RCPP
}


List ExpandR::expandTreeReg(const List& lTrain,
                             const IntegerVector& predMap) {
  BEGIN_RCPP

  LeafExpandReg leaf(LeafExpandReg::unwrap(lTrain));
  ForestExpand forest(ForestExpand::unwrap(lTrain, predMap));
  SamplerExpand sampler = SamplerR::unwrapExpand(lTrain);

  unsigned int nTree = sampler.nTree;
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    List ffReg =
      List::create(
                   _["internal"] = expandForest(forest, tIdx),
                   _["leaf"] = expandLeafReg(leaf, tIdx),
                   _["bag"] = expandBag(sampler, leaf, tIdx)
                   );
      ffReg.attr("class") = "expandTreeReg";
      trees[tIdx] = std::move(ffReg);
  }
  return trees;

  END_RCPP
}


List ExpandR::expandCtg(const List& lTrain) {
  BEGIN_RCPP

  IntegerVector predMap((SEXP) lTrain["predMap"]);
  SignatureExpand signature = SignatureExpand::unwrap(lTrain);

  LeafExpandCtg leaf(LeafExpandCtg::unwrap(lTrain));
  int facCount = signature.level.length();
  List ffe =
    List::create(
                 _["facMap"] = IntegerVector(predMap.end() - facCount, predMap.end()),
                 _["predLevel"] = signature.level,
                 _["yLevel"] = leaf.getLevelsTrain(),
                 _["tree"] = expandTreeCtg(lTrain, predMap)
  );
  ffe.attr("class") = "ExpandCtg";
  return ffe;

  END_RCPP
}
