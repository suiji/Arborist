// Copyright (C)  2012-2023   Mark Seligman
//
// This file is part of RboristBase.
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
   @file extendR.cc

   @brief C++ interface to R entry for extension methods.

   @author Mark Seligman
 */

#include "expandR.h"
#include "leafR.h"
#include "samplerR.h"
#include "forestR.h"
#include "signatureR.h"

#include <vector>


RcppExport SEXP expandRf(SEXP sTrain) {
  BEGIN_RCPP
    
  List lTrain(sTrain);
  if (!lTrain.inherits("rfArb")) {
    warning("Expecting an rfArb object");
    return List::create(0);
  }

  List leaf((SEXP) lTrain["leaf"]);
  if (leaf.inherits("Leaf")) {
    List lSampler((SEXP) lTrain["sampler"]);
    SEXP yTrain = lSampler[SamplerR::strYTrain];
    if (Rf_isFactor(yTrain)) {
      return ExportRf::exportCtg(lTrain);
    }
    else {
      return ExportRf::exportReg(lTrain);
    }
  }
  else {
    warning("Unrecognized leaf type.");
    return List::create(0);
  }

  END_RCPP
}


List ExportRf::exportForest(const ForestExport& forest,
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

  ffTree.attr("class") = "exportTree";
  return ffTree;
  END_RCPP
}


IntegerVector ExportRf::exportBag(const SamplerExport& sampler,
				  const LeafExport& leaf,
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


List ExportRf::exportLeafReg(const LeafExportReg& leaf, unsigned int tIdx) {
  BEGIN_RCPP

  List ffLeaf = List::create(
                             _["score"] = leaf.getScoreTree(tIdx)
                             );

  ffLeaf.attr("class") = "exportLeafReg";
  return ffLeaf;

  END_RCPP
}


List ExportRf::exportTreeCtg(const List& lTrain,
			     const IntegerVector& predMap) {
  BEGIN_RCPP

  LeafExportCtg leaf(LeafExportCtg::unwrap(lTrain));
  ForestExport forest(ForestExport::unwrap(lTrain, predMap));
  SamplerExport sampler(SamplerR::unwrapExport(lTrain));

  unsigned int nTree = sampler.nTree;
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    List ffCtg =
      List::create(
                   _["internal"] = exportForest(forest, tIdx),
                   _["leaf"] = exportLeafCtg(leaf, tIdx),
                   _["bag"] = exportBag(sampler, leaf, tIdx)
                   );
    ffCtg.attr("class") = "exportTreeCtg";
    trees[tIdx] = std::move(ffCtg);
  }
  return trees;

  END_RCPP
}


List ExportRf::exportLeafCtg(const LeafExportCtg& leaf,
                             unsigned int tIdx) {
  BEGIN_RCPP

  List ffLeaf =
    List::create(
                 _["score"] = leaf.getScoreTree(tIdx)
                 );
  ffLeaf.attr("class") = "exportLeafCtg";
  return ffLeaf;
  END_RCPP
}


List ExportRf::exportReg(const List& lTrain) {
  BEGIN_RCPP

  IntegerVector predMap((SEXP) lTrain["predMap"]);
  SignatureExport signature = SignatureExport::unwrap(lTrain);
  unsigned int facCount = signature.level.length();
  List ffe =
    List::create(_["predMap"] = IntegerVector(predMap),
                 _["factorMap"] = IntegerVector(predMap.end() - facCount, predMap.end()),
                 _["predLevel"] = signature.level,
		 _["predFactor"] = signature.factor,
                 _["tree"] = exportTreeReg(lTrain, predMap)
                 );

  ffe.attr("class") = "ExportReg";
  return ffe;

  END_RCPP
}


List ExportRf::exportTreeReg(const List& lTrain,
                             const IntegerVector& predMap) {
  BEGIN_RCPP

  LeafExportReg leaf(LeafExportReg::unwrap(lTrain));
  ForestExport forest(ForestExport::unwrap(lTrain, predMap));
  SamplerExport sampler = SamplerR::unwrapExport(lTrain);

  unsigned int nTree = sampler.nTree;
  List trees(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    List ffReg =
      List::create(
                   _["internal"] = exportForest(forest, tIdx),
                   _["leaf"] = exportLeafReg(leaf, tIdx),
                   _["bag"] = exportBag(sampler, leaf, tIdx)
                   );
      ffReg.attr("class") = "exportTreeReg";
      trees[tIdx] = std::move(ffReg);
  }
  return trees;

  END_RCPP
}


List ExportRf::exportCtg(const List& lTrain) {
  BEGIN_RCPP

  IntegerVector predMap((SEXP) lTrain["predMap"]);
  SignatureExport signature = SignatureExport::unwrap(lTrain);

  LeafExportCtg leaf(LeafExportCtg::unwrap(lTrain));
  int facCount = signature.level.length();
  List ffe =
    List::create(
                 _["facMap"] = IntegerVector(predMap.end() - facCount, predMap.end()),
                 _["predLevel"] = signature.level,
                 _["yLevel"] = leaf.getLevelsTrain(),
                 _["tree"] = exportTreeCtg(lTrain, predMap)
  );
  ffe.attr("class") = "ExportCtg";
  return ffe;

  END_RCPP
}
