// Copyright (C)  2012-2016   Mark Seligman
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


#include <Rcpp.h>
using namespace Rcpp;

#include "forest.h"

/**
   @file rcppForestFloor.cc

   @brief C++ interface to R entry for ForestFloor export.

   @author Mark Seligman
 */


/**
   @brief Structures forest summary for analysis by ForestFloor package.

   @param sForest is the Forest summary.

   @return wrapped ForestFloorExport object.
 */
RcppExport SEXP RcppForestFloorExport(SEXP sForest, SEXP sLeaf) {
  /*
  List forest(sForest);
  if (!forest.inherits("Forest"))
    stop("Expecting forest");

  List ffe = List::create(nTree);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++)
    ffe[i] = FFTree(forest, leaf, bag, facSplit, tIdx);
  ffe.attr("class") = "ForestFloorExport";
  */
  List ffe;
  return wrap(ffe);
}


/**
   @brief Packages individual tree.

   @param tIdx is the tree index.

   @return wrapped tree summary.
 */
RcppExport SEXP FFTree(std::vector<ForestNode> &forestNode, std::vector<unsigned int> tOrigin, int tIdx) {
  List ffeTree;
  /*
  List ffeTree = List::create(
      _["Tree"] = Forest::TreeExportforestNode, tOrigin, tIdx, incrL, pred, split),
      _["Leaf"] = Leaf::TreeExport(tIdx),
      _["Bag"] = bag->Expand(tIdx),
      _["Fac"] = facSplit->Expand(tIdx)
     );
  */
  ffeTree.attr("class") = "FFETree";
  return ffeTree;
}


RcppExport SEXP BagExpand(unsigned int tIdx) {
  return wrap(0);
}


RcppExport SEXP FacExpand(unsigned int tIdx) {
  return wrap(0);
}
