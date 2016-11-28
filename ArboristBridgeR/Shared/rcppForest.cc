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

/**
   @file rcppForest.cc

   @brief C++ interface to R entry for Forest methods.

   @author Mark Seligman
 */


#include "forest.h"
#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

#include "rcppForest.h"

//#include <iostream>

SEXP RcppForest::Wrap(const std::vector<unsigned int> &origin, const std::vector<unsigned int> &facOrigin, const std::vector<unsigned int> &facSplit, const std::vector<ForestNode> &forestNode) {
  size_t numSize = forestNode.size() * (sizeof(ForestNode) / sizeof(double));
  NumericVector fnNum(numSize);
  for (size_t i = 0; i < numSize; i++) {
    fnNum[i] = ((double*) &forestNode[0])[i];
  }

  List forest = List::create(
     _["forestNode"] = fnNum,
     _["origin"] = origin,
     _["facOrig"] = facOrigin,
     _["facSplit"] = facSplit);
  forest.attr("class") = "Forest";

  return forest;
}


/**
   @brief Exposes front-end Forest fields for transmission to core.

   @return void.
 */
void RcppForest::Unwrap(SEXP sForest, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facSplit, size_t &_facLen, std::vector<unsigned int> &_facOrig, ForestNode *&_forestNode, unsigned int &_nodeEnd) {
  List forest(sForest);
  if (!forest.inherits("Forest"))
    stop("Expecting Forest");

  _origin = as<std::vector<unsigned int> >(forest["origin"]);
  _facSplit = as<std::vector<unsigned int> >(forest["facSplit"]);
  _facLen = _facSplit.size();
  _facOrig = as<std::vector<unsigned int> >(forest["facOrig"]);

  NumericVector fnNum((SEXP) forest["forestNode"]);
  _forestNode = (ForestNode *) fnNum.begin();
  _nodeEnd = fnNum.length() / (sizeof(ForestNode) / sizeof(double));
}
