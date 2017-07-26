// Copyright (C)  2012-2017   Mark Seligman
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
  size_t forestSize = forestNode.size() * sizeof(ForestNode);
  RawVector forestRaw(forestSize);
  for (size_t i = 0; i < forestSize; i++) {
    forestRaw[i] = ((unsigned char *) &forestNode[0])[i];
  }

  size_t facSize = facSplit.size() * sizeof(unsigned int);
  RawVector facRaw(facSize);
  for (size_t i = 0; i < facSize; i++) {
    facRaw[i] = ((unsigned char*) &facSplit[0])[i];
  }
  
  List forest = List::create(
     _["forestNode"] = forestRaw,
     _["origin"] = origin,
     _["facOrig"] = facOrigin,
     _["facSplit"] = facRaw);
  forest.attr("class") = "Forest";

  return forest;
}

RawVector RcppForest::rv1 = RawVector(0);
RawVector RcppForest::rv2 = RawVector(0);
IntegerVector RcppForest::iv1 = IntegerVector(0);
IntegerVector RcppForest::iv2 = IntegerVector(0);

/**
   @brief Exposes front-end Forest fields for transmission to core.

   @return void.
 */
void RcppForest::Unwrap(SEXP sForest, unsigned int *&_origin, unsigned int &_nTree, unsigned int *&_facSplit, size_t &_facLen, unsigned int *&_facOrig, unsigned int &_nFac, ForestNode *&_forestNode, unsigned int &_nodeEnd) {
  List forest(sForest);
  try {
    if (!forest.inherits("Forest"))
      throw std::domain_error("Expecting Forest");
  }
  catch(std::exception &ex) {
    forward_exception_to_r(ex);
  }

  // Alignment should be sufficient to guarantee safety of
  // the casted loads.
  //
  iv1 = IntegerVector((SEXP) forest["origin"]);
  _origin = (unsigned int*) &iv1[0];
  _nTree = iv1.length();

  rv1 = RawVector((SEXP) forest["facSplit"]);
  _facSplit = (unsigned int*) &rv1[0];
  _facLen = rv1.length() / sizeof(unsigned int);

  iv2 = IntegerVector((SEXP) forest["facOrig"]);
  _facOrig = (unsigned int*) &iv2[0];
  _nFac = iv2.length();

  rv2 = RawVector((SEXP) forest["forestNode"]);
  _forestNode = (ForestNode*) &rv2[0];
  _nodeEnd = rv2.length() / sizeof(ForestNode);
}


void RcppForest::Clear() {
  rv1 = RawVector(0);
  rv2 = RawVector(0);
  iv1 = IntegerVector(0);
  iv2 = IntegerVector(0);
}
