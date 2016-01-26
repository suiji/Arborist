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

#include <R.h>
#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

#include "rcppForest.h"

//#include <iostream>

RcppExport SEXP RcppForestWrap(std::vector<int> pred, std::vector<double> split, std::vector<int> bump, IntegerVector origin, IntegerVector facOrigin, std::vector<unsigned int> facSplit) {
  List forest = List::create(
     _["pred"] = pred,
     _["split"] = split,
     _["bump"] = bump,
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
void RcppForestUnwrap(SEXP sForest, int *&_pred, double *&_split, int *&_bump, int *&_origin, int *&_facOrig, unsigned int *&_facSplit, int &_nTree, int &_height) {
  List forest(sForest);
  if (!forest.inherits("Forest"))
    stop("Expecting Forest");

  _pred = IntegerVector((SEXP) forest["pred"]).begin();
  _split = NumericVector((SEXP) forest["split"]).begin();
  _bump = IntegerVector((SEXP) forest["bump"]).begin();
  _origin = IntegerVector((SEXP) forest["origin"]).begin();
  _facOrig = IntegerVector((SEXP) forest["facOrig"]).begin();
  _facSplit = (unsigned int*) IntegerVector((SEXP) forest["facSplit"]).begin();
  _nTree = IntegerVector((SEXP) forest["origin"]).length();
  _height = IntegerVector((SEXP) forest["pred"]).length();
}
