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
   @file rcppLeaf.cc

   @brief C++ interface to R entry for Leaf methods.

   @author Mark Seligman
 */

#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

#include "rcppLeaf.h"

//#include <iostream>

/**
   @brief Wraps core (regression) Leaf vectors for reference by front end.
 */
RcppExport SEXP RcppLeafWrapReg(std::vector<unsigned int> rank, std::vector<unsigned int> sCount, NumericVector yRanked) {
  List leaf = List::create(
      _["rank"] = rank,
      _["sCount"] = sCount,
      _["yRanked"] = yRanked
  );
  leaf.attr("class") = "LeafReg";
  
  return leaf;
}


/**
   @brief Wraps core (classification) Leaf vectors for reference by front end.
 */
RcppExport SEXP RcppLeafWrapCtg(std::vector<double> weight, CharacterVector levels) {
  List leaf = List::create(
   _["weight"] = weight,
   _["levels"] = levels
   );
  leaf.attr("class") = "LeafCtg";

  return leaf;
}


/**
   @brief Exposes front-end (regression) Leaf fields for transmission to core.

   @param sLeaf is the R object containing the leaf (list) data.

   @param _yRanked outputs the sorted response.

   @param _rank outputs the sample ranks, organized by leaf; unwrapped to unsigned.

   @param _sCount outputs the sample counts, organized by leaf; unwrapped to unsigned.

   @return void, with output reference parameters.
 */
void RcppLeafUnwrapReg(SEXP sLeaf, std::vector<double> &_yRanked, std::vector<unsigned int> &_rank, std::vector<unsigned int> &_sCount) {
  List leaf(sLeaf);
  if (!leaf.inherits("LeafReg"))
    stop("Expecting LeafReg");

  _yRanked = as<std::vector<double > >(leaf["yRanked"]);
  _rank = leaf["rank"];
  _sCount = leaf["sCount"];
}


/**
   @brief Exposes front-end (classification) Leaf fields for transmission to core.

   @param sLeaf is the R object containing the leaf (list) data.

   @param _weight outputs the sample weights.

   @param _levels outputs the category levels; retains as front-end object.

   @return void, with output reference parameters.
 */
void RcppLeafUnwrapCtg(SEXP sLeaf, double *&_weight, CharacterVector &_levels) {
  List leaf(sLeaf);
  if (!leaf.inherits("LeafCtg")) {
    stop("Expecting LeafCtg");
  }
  _weight = NumericVector((SEXP) leaf["weight"]).begin();
  _levels = CharacterVector((SEXP) leaf["levels"]);
}
