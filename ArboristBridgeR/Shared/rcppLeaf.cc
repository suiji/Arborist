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


#include <RcppCommon.h>

#include "leaf.h"

namespace Rcpp {
  template <> SEXP wrap(const std::vector<LeafNode> &);
  template <> SEXP wrap(const std::vector<BagRow> &);
  template <> std::vector<LeafNode>* as(SEXP);
  template <> std::vector<BagRow>* as(SEXP);
}

#include "rcppLeaf.h"

template <> SEXP Rcpp::wrap(const std::vector<LeafNode> &leafNode) {
  XPtr<const std::vector<LeafNode> > extWrap(new std::vector<LeafNode>(leafNode), true);

  return wrap(extWrap);
}


template <> SEXP Rcpp::wrap(const std::vector<BagRow> &bagRow) {
  XPtr<const std::vector<BagRow> > extWrap(new std::vector<BagRow>(bagRow), true);

  return wrap(extWrap);
}


template <> std::vector<LeafNode>* Rcpp::as(SEXP sLNReg) {
  Rcpp::XPtr<std::vector<LeafNode> > xp(sLNReg);
  return (std::vector<LeafNode>*) xp;
}


template <> std::vector<BagRow>* Rcpp::as(SEXP sLNReg) {
  Rcpp::XPtr<std::vector<BagRow> > xp(sLNReg);
  return (std::vector<BagRow>*) xp;
}


/**
   @brief Wraps core (regression) Leaf vectors for reference by front end.
 */
RcppExport SEXP LeafWrapReg(const std::vector<unsigned int> &leafOrigin, const std::vector<LeafNode> &leafNode, const std::vector<BagRow> &bagRow, unsigned int rowTrain, const std::vector<unsigned int> &rank, const std::vector<double> &yRanked) {
  List leaf = List::create(
   _["origin"] = leafOrigin,
   _["node"] = leafNode,
   _["bagRow"] = bagRow,
   _["rowTrain"] = rowTrain,
   _["rank"] = rank,
   _["yRanked"] = yRanked
  );
  leaf.attr("class") = "LeafReg";
  
  return leaf;
}


/**
   @brief Wraps core (classification) Leaf vectors for reference by front end.
 */
RcppExport SEXP LeafWrapCtg(const std::vector<unsigned int> &leafOrigin, const std::vector<LeafNode> &leafNode, const std::vector<BagRow> &bagRow, unsigned int rowTrain, const std::vector<double> &weight, const CharacterVector &levels) {
  List leaf = List::create(
   _["origin"] = leafOrigin,	
   _["node"] = leafNode,
   _["bagRow"] = bagRow,
   _["rowTrain"] = rowTrain,
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

   @param _leafInfoReg outputs the sample ranks and counts, organized by leaf.

   @return void, with output reference parameters.
 */
void LeafUnwrapReg(SEXP sLeaf, std::vector<double> &_yRanked, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> *&_leafNode, std::vector<BagRow> *&_bagRow, unsigned int &_rowTrain, std::vector<unsigned int> &_rank) {
  List leaf(sLeaf);
  if (!leaf.inherits("LeafReg"))
    stop("Expecting LeafReg");

  _yRanked = as<std::vector<double> >(leaf["yRanked"]);
  _leafOrigin = as<std::vector<unsigned int>>(leaf["origin"]);
  _leafNode = as<std::vector<LeafNode> *>(leaf["node"]);
  _bagRow = as<std::vector<BagRow> *>(leaf["bagRow"]);
  _rowTrain = as<unsigned int>(leaf["rowTrain"]);
  _rank = as<std::vector<unsigned int> >(leaf["rank"]);
}


/**
   @brief Exposes front-end (classification) Leaf fields for transmission to core.

   @param sLeaf is the R object containing the leaf (list) data.

   @param _weight outputs the sample weights.

   @param _levels outputs the category levels; retains as front-end object.

   @return void, with output reference parameters.
 */
void LeafUnwrapCtg(SEXP sLeaf, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> *&_leafNode, std::vector<BagRow> *&_bagRow, unsigned int &_rowTrain, std::vector<double> &_weight, CharacterVector &_levels) {
  List leaf(sLeaf);
  if (!leaf.inherits("LeafCtg")) {
    stop("Expecting LeafCtg");
  }
  _leafOrigin = as<std::vector<unsigned int> >(leaf["origin"]);
  _leafNode = as<std::vector<LeafNode> *>(leaf["node"]);
  _bagRow = as<std::vector<BagRow> *>(leaf["bagRow"]);
  _rowTrain = as<unsigned int>(leaf["rowTrain"]);
  _weight = as<std::vector<double> >(leaf["weight"]);
  _levels = as<CharacterVector>((SEXP) leaf["levels"]);
}
