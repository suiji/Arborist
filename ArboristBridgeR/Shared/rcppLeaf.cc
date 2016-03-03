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
  template <> SEXP wrap(const std::vector<RankCount> &);
  template <> std::vector<LeafNode>* as(SEXP);
  template <> std::vector<RankCount>* as(SEXP);
}

#include "rcppLeaf.h"

template <> SEXP Rcpp::wrap(const std::vector<LeafNode> &leafNode) {
  XPtr<const std::vector<LeafNode> > extWrap(new std::vector<LeafNode>(leafNode), true);

  return wrap(extWrap);
}


template <> SEXP Rcpp::wrap(const std::vector<RankCount> &leafInfo) {
  XPtr<const std::vector<RankCount> > extWrap(new std::vector<RankCount>(leafInfo), true);

  return wrap(extWrap);
}


template <> std::vector<LeafNode>* Rcpp::as(SEXP sLNReg) {
  Rcpp::XPtr<std::vector<LeafNode> > xp(sLNReg);
  return (std::vector<LeafNode>*) xp;
}


template <> std::vector<RankCount>* Rcpp::as(SEXP sLNReg) {
  Rcpp::XPtr<std::vector<RankCount> > xp(sLNReg);
  return (std::vector<RankCount>*) xp;
}


/**
   @brief Wraps core (regression) Leaf vectors for reference by front end.
 */
RcppExport SEXP LeafWrapReg(const std::vector<unsigned int> &leafOrigin, const std::vector<LeafNode> &leafNode, const std::vector<RankCount> &leafInfo, const std::vector<double> &yRanked) {
  List leaf = List::create(
   _["origin"] = leafOrigin,
   _["node"] = leafNode,			   
   _["info"] = leafInfo,
   _["yRanked"] = yRanked
  );
  leaf.attr("class") = "LeafReg";
  
  return leaf;
}


/**
   @brief Wraps core (classification) Leaf vectors for reference by front end.
 */
RcppExport SEXP LeafWrapCtg(const std::vector<unsigned int> &leafOrigin, const std::vector<LeafNode> &leafNode, const std::vector<double> &leafInfo, const CharacterVector &levels) {
  List leaf = List::create(
   _["origin"] = leafOrigin,	
   _["node"] = leafNode,
   _["info"] = leafInfo,
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
void LeafUnwrapReg(SEXP sLeaf, std::vector<double> &_yRanked, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> *&_leafNode, std::vector<RankCount> *&_leafInfo) {
  List leaf(sLeaf);
  if (!leaf.inherits("LeafReg"))
    stop("Expecting LeafReg");

  _yRanked = leaf["yRanked"];
  _leafOrigin = leaf["origin"];
  _leafNode = leaf["node"];
  _leafInfo = leaf["info"];
}


/**
   @brief Exposes front-end (classification) Leaf fields for transmission to core.

   @param sLeaf is the R object containing the leaf (list) data.

   @param _weight outputs the sample weights.

   @param _levels outputs the category levels; retains as front-end object.

   @return void, with output reference parameters.
 */
void LeafUnwrapCtg(SEXP sLeaf, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> *&_leafNode, std::vector<double> &_leafInfo, CharacterVector &_levels) {
  List leaf(sLeaf);
  if (!leaf.inherits("LeafCtg")) {
    stop("Expecting LeafCtg");
  }
  _leafOrigin = leaf["origin"];
  _leafNode = leaf["node"];
  _leafInfo = leaf["info"];
  _levels = CharacterVector((SEXP) leaf["levels"]);
}
