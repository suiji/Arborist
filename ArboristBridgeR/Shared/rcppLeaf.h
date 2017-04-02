// Copyright (C)  2012-2017  Mark Seligman
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
   @file rcppLeaf.h

   @brief C++ class definitions for managing Leaf object.

   @author Mark Seligman

 */


#ifndef ARBORIST_RCPP_LEAF_H
#define ARBORIST_RCPP_LEAF_H

#include <vector>
#include <Rcpp.h>
using namespace Rcpp;

class RcppLeaf {
 public:
  static SEXP WrapReg(const std::vector<unsigned int> &leafOrigin, std::vector<class LeafNode> &leafNode, const std::vector<class BagLeaf> &bagLeaf, const std::vector<unsigned int> &bagBits, const std::vector<double> &yTrain);
  static SEXP WrapCtg(const std::vector<unsigned int> &leafOrigin, const std::vector<LeafNode> &leafNode, const std::vector<BagLeaf> &bagLeaf, const std::vector<unsigned int> &bagBits, const std::vector<double> &weight, const CharacterVector &levels);
  static void UnwrapReg(SEXP sLeaf, std::vector<double> &_yTrain, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, bool bag);
  static void UnwrapCtg(SEXP sLeaf, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, std::vector<double> &_weight, CharacterVector &_levels, bool bag);
};

#endif
