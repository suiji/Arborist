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
  static RawVector rv1, rv2, rv3;
  static NumericVector nv1;
  
  static void Serialize(const std::vector<class LeafNode> &leafNode, const std::vector<class BagLeaf> &bagLeaf, const std::vector<unsigned int> &bagBits, RawVector &leafRaw, RawVector &blRaw, RawVector &bbRaw);


 public:
  static SEXP WrapReg(const std::vector<unsigned int> &leafOrigin, std::vector<class LeafNode> &leafNode, const std::vector<class BagLeaf> &bagLeaf, const std::vector<unsigned int> &bagBits, const std::vector<double> &yTrain);
  static SEXP WrapCtg(const std::vector<unsigned int> &leafOrigin, const std::vector<LeafNode> &leafNode, const std::vector<BagLeaf> &bagLeaf, const std::vector<unsigned int> &bagBits, const std::vector<double> &weight, unsigned int rowTrain, const CharacterVector &levels);
  static void UnwrapReg(SEXP sLeaf, std::vector<double> &_yTrain, std::vector<unsigned int> &_leafOrigin, class LeafNode *&_leafNode, unsigned int &_leafCount, class BagLeaf *&_bagLeaf, unsigned int &bagLeafTot, unsigned int *&_bagBits, bool bag);
  static void UnwrapCtg(SEXP sLeaf, std::vector<unsigned int> &_leafOrigin, class LeafNode *&_leafNode, unsigned int &_leafCount, class BagLeaf *&_bagLeaf, unsigned int &bagLeafTot, unsigned int *&_bagBits, double *&_weight, unsigned int &_rowTrain, CharacterVector &_levels, bool bag);
static void Clear();
};

#endif
