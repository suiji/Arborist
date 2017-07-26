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
   @file rcppLeaf.cc

   @brief C++ interface to R entry for Leaf methods.

   @author Mark Seligman
 */


#include "leaf.h"
#include "rcppLeaf.h"


/**
   @brief Wraps core (regression) Leaf vectors for reference by front end.
 */
SEXP RcppLeaf::WrapReg(const std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, const std::vector<BagLeaf> &bagLeaf, const std::vector<unsigned int> &bagBits, const std::vector<double> &yTrain) {
  RawVector leafRaw(leafNode.size() * sizeof(LeafNode));
  RawVector blRaw(bagLeaf.size() * sizeof(BagLeaf));
  RawVector bbRaw(bagBits.size() * sizeof(unsigned int));
  Serialize(leafNode, bagLeaf, bagBits, leafRaw, blRaw, bbRaw);
  List leaf = List::create(
   _["origin"] = leafOrigin,
   _["node"] = leafRaw,
   _["bagLeaf"] = blRaw,
   _["bagBits"] = bbRaw,
   _["yTrain"] = yTrain
  );
  leaf.attr("class") = "LeafReg";
  
  return leaf;
}


RawVector RcppLeaf::rv1 = RawVector(0);
RawVector RcppLeaf::rv2 = RawVector(0);
RawVector RcppLeaf::rv3 = RawVector(0);
NumericVector RcppLeaf::nv1 = NumericVector(0);

/**
   @brief Exposes front-end (regression) Leaf fields for transmission to core.

   @param sLeaf is the R object containing the leaf (list) data.

   @param _yTrain outputs the training response.

   @param _leafInfoReg outputs the sample counts, organized by leaf.

   @param bag indicates whether to include bagging information.

   @return void, with output reference parameters.
 */
void RcppLeaf::UnwrapReg(SEXP sLeaf, std::vector<double> &_yTrain, std::vector<unsigned int> &_leafOrigin, LeafNode *&_leafNode, unsigned int &_leafCount, BagLeaf *&_bagLeaf, unsigned int &_bagLeafTot, unsigned int *&_bagBits, bool bag) {
  List leaf(sLeaf);
  try {
    if (!leaf.inherits("LeafReg")) {
      throw std::domain_error("Expecting LeafReg");
    }
  }
  catch(std::exception &ex) {
    forward_exception_to_r(ex);
  }

  rv1 = RawVector((SEXP) leaf["bagBits"]);
  _bagBits = bag ? (unsigned int *) &rv1[0] : 0;
  
  rv2 = RawVector((SEXP) leaf["bagLeaf"]);
  _bagLeaf = bag ? (BagLeaf *) &rv2[0] : 0;
  _bagLeafTot = bag ? rv2.length() / sizeof(BagLeaf) : 0;
  
  _leafOrigin = as<std::vector<unsigned int> >(leaf["origin"]);

  rv3 = RawVector((SEXP) leaf["node"]);
  _leafNode = (LeafNode*) &rv3[0];
  _leafCount = rv3.length() / sizeof(LeafNode);

  _yTrain = as<std::vector<double> >(leaf["yTrain"]);
}


/**
   @brief Wraps core (classification) Leaf vectors for reference by front end.
 */
SEXP RcppLeaf::WrapCtg(const std::vector<unsigned int> &leafOrigin, const std::vector<LeafNode> &leafNode, const std::vector<BagLeaf> &bagLeaf, const std::vector<unsigned int> &bagBits, const std::vector<double> &weight, unsigned int rowTrain, const CharacterVector &levels) {
  RawVector leafRaw(leafNode.size() * sizeof(LeafNode));
  RawVector blRaw(bagLeaf.size() * sizeof(BagLeaf));
  RawVector bbRaw(bagBits.size() * sizeof(unsigned int));
  Serialize(leafNode, bagLeaf, bagBits, leafRaw, blRaw, bbRaw);
  List leaf = List::create(
   _["origin"] = leafOrigin,	
   _["node"] = leafRaw,
   _["bagLeaf"] = blRaw,
   _["bagBits"] = bbRaw,
   _["weight"] = weight,
   _["rowTrain"] = rowTrain,
   _["levels"] = levels
   );
  leaf.attr("class") = "LeafCtg";

  return leaf;
}


/** 
    @brief Serializes the internally-typed objects, 'LeafNode', as well
    as the unsigned integer (packed bit) vector, "bagBits".
*/
void RcppLeaf::Serialize(const std::vector<LeafNode> &leafNode, const std::vector<BagLeaf> &bagLeaf, const std::vector<unsigned int> &bagBits, RawVector &leafRaw, RawVector &blRaw, RawVector &bbRaw) {
  for (size_t i = 0; i < leafNode.size() * sizeof(LeafNode); i++) {
    leafRaw[i] = ((unsigned char*) &leafNode[0])[i];
  }

  for (size_t i = 0; i < bagLeaf.size() * sizeof(BagLeaf); i++) {
    blRaw[i] = ((unsigned char*) &bagLeaf[0])[i];
  }

  for (size_t i = 0; i < bagBits.size() * sizeof(unsigned int); i++) {
    bbRaw[i] = ((unsigned char*) &bagBits[0])[i];
  }
}


/**
   @brief Exposes front-end (classification) Leaf fields for transmission to core.

   @param sLeaf is the R object containing the leaf (list) data.

   @param _weight outputs the sample weights.

   @param _levels outputs the category levels; retains as front-end object.

   @param bag indicates whether to include bagging information.

   @return void, with output reference parameters.
 */
void RcppLeaf::UnwrapCtg(SEXP sLeaf, std::vector<unsigned int> &_leafOrigin, LeafNode *&_leafNode, unsigned int &_leafCount, BagLeaf *&_bagLeaf, unsigned int &_bagLeafTot, unsigned int *&_bagBits, double *&_weight, unsigned int &_rowTrain, CharacterVector &_levels, bool bag) {
  List leaf(sLeaf);
  try {
    if (!leaf.inherits("LeafCtg")) {
      throw std::domain_error("Expecting LeafCtg");
    }
  }
  catch(std::exception &ex) {
    forward_exception_to_r(ex);
  }

  rv1 = RawVector((SEXP) leaf["bagBits"]);
  _bagBits = bag ? (unsigned int *) &rv1[0] : 0;
  
  rv2 = RawVector((SEXP) leaf["bagLeaf"]);
  _bagLeaf = bag ? (BagLeaf *) &rv2[0] : 0;
  _bagLeafTot = bag ? rv2.length() / sizeof(BagLeaf) : 0;

  _leafOrigin = as<std::vector<unsigned int> >(leaf["origin"]);

  rv3 = RawVector((SEXP) leaf["node"]);
  _leafNode = (LeafNode*) &rv3[0];
  _leafCount = rv3.length() / sizeof(LeafNode);

  nv1 = NumericVector((SEXP) leaf["weight"]);
  _weight = &nv1[0];

  _rowTrain = as<unsigned int>((SEXP) leaf["rowTrain"]);
  _levels = as<CharacterVector>((SEXP) leaf["levels"]);
}


void RcppLeaf::Clear() {
  rv1 = RawVector(0);
  rv2 = RawVector(0);
  rv3 = RawVector(0);
  nv1 = NumericVector(0);
}
