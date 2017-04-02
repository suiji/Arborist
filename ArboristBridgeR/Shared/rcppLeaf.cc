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
  // Serializes the two internally-typed objects, 'LeafNode' and 'BagLeaf'.
  //
  unsigned int rawSize = leafNode.size() * sizeof(LeafNode);
  RawVector leafRaw(rawSize);
  for (unsigned int i = 0; i < rawSize; i++) {
    leafRaw[i] = ((unsigned char*) &leafNode[0])[i];
  }

  unsigned int BRSize = bagLeaf.size() * sizeof(BagLeaf);
  RawVector BRRaw(BRSize);
  for (unsigned int i = 0; i < BRSize; i++) {
    BRRaw[i] = ((unsigned char*) &bagLeaf[0])[i];
  }

  List leaf = List::create(
   _["origin"] = leafOrigin,
   _["node"] = leafRaw,
   _["bagLeaf"] = BRRaw,
   _["bagBits"] = bagBits,
   _["yTrain"] = yTrain
  );
  leaf.attr("class") = "LeafReg";
  
  return leaf;
}


/**
   @brief Exposes front-end (regression) Leaf fields for transmission to core.

   @param sLeaf is the R object containing the leaf (list) data.

   @param _yTrain outputs the training response.

   @param _leafInfoReg outputs the sample counts, organized by leaf.

   @param bag indicates whether to include bagging information.

   @return void, with output reference parameters.
 */
void RcppLeaf::UnwrapReg(SEXP sLeaf, std::vector<double> &_yTrain, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, bool bag) {
  List leaf(sLeaf);
  if (!leaf.inherits("LeafReg"))
    stop("Expecting LeafReg");

  // Deserializes:
  //
  RawVector leafRaw = leaf["node"];
  unsigned int rawSize = leafRaw.length();
  std::vector<LeafNode> leafNode(rawSize / sizeof(LeafNode));
  for (unsigned int i = 0; i < rawSize; i++) {
    ((unsigned char*) &leafNode[0])[i] = leafRaw[i];
  }

  if (bag) {
    RawVector BRRaw = leaf["bagLeaf"];
    unsigned int BRSize = BRRaw.length();
    std::vector<BagLeaf> bagLeaf(BRSize / sizeof(BagLeaf));
    for (unsigned int i = 0; i < BRSize; i++) {
      ((unsigned char*) &bagLeaf[0])[i] = BRRaw[i];
    }
  _bagLeaf = std::move(bagLeaf);
  _bagBits = as<std::vector<unsigned int> >(leaf["bagBits"]);
  }
  else {
    _bagBits = std::vector<unsigned int>(0);
  }
  
  _yTrain = as<std::vector<double> >(leaf["yTrain"]);
  _leafOrigin = as<std::vector<unsigned int>>(leaf["origin"]);
  _leafNode = std::move(leafNode);
}


/**
   @brief Wraps core (classification) Leaf vectors for reference by front end.
 */
SEXP RcppLeaf::WrapCtg(const std::vector<unsigned int> &leafOrigin, const std::vector<LeafNode> &leafNode, const std::vector<BagLeaf> &bagLeaf, const std::vector<unsigned int> &bagBits, const std::vector<double> &weight, const CharacterVector &levels) {
  unsigned int rawSize = leafNode.size() * sizeof(LeafNode);
  RawVector leafRaw(rawSize);
  for (unsigned int i = 0; i < rawSize; i++) {
    leafRaw[i] = ((unsigned char*) &leafNode[0])[i];
  }

  size_t BRSize = bagLeaf.size() * sizeof(BagLeaf);
  RawVector BRRaw(BRSize);
  for (size_t i = 0; i < BRSize; i++) {
    BRRaw[i] = ((unsigned char*) &bagLeaf[0])[i];
  }

  List leaf = List::create(
   _["origin"] = leafOrigin,	
   _["node"] = leafRaw,
   _["bagLeaf"] = BRRaw,
   _["bagBits"] = bagBits,
   _["weight"] = weight,
   _["levels"] = levels
   );
  leaf.attr("class") = "LeafCtg";

  return leaf;
}


/**
   @brief Exposes front-end (classification) Leaf fields for transmission to core.

   @param sLeaf is the R object containing the leaf (list) data.

   @param _weight outputs the sample weights.

   @param _levels outputs the category levels; retains as front-end object.

   @param bag indicates whether to include bagging information.

   @return void, with output reference parameters.
 */
void RcppLeaf::UnwrapCtg(SEXP sLeaf, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, std::vector<double> &_weight, CharacterVector &_levels, bool bag) {
  List leaf(sLeaf);
  if (!leaf.inherits("LeafCtg")) {
    stop("Expecting LeafCtg");
  }
  RawVector leafRaw = leaf["node"];
  unsigned int rawSize = leafRaw.length();
  std::vector<LeafNode> leafNode(rawSize / sizeof(LeafNode));
  for (unsigned int i = 0; i < rawSize; i++) {
    ((unsigned char*) &leafNode[0])[i] = leafRaw[i];
  }

  if (bag) {
    RawVector BRRaw = leaf["bagLeaf"];
    size_t BRSize = BRRaw.length();
    std::vector<BagLeaf> bagLeaf(BRSize / sizeof(BagLeaf));
    for (size_t i = 0; i < BRSize; i++) {
      ((unsigned char*) &bagLeaf[0])[i] = BRRaw[i];
    }
    _bagLeaf = move(bagLeaf);
    _bagBits = as<std::vector<unsigned int> >(leaf["bagBits"]);
  }
  else {
    _bagBits = std::vector<unsigned int>(0);
  }
  _leafOrigin = as<std::vector<unsigned int> >(leaf["origin"]);
  _leafNode = move(leafNode);
  _weight = as<std::vector<double> >(leaf["weight"]);
  _levels = as<CharacterVector>((SEXP) leaf["levels"]);
}
