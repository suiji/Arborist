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


#include "leaf.h"
#include "rcppLeaf.h"


/**
   @brief Wraps core (regression) Leaf vectors for reference by front end.
 */
RcppExport SEXP LeafWrapReg(const std::vector<unsigned int> &leafOrigin, std::vector<LeafNode> &leafNode, const std::vector<BagRow> &bagRow, unsigned int rowTrain, const std::vector<unsigned int> &rank, const std::vector<double> &yRanked) {
  unsigned int rawSize = leafNode.size() * sizeof(LeafNode);
  RawVector leafRaw(rawSize);
  for (unsigned int i = 0; i < rawSize; i++) {
    leafRaw[i] = ((unsigned char*) &leafNode[0])[i];
  }

  unsigned int BRSize = bagRow.size() * sizeof(BagRow);
  RawVector BRRaw(BRSize);
  for (unsigned int i = 0; i < BRSize; i++) {
    BRRaw[i] = ((unsigned char*) &bagRow[0])[i];
  }

  List leaf = List::create(
   _["origin"] = leafOrigin,
   _["node"] = leafRaw,
   _["bagRow"] = BRRaw,
   _["rowTrain"] = rowTrain,
   _["rank"] = rank,
   _["yRanked"] = yRanked
  );
  leaf.attr("class") = "LeafReg";
  
  return leaf;
}


/**
   @brief Exposes front-end (regression) Leaf fields for transmission to core.

   @param sLeaf is the R object containing the leaf (list) data.

   @param _yRanked outputs the sorted response.

   @param _leafInfoReg outputs the sample ranks and counts, organized by leaf.

   @return void, with output reference parameters.
 */
void LeafUnwrapReg(SEXP sLeaf, std::vector<double> &_yRanked, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, unsigned int &_rowTrain, std::vector<unsigned int> &_rank) {
  List leaf(sLeaf);
  if (!leaf.inherits("LeafReg"))
    stop("Expecting LeafReg");

  RawVector leafRaw = leaf["node"];
  unsigned int rawSize = leafRaw.length();
  std::vector<LeafNode> leafNode(rawSize / sizeof(LeafNode));
  for (unsigned int i = 0; i < rawSize; i++) {
    ((unsigned char*) &leafNode[0])[i] = leafRaw[i];
  }
  
  RawVector BRRaw = leaf["bagRow"];
  unsigned int BRSize = BRRaw.length();
  std::vector<BagRow> bagRow(BRSize / sizeof(BagRow));
  for (unsigned int i = 0; i < BRSize; i++) {
    ((unsigned char*) &bagRow[0])[i] = BRRaw[i];
  }
  
  _yRanked = as<std::vector<double> >(leaf["yRanked"]);
  _leafOrigin = as<std::vector<unsigned int>>(leaf["origin"]);
  _leafNode = std::move(leafNode);
  _bagRow = std::move(bagRow);
  _rowTrain = as<unsigned int>(leaf["rowTrain"]);
  _rank = as<std::vector<unsigned int> >(leaf["rank"]);
}


/**
   @brief Wraps core (classification) Leaf vectors for reference by front end.
 */
RcppExport SEXP LeafWrapCtg(const std::vector<unsigned int> &leafOrigin, const std::vector<LeafNode> &leafNode, const std::vector<BagRow> &bagRow, unsigned int rowTrain, const std::vector<double> &weight, const CharacterVector &levels) {
  unsigned int rawSize = leafNode.size() * sizeof(LeafNode);
  RawVector leafRaw(rawSize);
  for (unsigned int i = 0; i < rawSize; i++) {
    leafRaw[i] = ((unsigned char*) &leafNode[0])[i];
  }

  unsigned int BRSize = bagRow.size() * sizeof(BagRow);
  RawVector BRRaw(BRSize);
  for (unsigned int i = 0; i < BRSize; i++) {
    BRRaw[i] = ((unsigned char*) &bagRow[0])[i];
  }

  List leaf = List::create(
   _["origin"] = leafOrigin,	
   _["node"] = leafRaw,
   _["bagRow"] = BRRaw,
   _["rowTrain"] = rowTrain,
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

   @return void, with output reference parameters.
 */
void LeafUnwrapCtg(SEXP sLeaf, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, unsigned int &_rowTrain, std::vector<double> &_weight, CharacterVector &_levels) {
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
  
  RawVector BRRaw = leaf["bagRow"];
  unsigned int BRSize = BRRaw.length();
  std::vector<BagRow> bagRow(BRSize / sizeof(BagRow));
  for (unsigned int i = 0; i < BRSize; i++) {
    ((unsigned char*) &bagRow[0])[i] = BRRaw[i];
  }
  
  _leafOrigin = as<std::vector<unsigned int> >(leaf["origin"]);
  _leafNode = move(leafNode);
  _bagRow = move(bagRow);
  _rowTrain = as<unsigned int>(leaf["rowTrain"]);
  _weight = as<std::vector<double> >(leaf["weight"]);
  _levels = as<CharacterVector>((SEXP) leaf["levels"]);
}
