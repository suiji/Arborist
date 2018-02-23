// Copyright (C)  2012-2018   Mark Seligman
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
   @file leafBridge.cc

   @brief C++ interface to R entry for Leaf methods.

   @author Mark Seligman
 */

#include "leafBridge.h"


/**
   @brief Wraps core (regression) Leaf vectors for reference by front end.
 */
List LeafBridge::Wrap(LeafTrainReg *leafReg, const NumericVector &yTrain) {
  RawVector leafRaw(leafReg->NodeBytes());
  RawVector blRaw(leafReg->BLBytes());
  RawVector bbRaw(leafReg->BagBytes());
  leafReg->Serialize((unsigned char *) &leafRaw[0], (unsigned char*) &blRaw[0], (unsigned char *) &bbRaw[0]);
  List leaf = List::create(
   _["origin"] = leafReg->Origin(),
   _["node"] = leafRaw,
   _["bagLeaf"] = blRaw,
   _["bagBits"] = bbRaw,
   _["yTrain"] = yTrain
  );
  leaf.attr("class") = "LeafReg";
  delete leafReg;
  
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
LeafRegBridge *LeafRegBridge::Unwrap(const List &leaf, bool aux) {
  Legal(leaf);
  return new LeafRegBridge(IntegerVector((SEXP) leaf["origin"]),
			   RawVector((SEXP) leaf["bagBits"]),
			   RawVector((SEXP) leaf["bagLeaf"]),
			   RawVector((SEXP) leaf["node"]),
			   NumericVector((SEXP) leaf["yTrain"]),
			   aux);
}


SEXP LeafRegBridge::Legal(const List &leaf) {
  BEGIN_RCPP

  if (!leaf.inherits("LeafReg")) {
    stop("Expecting LeafReg");
  }

  END_RCPP
}


LeafRegBridge::LeafRegBridge(const IntegerVector &_feOrig,
			     const RawVector &_feBagBits,
			     const RawVector &_feBagLeaf,
			     const RawVector &_feNode,
			     const NumericVector &_yTrain,
			     bool aux) :
  LeafReg((unsigned int *) &_feOrig[0],
	      _feOrig.length(),
	      (LeafNode*) &_feNode[0],
	      _feNode.length()/sizeof(LeafNode),
	      aux ? (BagLeaf*) &_feBagLeaf[0] : nullptr,
	      aux ? _feBagLeaf.length() / sizeof(BagLeaf) : 0,
	      (unsigned int *) &_feBagBits[0],
	      &_yTrain[0],
	      _yTrain.length(),
	      mean(_yTrain)),  
  feOrig(_feOrig),
  feBagBits(_feBagBits),
  feBagLeaf(_feBagLeaf),
  feNode(_feNode),
  yTrain(_yTrain) {
}




/**
   @brief Wraps core (classification) Leaf vectors for reference by front end.
 */
List LeafBridge::Wrap(LeafTrainCtg *leafCtg, const CharacterVector &levels) {
  RawVector leafRaw(leafCtg->NodeBytes());
  RawVector blRaw(leafCtg->BLBytes());
  RawVector bbRaw(leafCtg->BagBytes());
  leafCtg->Serialize((unsigned char *) &leafRaw[0], (unsigned char *) &blRaw[0], (unsigned char *) &bbRaw[0]);
  List leaf = List::create(
   _["origin"] = leafCtg->Origin(),
   _["node"] = leafRaw,
   _["bagLeaf"] = blRaw,
   _["bagBits"] = bbRaw,
   _["weight"] = leafCtg->Weight(),
   _["rowTrain"] = leafCtg->RowTrain(),
   _["levels"] = levels
   );
  leaf.attr("class") = "LeafCtg";

  delete leafCtg;
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
LeafCtgBridge *LeafCtgBridge::Unwrap(const List &leaf,
			 bool aux) {
  Legal(leaf);
  return new LeafCtgBridge(IntegerVector((SEXP) leaf["origin"]),
			   RawVector((SEXP) leaf["bagBits"]),
			   RawVector((SEXP) leaf["bagLeaf"]),
			   RawVector((SEXP) leaf["node"]),
			   NumericVector((SEXP) leaf["weight"]),
			   as<unsigned int>((SEXP) leaf["rowTrain"]),
			   as<CharacterVector>((SEXP) leaf["levels"]),
			   aux);
}


SEXP LeafCtgBridge::Legal(const List &leaf) {
  BEGIN_RCPP

  if (!leaf.inherits("LeafCtg")) {
    stop("Expecting LeafCtg");
  }

  END_RCPP
}


LeafCtgBridge::LeafCtgBridge(const IntegerVector &_feOrig,
			     const RawVector &_feBagBits,
			     const RawVector &_feBagLeaf,
			     const RawVector &_feNode,
			     const NumericVector &_feWeight,
			     unsigned int _feRowTrain,
			     const CharacterVector &_feLevels,
			     bool aux) :
  // Ctg prediction does not employ BagLeaf information.
  LeafCtg((unsigned int *) &_feOrig[0],
	      _feOrig.length(),
	      (LeafNode*) &_feNode[0],
	      _feNode.length()/sizeof(LeafNode),
	      aux ? (BagLeaf*) &_feBagLeaf[0] : nullptr,
	      aux ? _feBagLeaf.length() / sizeof(BagLeaf) : 0,
	      (unsigned int *) &_feBagBits[0],
	      _feRowTrain,
	      &_feWeight[0],
	      _feLevels.length()),
  feOrig(_feOrig),
  feBagBits(_feBagBits),
  feBagLeaf(_feBagLeaf),
  feNode(_feNode),
  feWeight(_feWeight),
  feLevels(_feLevels) {
}


LeafExportCtg::LeafExportCtg(const List &_leaf, bool aux) :
  leaf(LeafCtgBridge::Unwrap(_leaf, aux)),
  nTree(leaf->NTree()),
  rowTree(vector<vector<unsigned int> >(nTree)),
  sCountTree(vector<vector<unsigned int> >(nTree)),
  extentTree(vector<vector<unsigned int> >(nTree)),
  scoreTree(vector<vector<double > >( nTree)),
  weightTree(vector<vector<double> >(nTree)) {
  leaf->Export(rowTrain, rowTree, sCountTree, scoreTree, extentTree, weightTree);
}


LeafExportReg::LeafExportReg(const List &_leaf, bool aux) :
  leaf(LeafRegBridge::Unwrap(_leaf, aux)),
  nTree(leaf->NTree()),
  rowTree(vector<vector<unsigned int> >(nTree)),
  sCountTree(vector<vector<unsigned int> >(nTree)),
  extentTree(vector<vector<unsigned int> >(nTree)),
  scoreTree(vector<vector<double > >( nTree)) {
  leaf->Export(rowTrain, rowTree, sCountTree, scoreTree, extentTree);
}
