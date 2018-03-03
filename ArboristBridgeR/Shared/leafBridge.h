// Copyright (C)  2012-2018  Mark Seligman
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
   @file leafBridge.h

   @brief C++ class definitions for managing Leaf object.

   @author Mark Seligman

 */


#ifndef ARBORIST_LEAF_BRIDGE_H
#define ARBORIST_LEAF_BRIDGE_H

#include <Rcpp.h>
using namespace Rcpp;

#include "leaf.h"


/**
   @brief Bridge specialization of Core LeafReg, q.v.
 */
class LeafRegBridge : public LeafReg {
  const IntegerVector &feOrig;
  const RawVector &feBagBits;
  const RawVector &feBagLeaf;
  const RawVector &feNode;
  const NumericVector &yTrain;
  static SEXP Legal(const List &list);

 public:
  LeafRegBridge(const IntegerVector &_feOrig,
		const RawVector &_feBagBits,
		const RawVector &_feBagLeaf,
		const RawVector &_feNode,
		const NumericVector &_yTrain,
		bool aux);


  static unique_ptr<LeafRegBridge> Unwrap(const List &leaf,
					  bool aux = false);
};


class LeafBridge {

 public:

  static List Wrap(LeafTrainReg *leafReg, const NumericVector &yTrain);

  /**
     @brief Bundles Core LeafCtg as R-style List.

     @param leafCtg is a Core-generated summary of categorical leaves.

     @param levels contains the front-end category string names.

     @return bundled list.
   */
  static List Wrap(LeafTrainCtg *leafCtg, const CharacterVector &levels);
};


/**
   @brief Bridge specialization of Core LeafCtg, q.v.
 */
class LeafCtgBridge : public LeafCtg {
  const IntegerVector &feOrig;
  const RawVector &feBagBits;
  const RawVector &feBagLeaf;
  const RawVector &feNode;
  const NumericVector &feWeight;
  const CharacterVector &feLevels;

  /**
     @brief Exception-throwing guard ensuring valid encapsulation.

     @return R-style List representing Core-generated LeafCtg.
   */
  static SEXP Legal(const List &list);

 public:
  /**
     @brief Invokes Core constructor and pins front-end vectors for scope.
   */
  LeafCtgBridge(const IntegerVector &_feOrig,
		const RawVector &_feBagBits,
		const RawVector &_feBagLeaf,
		const RawVector &_feNode,
		const NumericVector &_feWeight,
		unsigned int _feRowTrain,
		const CharacterVector &_feLevels,
		    bool aux);

  /**
     @brief Accessor exposes category name strings.

     @return vector of string names.
   */
  const CharacterVector &Levels() const {
    return feLevels;
  }
  
  static unique_ptr<LeafCtgBridge> Unwrap(const List &leaf,
				   bool aux = false);

  static unique_ptr<LeafCtgBridge> Unwrap(const SEXP sLeaf,
				   bool aux) {
    return Unwrap(List(sLeaf), aux);
  }
};


class LeafExportCtg {
  const LeafCtgBridge *leaf;
  unsigned int nTree;
  unsigned int rowTrain;

  vector<vector<unsigned int> > rowTree;
  vector<vector<unsigned int> > sCountTree;
  vector<vector<unsigned int> > extentTree;
  vector<vector<double > > scoreTree;
  vector<vector<double> > weightTree;

 public:
  LeafExportCtg(const List &leaf, bool aux);

  unsigned int RowTrain() {
    return rowTrain;
  }
  
  const vector<vector<unsigned int> > &RowTree() {
    return rowTree;
  }


  const vector<vector<unsigned int> > &SCountTree() {
    return sCountTree;
  }


  const vector<vector<unsigned int> > &ExtentTree() {
    return extentTree;
  }

  
  const vector<vector<double> > &ScoreTree() {
    return scoreTree;
  }

  
  const vector<vector<double> > &WeightTree() {
    return weightTree;
  }

  const CharacterVector &YLevel() const {
    return leaf->Levels();
  }
};


class LeafExportReg {
  const LeafRegBridge *leaf;
  unsigned int nTree;
  unsigned int rowTrain;

  vector<vector<unsigned int> > rowTree;
  vector<vector<unsigned int> > sCountTree;
  vector<vector<unsigned int> > extentTree;
  vector<vector<double > > scoreTree;
  
 public:
  LeafExportReg(const List &leaf, bool aux);

  unsigned int RowTrain() {
    return rowTrain;
  }
  
  const vector<vector<unsigned int> > &RowTree() {
    return rowTree;
  }


  const vector<vector<unsigned int> > &SCountTree() {
    return sCountTree;
  }


  const vector<vector<unsigned int> > &ExtentTree() {
    return extentTree;
  }

  
  const vector<vector<double> > &ScoreTree() {
    return scoreTree;
  }
};

#endif
