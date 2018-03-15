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

#include <vector>
#include <memory>
using namespace std;


/**
   @brief Bridge specialization of Core LeafReg, q.v.
 */
class LeafRegBridge {
  const IntegerVector &feOrig;
  const RawVector &feBagBits;
  const RawVector &feBagLeaf;
  const RawVector &feNode;
  const NumericVector &yTrain;

  static SEXP Legal(const List &list);
 protected:
  unique_ptr<class LeafReg> leaf;
  
 public:
  LeafRegBridge(const IntegerVector &_feOrig,
		const RawVector &_feBagBits,
		const RawVector &_feBagLeaf,
		const RawVector &_feNode,
		const NumericVector &_yTrain,
		unsigned int _rowPredict);

  LeafRegBridge(const IntegerVector &_feOrig,
		const RawVector &_feBagBits,
		const RawVector &_feBagLeaf,
		const RawVector &_feNode,
		const NumericVector &_yTrain,
		unsigned int _rowPredict,
		const NumericVector &_quantiles,
		const unsigned int _qBin);

  static List Prediction(const List &list,
			 SEXP sYTest,
			 class Predict *predict);

  static List Prediction(const List &list,
			 SEXP sYTest,
			 class Predict *predict,
			 const NumericVector &quantVec,
			 unsigned int qBin);
  
  static unique_ptr<LeafRegBridge> Unwrap(const List &leaf,
					  unsigned int nRow);


  static unique_ptr<LeafRegBridge> Unwrap(const List &leaf,
					  unsigned int nRow,
					  const NumericVector &sQuantVec,
					  unsigned int qBin);

  class LeafReg *GetLeaf() const {
    return leaf.get();
  }
  
  List Summary(SEXP sYTest);

  NumericMatrix QPred();
  
  double MSE(const vector<double> &yPred,
	     const NumericVector &yTest,
	     double &rsq,
	     double &mse);
};


class LeafBridge {

 public:

  static List Wrap(class LeafTrainReg *leafReg, const NumericVector &yTrain);

  /**
     @brief Bundles Core LeafCtg as R-style List.

     @param leafCtg is a Core-generated summary of categorical leaves.

     @param levels contains the front-end category string names.

     @return bundled list.
   */
  static List Wrap(class LeafTrainCtg *leafCtg,
		   const CharacterVector &levels);
};


/**
   @brief Bridge specialization of Core LeafCtg, q.v.
 */
class LeafCtgBridge {
  const IntegerVector &feOrig;
  const RawVector &feBagBits;
  const RawVector &feBagLeaf;
  const RawVector &feNode;
  const NumericVector &feWeight;
  const CharacterVector levelsTrain; // Pinned for summary reuse.

  /**
     @brief Exception-throwing guard ensuring valid encapsulation.

     @return R-style List representing Core-generated LeafCtg.
   */
  static SEXP Legal(const List &list);

 protected:
  unique_ptr<class LeafCtg> leaf;

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
		unsigned int _rowPredict,
		bool doProb);

  class LeafCtg *GetLeaf() const {
    return leaf.get();
  }


  static List Prediction(const List &list,
		  SEXP sYTest,
		  const List &signature,
		  class Predict *predict,
		  bool doProb);
  
  /**
     @brief Accessor exposes category name strings.

     @return vector of string names.
   */
  const CharacterVector &LevelsTrain() const {
    return levelsTrain;
  }


  SEXP CtgReconcile(IntegerVector &yOneTest);

  SEXP MergeLevels(IntegerVector &yOneTest);

  static unique_ptr<LeafCtgBridge> Unwrap(const List &leaf,
					  unsigned int nRow,
					  bool doProb);

  List Summary(SEXP sYTest, const List &signature);


  IntegerMatrix Census(const CharacterVector &rowNames);

  NumericMatrix Prob(const CharacterVector &rowNames);
};


class LeafExportCtg : public LeafCtgBridge {
  unsigned int nTree;
  unsigned int rowTrain;

  vector<vector<unsigned int> > rowTree;
  vector<vector<unsigned int> > sCountTree;
  vector<vector<unsigned int> > extentTree;
  vector<vector<double > > scoreTree;
  vector<vector<double> > weightTree;

 public:
  LeafExportCtg(const List &leaf);

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

};


class TestCtg {
  const unsigned int rowPredict;
  const CharacterVector levelsTrain;
  const IntegerVector yTestOne;
  const CharacterVector levels;
  const unsigned int nCtg;
  const IntegerVector test2Merged;
  const IntegerVector yTestZero;
  const unsigned int ctgMerged;
  NumericVector misPred;
  vector<unsigned int> confusion;

 public:
  TestCtg(SEXP sYTest,
	  unsigned int _rowPredict,
	  const CharacterVector &_levelsTrain);
  static IntegerVector Reconcile(const IntegerVector &test2Train,
				 const IntegerVector &yTestOne);
  
  static IntegerVector MergeLevels(const CharacterVector &levelsTest,
				   const CharacterVector &levelsTrain);

  void Validate(class LeafCtg *leaf, const vector<unsigned int> &yPred);
  IntegerMatrix Confusion();
  NumericVector MisPred();
  double OOB(const vector<unsigned int> &yPred) const;
};


class LeafExportReg : public LeafRegBridge {
  unsigned int nTree;
  unsigned int rowTrain;

  vector<vector<unsigned int> > rowTree;
  vector<vector<unsigned int> > sCountTree;
  vector<vector<unsigned int> > extentTree;
  vector<vector<double > > scoreTree;
  
 public:
  LeafExportReg(const List &leaf);

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
