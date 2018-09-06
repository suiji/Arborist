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

class LeafBridge {
 protected:
  vector<vector<unsigned int> > rowTree;
  vector<vector<unsigned int> > sCountTree;
  vector<vector<unsigned int> > extentTree;

 public:

  LeafBridge(unsigned int exportLength);

  /**
     @brief Accessor for per-tree sampled row vector.

     @param tIdx is the tree index.

     @return sampled row vector.
   */
  const vector<unsigned int> &getRowTree(unsigned int tIdx) const {
    return rowTree[tIdx];
  }


  /**
     @brief Accessor for per-tree sample-count vector.

     @param tIdx is the tree index.

     @return sample-count vector.
   */
  const vector<unsigned int> &getSCountTree(unsigned int tIdx) const {
    return sCountTree[tIdx];
  }


  /**
     @brief Accessor for per-tree extent vector.

     @param tIdx is the tree index.

     @return extent vector.
   */
  const vector<unsigned int> &getExtentTree(unsigned int tIdx) const {
    return extentTree[tIdx];
  }
};

/**
   @brief Bridge specialization of Core LeafReg, q.v.
 */
class LeafRegBridge : public LeafBridge {
  const IntegerVector &feOrig;
  const RawVector &feBagLeaf;
  const RawVector &feNode;
  const NumericVector &yTrain;

  vector<vector<double > > scoreTree;

  
  static SEXP Legal(const List &lLeaf);
 protected:
  unique_ptr<class LeafReg> leaf;
  
 public:
  /**
     @brief Constructor for prediction, no export.
   */
  LeafRegBridge(const IntegerVector &_feOrig,
                const RawVector &_feBagLeaf,
                const RawVector &_feNode,
                const NumericVector &_yTrain,
                unsigned int _rowPredict);

  /**
    @brief Constructor for export, no prediction.
   */
  LeafRegBridge(const IntegerVector &feOrig_,
                const RawVector &feBagLeaf_,
                const RawVector &feNode_,
                const NumericVector &yTrain_,
                const class BitMatrix *baggedRows);

  ~LeafRegBridge();
  
  static List predict(const List &list,
                         SEXP sYTest,
                         class Predict *predict);

  static unique_ptr<LeafRegBridge> unwrap(const List &leaf,
                                          unsigned int nRow);

  static unique_ptr<LeafRegBridge> unwrap(const List &lTrain,
                                          const class BitMatrix *baggedRows);

  class LeafReg *getLeaf() const {
    return leaf.get();
  }
  

  const vector<double> &getScoreTree(unsigned int tIdx) const {
    return scoreTree[tIdx];
  }
  
  List Summary(SEXP sYTest, const class Quant *quant = nullptr);

  NumericMatrix QPred(const class Quant *quant);

  /**
     @return mean-squared prediction error.
   */
  double MSE(const vector<double> &yPred,
             const NumericVector &yTest,
             double &rsq,
             double &mse);
};


/**
   @brief Bridge specialization of Core LeafCtg, q.v.
 */
class LeafCtgBridge : public LeafBridge {
  const IntegerVector &feOrig;
  const RawVector &feBagLeaf;
  const RawVector &feNode;
  const NumericVector &feWeight;
  const CharacterVector levelsTrain; // Pinned for summary reuse.

  vector<vector<double > > scoreTree;
  vector<vector<double> > weightTree;

  /**
     @brief Exception-throwing guard ensuring valid encapsulation.

     @return R-style List representing Core-generated LeafCtg.
   */
  static SEXP Legal(const List &lLeaf);

 protected:
  unique_ptr<class LeafCtg> leaf;

 public:
  /**
     @brief Constructor for prediction; no export.
   */
  LeafCtgBridge(const IntegerVector &_feOrig,
                const RawVector &_feBagLeaf,
                const RawVector &_feNode,
                const NumericVector &_feWeight,
                const CharacterVector &_feLevels,
                unsigned int _rowPredict,
                bool doProb);

  /**
     @brief Constructor for export; no prediction.
   */
  LeafCtgBridge(const IntegerVector &_feOrig,
                const RawVector &_feBagLeaf,
                const RawVector &_feNode,
                const NumericVector &_feWeight,
                const CharacterVector &_feLevels,
                const class BitMatrix *bitMatrix);

  ~LeafCtgBridge();
  
  class LeafCtg *getLeaf() const {
    return leaf.get();
  }


  static List predict(const List &list,
                  SEXP sYTest,
                  const List &signature,
                  class Predict *predict,
                  bool doProb);
  
  /**
     @brief Accessor exposes category name strings.

     @return level names vector.
   */
  const CharacterVector &getLevelsTrain() const {
    return levelsTrain;
  }


  /**
     @brief Accessor for per-tree score vector.

     @param tIdx is the tree index.

     @return score vector.
   */
  const vector<double> &getScoreTree(unsigned int tIdx) const {
    return scoreTree[tIdx];
  }


  /**
     @brief Accessor for per-tree weight vector.

     @param tIdx is the tree index.

     @return weight vector.
   */
  const vector<double> &getWeightTree(unsigned int tIdx) const {
    return weightTree[tIdx];
  }


  SEXP CtgReconcile(IntegerVector &yOneTest);

  SEXP MergeLevels(IntegerVector &yOneTest);

  /**
     @brief Instantiates front-end leaf.

     @param lTrain is the R-style trained forest.
   */
  static unique_ptr<LeafCtgBridge> unwrap(const List &leaf,
                                          unsigned int nRow,
                                          bool doProb);
  static unique_ptr<LeafCtgBridge> unwrap(const List &lTrain,
                                          const class BitMatrix *baggedRows);

  
  List Summary(SEXP sYTest, const List &signature);


  IntegerMatrix Census(const CharacterVector &rowNames);

  NumericMatrix Prob(const CharacterVector &rowNames);
};


/**
   @brief Internal back end-style vectors cache annotations for
   per-tree access.
 */
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
          unsigned int rowPredict_,
          const CharacterVector &levelsTrain_);

  static IntegerVector Reconcile(const IntegerVector &test2Train,
                                 const IntegerVector &yTestOne);
  
  static IntegerVector MergeLevels(const CharacterVector &levelsTest,
                                   const CharacterVector &levelsTrain);

  void Validate(class LeafCtg *leaf, const vector<unsigned int> &yPred);
  IntegerMatrix Confusion();
  NumericVector MisPred();
  double OOB(const vector<unsigned int> &yPred) const;
};


/**
   @brief Maintains R-style vectors represting the crescent leaf component
   of the forest during training.
 */
struct LBTrain {
  RawVector nodeRaw;
  RawVector blRaw;
  R_xlen_t nodeOff;
  R_xlen_t blOff;

  IntegerVector origin;
  LBTrain(unsigned int nTree);

  void consume(const class LeafTrain* leaf,
               unsigned int treeOff,
               double scale);

  virtual List wrap() = 0;
};

struct LBTrainReg : public LBTrain {
  const NumericVector yTrain;

  LBTrainReg(const NumericVector& yTrain_,
             unsigned int nTree);
             
  List wrap();
};


/**
   @brief Specialization for categorical leaves, which maintain an
   additional field for weights.
 */
struct LBTrainCtg : public LBTrain {
  NumericVector weight;
  R_xlen_t weightOff;
  const IntegerVector& yTrain;

  LBTrainCtg(const IntegerVector& yTrain_,
             unsigned int nTree);
  
  void consume(const class LeafTrainCtg* leaf,
               unsigned int treeOff,
               double scale);

  List wrap();
};
#endif
