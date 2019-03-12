// Copyright (C)  2012-2019  Mark Seligman
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
  virtual ~LeafBridge() {}
  
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

  /**
     @brief Subclasses forget their base leaf type.
   */
  virtual class LeafFrame* getLeaf() const = 0;
};


/**
   @brief Bridge specialization of Core LeafReg, q.v.
 */
class LeafRegBridge : public LeafBridge {
  const IntegerVector &feNodeHeight;
  const RawVector &feNode;
  const IntegerVector &feBagHeight;
  const RawVector &feBagSample;
  const NumericVector &yTrain;

  vector<vector<double > > scoreTree;

  /**
     @brief Validates contents of front-end leaf object.

     Exception thrown if contents invalid.

     @return wrapped zero.
   */  
  static SEXP checkLeaf(const List &lLeaf);
 protected:
  unique_ptr<class LeafFrameReg> leaf;
  
 public:
  /**
     @brief Constructor for prediction, no export.
   */
  LeafRegBridge(const IntegerVector &feNodeHeight_,
                const RawVector &feNode_,
                const IntegerVector& feBagHeight_,
                const RawVector& feBagSample_,
                const NumericVector& yTrain_,
                unsigned int rowPredict);

  /**
    @brief Constructor for export, no prediction.
   */
  LeafRegBridge(const IntegerVector& feNodeHeight_,
                const RawVector& feNode_,
                const IntegerVector& feBagHeight_,
                const RawVector& feBagSample_,
                const NumericVector& yTrain_,
                const class BitMatrix* baggedRows);

  ~LeafRegBridge() {}
  
  static List predict(const List &list,
                         SEXP sYTest,
                         class Predict *predict);

  /**
     @brief Builds bridge object from wrapped front-end data.
   */
  static unique_ptr<LeafRegBridge> unwrap(const List& leaf,
                                          const List& sPredBlock);

  /**
     @brief Builds bridge object from wrapped front-end data.
   */
  static unique_ptr<LeafRegBridge> unwrap(const List &lTrain,
                                          const class BitMatrix *baggedRows);

  /**
     @brief Forgetful getter for pointer to core leaf representation.
   */
  class LeafFrame *getLeaf() const;
  

  const vector<double> &getScoreTree(unsigned int tIdx) const {
    return scoreTree[tIdx];
  }
  
  List summary(SEXP sYTest, const class Quant *quant = nullptr);


  /**
     @brief Builds a NumericMatrix representation of the quantile predictions.
     
     @param quant is a quantile prediction summary.

     @return transposed core matrix if quantiles requested, else empty matrix.
  */
  NumericMatrix qPred(const class Quant *quant);


  /**
     @brief Utility for computing mean-square error of prediction.
   
     @param yPred is the prediction.

     @param yTest is the observed response.

     @param rsq[out] is the r-squared statistic.

     @param mae[out] is the mean absolute error.

     @return mean squared error.
  */
  double mse(const vector<double> &yPred,
             const NumericVector &yTest,
             double &rsq,
             double &mse);
};


/**
   @brief Bridge specialization of Core LeafCtg, q.v.
 */
class LeafCtgBridge : public LeafBridge {
  const IntegerVector& feNodeHeight;
  const RawVector& feNode;
  const IntegerVector& feBagHeight;
  const RawVector& feBagSample;
  const NumericVector& feWeight;
  const CharacterVector levelsTrain; // Pinned for summary reuse.

  vector<vector<double > > scoreTree;
  vector<vector<double> > weightTree;

  /**
     @brief Exception-throwing guard ensuring valid encapsulation.

     @return R-style List representing Core-generated LeafCtg.
   */
  static SEXP checkLeaf(const List &lLeaf);

 protected:
  unique_ptr<class LeafFrameCtg> leaf;

 public:
  /**
     @brief Constructor for prediction; no export.
   */
  LeafCtgBridge(const IntegerVector& feNodeHeight_,
                const RawVector &feNode_,
                const IntegerVector& feBagheight_,
                const RawVector& feBagSample_,
                const NumericVector& feWeight_,
                const CharacterVector& feLevels_,
                unsigned int rowPredict_,
                bool doProb);

  /**
     @brief Constructor for export; no prediction.
   */
  LeafCtgBridge(const IntegerVector& feNodeHeight_,
                const RawVector& feNode_,
                const IntegerVector& feBagHeight_,
                const RawVector& feBagSample_,
                const NumericVector& feWeight_,
                const CharacterVector& feLevels_,
                const class BitMatrix* bitMatrix);

  ~LeafCtgBridge() {}

  /**
     @brief Forgetful getter to core leaf object.
   */
  class LeafFrame *getLeaf() const;

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


  /**
     @brief Instantiates front-end leaf.

     @param lTrain is the R-style trained forest.
   */
  static unique_ptr<LeafCtgBridge> unwrap(const List& leaf,
                                          const List& sPredBlock,
                                          bool doProb);

  static unique_ptr<LeafCtgBridge> unwrap(const List &lTrain,
                                          const class BitMatrix *baggedRows);

  
  List summary(SEXP sYTest, const List& sPredBlock);


  /**
     @brief Produces census summary, which is common to all categorical
     prediction.

     @param rowNames is the user-supplied specification of row names.

     @return matrix of predicted categorical responses, by row.
  */
  IntegerMatrix Census(const CharacterVector &rowNames);

  
  /**
     @param rowNames is the user-supplied collection of row names.

     @return probability matrix if requested, otherwise empty matrix.
  */
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
  
  /**
     @brief Reconciles factor encodings of training and test responses.
   */
  static IntegerVector mergeLevels(const CharacterVector &levelsTest,
                                   const CharacterVector &levelsTrain);


  /**
     @brief Fills in confusion matrix and misprediction vector.

     @param leaf summarizes the trained leaf frame.

     @param yPred contains the zero-based predictions.
  */
  void validate(class LeafFrameCtg *leaf,
                const vector<unsigned int> &yPred);


  IntegerMatrix Confusion();
  NumericVector MisPred();
  double OOB(const vector<unsigned int> &yPred) const;
};


/**
   @brief Maintains R-style vectors represting the crescent leaf component
   of the forest during training.
 */
struct LBTrain {
private:
  static bool thin; // User option:  whether to annotate bag state.

  /**
     @brief Consumes core Node recrods and writes as raw data.

     @param leaf is the core representation of a trained leaf.

     @param tIdx is the absolute tree index.

     @param scale estimates a resizing factor.
   */
  void writeNode(const class LFTrain* leaf,
                 unsigned int tIdx,
                 double scale);
  
  /**
     @brief Consumes the BagSample records and writes as raw data.
   */
  void writeBagSample(const class LFTrain* leaf,
                    unsigned int treeOff,
                    double scale);
public:
  IntegerVector nodeHeight;  // Accumulated per-tree extent of Leaf vector.
  RawVector nodeRaw; // Packed node structures as raw data.

  IntegerVector bagHeight; // Accumulated per-tree extent of BagSample vector.
  RawVector blRaw; // Packed bag/sample structures as raw data.

  /**
     @brief Constructor.

     @param nTree is the number of trees over which to train.
   */
  LBTrain(unsigned int nTree);

  virtual ~LBTrain() {}
  
  /**
     @brief Static initialization.

     @param thin_ indicates whether certain annotations may be omitted.
   */
  static void init(bool thin_);

  /**
     @brief Resets static initializations.
   */
  static void deInit();


  /**
     @brief High-level entry for writing contents of a tree's leaves.

     @param leaf is the core represenation of the trained leaves.

     @param tIdx is the absolute index of the tree.

     @param scale estimates a resizing factor.
   */
  virtual void consume(const class LFTrain* leaf,
               unsigned int treeOff,
               double scale);

  /**
     @brief Packages contents for storage by front end.

     @return named list of summary fields.
   */
  virtual List wrap() = 0;
};


struct LBTrainReg : public LBTrain {
  const NumericVector yTrain; // Training response.

  LBTrainReg(const NumericVector& yTrain_,
             unsigned int nTree);

  ~LBTrainReg() {}

  /**
     @brief Description and parameters as with virutal declaration.
   */
  void consume(const class LFTrain* leaf,
               unsigned int treeOff,
               double scale);
  /**
     @brief Description as with virtual declaration.s
   */
  List wrap();
};


/**
   @brief Specialization for categorical leaves, which maintain an
   additional field for weights.
 */
struct LBTrainCtg : public LBTrain {
  NumericVector weight; // Per-category probabilities.
  R_xlen_t weightSize; // Running Size of weight vector.  Not saved.
  const IntegerVector& yTrain; // Training response.

  LBTrainCtg(const IntegerVector& yTrain_,
             unsigned int nTree);

  ~LBTrainCtg() {}

  /**
     @brief Description and parameters as with virtual declaration.
   */
  void consume(const class LFTrain* leaf,
               unsigned int treeOff,
               double scale);

  /**
     @brief Description as with virtual declaration.
   */
  List wrap();


private:
  /**
     @brief Writes leaf weights from core representation.

     @param leaf is the core representation of a tree's leaves.

     @param tIdx is the absolute tree index.

     @double scale estimates a resizing factor.
   */
  void writeWeight(const class LFTrainCtg* leaf,
                   unsigned int tIdx,
                   double scale);

};
#endif
