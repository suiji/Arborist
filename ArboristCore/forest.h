// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.h

   @brief Data structures and methods for constructing and walking
       the decision tree.

   @author Mark Seligman
 */

#ifndef ARBORIST_FOREST_H
#define ARBORIST_FOREST_H


/**
   @brief The decision forest is a collection of decision trees.  DecTree members and methods are currently all static.
*/
class Forest {
 protected:
  const int nTree;
  static unsigned int nRow; // Varies across predictions, as well as training.
  static int nPred;
  static int nPredNum;
  static int nPredFac;
  static Forest *forest;

  int *treeOrigin;
  int *facOff;
  int *facSplit; // Consolidation of per-tree values.

  int* pred;  // Split predictor / sample extent : nonterminal / terminal.
  double* num; // Split value / score : nonterminal / terminal.
  int* bump;  // Successor offset / zero :  nonterminal / terminal.

  int forestSize;
  void PredictAcross(int predictLeaves[], const unsigned int bag[]);
  void PredictAcrossNum(int predictLeaves[], const unsigned int bag[]);
  void PredictAcrossFac(int predictLeaves[], const unsigned int bag[]);
  void PredictAcrossMixed(int predictLeaves[], const unsigned int bag[]);
  void PredictRowNum(unsigned int row, double rowT[], int leaves[], const unsigned int bag[]);
  void PredictRowFac(unsigned int row, int rowT[], int leaves[], const unsigned int bag[]);
  void PredictRowMixed(unsigned int row, double rowNT[], int rowIT[], int leaves[], const unsigned int bag[]);

 public:
  Forest(int _nTree, int _nRow);
  Forest(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], int _facSplit[]);
  virtual ~Forest() {}

  static int ValidateCtg(double y[], int *census, int *conf, double error[], double *prob);
  static int PredictCtg(double y[], int *census, double *prob);

  
  /**
     @brief Accessor for tree count.
   */
  int NTree() {
    return nTree;
  }

  static unsigned int PredImmutables();
  static void PredDeImmutables();

  void TreeBlock(class PreTree *ptBlock[], int treeBlock, int treeStart);


  static class ForestReg *FactoryReg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], int _facSplit[], int _rank[], int _sCount[], double _yRanked[]);

  static class ForestCtg *FactoryCtg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], int _facSplit[], unsigned int _ctgWidth, double _leafWeight[]);

  static void DeFactory(Forest *forest);

/**
   @brief Determines whether a given row index is in-bag in a given tree.

   @param treeNum is the index of a given tree.

   @param row is the row index to be tested.

   @return True iff the row is in-bag.
 */
  bool inline InBag(const unsigned int bag[], int treeNum, unsigned int row) {
    if (bag == 0)
      return false;

    const unsigned int slotBits = 8 * sizeof(unsigned int);
    unsigned int idx = row * nTree + treeNum;
    unsigned int off = idx / slotBits; // Compiler should generate right-shift.
    unsigned int bit = idx & (slotBits - 1);

    return (bag[off] & (1 << bit)) != 0;
  }


  /**
     @brief Sets the in-bag bit for a given <tree, row> pair.

     @param bag is the forest-wide bit matrix.

     @param _nTree is the number of trees being grown.

     @param treeNum is the current tree index.

     @param row is the row index.

     @return void.
   */
  static void BagSet(unsigned int bag[], int _nTree, unsigned int treeNum, unsigned int row) {
    const unsigned int slotBits = 8 * sizeof(unsigned int);
    unsigned int idx = row * _nTree + treeNum;
    unsigned int off = idx / slotBits; // Compiler should generate right-shift.
    unsigned int bit = idx & (slotBits - 1);
    bag[off] |= (1 << bit);
  }
};


class ForestReg : public Forest {
  int totBagCount;
  int *rank;
  int *sCount;
  double *yRanked;
 public:
  ForestReg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], int _facSplit[], int _rank[], int _sCount[], double _yRanked[]);
  int QuantFields(int &_nTree, unsigned int &_nRow, int *&_origin, int *&_nonTerm, int *&_extent, double *&_yRanked, int *&_rank, int *&_sCount) const;
  void Score(int predictLeaves[], double yPred[]);
  void Predict(double yPred[], int predictLeaves[], const unsigned int bag[]);
};


class ForestCtg : public Forest {
  unsigned int ctgWidth;
 public:
  double *leafWeight;
  double **treeLeafWeight;
  ForestCtg(int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], int _facSplit[], unsigned int _ctgWidth, double *_leafWeights);
  void Predict(int *predictLeaves, const unsigned int bag[]);
  double *Score(int *predictLeaves);
  void Prob(int *predictLeaves, double *prob);
};

#endif
