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
   @brief To replace parallel array access.
 */
class ForestNode {
  int pred;
  unsigned int bump;
  double num;
 public:
  inline void Set(int _pred, unsigned int _bump, double _num) {
    pred = _pred;
    bump = _bump;
    num = _num;
  }


  inline void Ref(int &_pred, unsigned int &_bump, double &_num) const {
    _pred = pred;
    _bump = bump;
    _num = num;
  }
};

/**
   @brief The decision forest is a collection of decision trees.  DecTree members and methods are currently all static.
*/
class Forest {
  const int nTree;

  int *treeOrigin; // Index of of decision tree bases into forestNode[].
  const int *facOrigin; // Offset of factor splitting bases into facSplit[];
  unsigned int *facSplit; // Consolidation of per-tree values.
  ForestNode *forestNode;

  int* fePred;  // Split predictor / sample extent : nonterminal / terminal.
  double* feNum; // Split value / score : nonterminal / terminal.
  int* feBump;  // Successor offset / zero :  nonterminal / terminal.
  const int height;

  /**
     @brief Sets the decision tree and factor splitting bases for a tree.

     @param tc is the index of the tree.

     @return void with output reference parameters.
   */
  inline void TreeBases(int tc, ForestNode *&treeBase, unsigned int *&bitBase) {
    treeBase = forestNode + treeOrigin[tc];
    bitBase = facSplit + facOrigin[tc];
  } 

 public:
  void PredictAcrossNum(class Predict *predict, int *leaves, unsigned int nRow, const unsigned int *bag);
  void PredictAcrossFac(class Predict *predict, int *leaves, unsigned int nRow, const unsigned int *bag);
  void PredictAcrossMixed(class Predict *predict, int *leaves, unsigned int nRow, const unsigned int *bag);
  void PredictRowNum(unsigned int row, const double rowT[], int leaves[], const unsigned int bag[]);
  void PredictRowFac(unsigned int row, const int rowT[], int leaves[], const unsigned int bag[]);
  void PredictRowMixed(unsigned int row, const double rowNT[], const int rowIT[], int leaves[], const unsigned int bag[]);

  Forest(int _nTree, int _height, int _preds[], double _splits[], int _bump[], int _origins[], int _facOrigin[], unsigned int _facSplit[]);
  ~Forest();

  static inline int Encode(int predIdx, bool isFactor) {
    return isFactor ? -(1 + predIdx) : predIdx;
  }

  static inline bool IsFactor(int predIdx) {
    return predIdx < 0;
  }

  static inline unsigned int Decode(int predIdx, bool &isFactor) {
    isFactor = IsFactor(predIdx);
    return isFactor ? -(1 + predIdx) : predIdx;
  }
  
  /**
     @brief Accessor for tree count.
   */
  int NTree() {
    return nTree;
  }

  
  /*
    @brief Accessor for forest-wide vector length.

   */
  int Height() {
    return height;
  }

  
  /**
     @brief Bump values nonzero iff nonterminal.
   */
  int *Nonterminal() {
    return feBump;
  }


  /**
     @brief Records sample extents for regression terminals.
   */
  int *Extent() {
    return fePred;
  }


  /**
   */
  int *Origin() {
    return treeOrigin;
  }
  
  static void Immutables(class Predict *_predict);
  static void DeImmutables();

  void TreeBlock(class PreTree *ptBlock[], int treeBlock, int treeStart);
  bool InBag(const unsigned int bag[], int treeNum, unsigned int row);

  inline int LeafPos(int treeNum, int leafIdx) const {
    return treeOrigin[treeNum] + leafIdx;
  }

  
  inline double LeafVal(int treeNum, int leafIdx) {
    return feNum[LeafPos(treeNum, leafIdx)];
  }
  static void BagSet(unsigned int bag[], int _nTree, unsigned int treeNum, unsigned int row);
};

#endif
