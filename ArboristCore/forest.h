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

#include <vector>

/**
   @brief To replace parallel array access.
 */
class ForestNode {
  unsigned int pred;
  unsigned int bump;
  double num;
 public:
  inline void Set(unsigned int _pred, unsigned int _bump, double _num) {
    pred = _pred;
    bump = _bump;
    num = _num;
  }


  inline void Ref(unsigned int &_pred, unsigned int &_bump, double &_num) const {
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
  const int height;

  unsigned int *treeOrigin; // Index of decision tree bases into forestNode[].
  class BVJagged *facSplit; // Consolidation of per-tree values.
  ForestNode *forestNode;

  unsigned int* fePred;  // Split predictor / sample extent : nonterminal / terminal.
  double* feNum; // Split value / score : nonterminal / terminal.
  unsigned int* feBump;  // Successor offset / zero :  nonterminal / terminal.

  void PredictAcrossNum(int *leaves, unsigned int nRow, const class BitMatrix *bag);
  void PredictAcrossFac(int *leaves, unsigned int nRow, const class BitMatrix *bag);
  void PredictAcrossMixed(int *leaves, unsigned int nRow, const class BitMatrix *bag);
 public:
  void PredictAcross(int *predictLeaves, const class BitMatrix *bag);
  
  void PredictRowNum(unsigned int row, const double rowT[], int leaves[], const class BitMatrix *bag);
  void PredictRowFac(unsigned int row, const int rowT[], int leaves[], const class BitMatrix *bag);
  void PredictRowMixed(unsigned int row, const double rowNT[], const int rowIT[], int leaves[], const class BitMatrix *bag);

  Forest(std::vector<unsigned int> &_pred, std::vector<double> &_split, std::vector<unsigned int> &_bump, std::vector<unsigned int> &_origin, const std::vector<unsigned int> &_facOrigin, const std::vector<unsigned int> &_facSplit);
  ~Forest();

  
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
  unsigned int *Nonterminal() {
    return feBump;
  }


  /**
     @brief Records sample extents for regression terminals.
   */
  unsigned int *Extent() {
    return fePred;
  }


  /**
   */
  unsigned int *Origin() {
    return treeOrigin;
  }
  

  void TreeBlock(class PreTree *ptBlock[], int treeBlock, int treeStart);

  unsigned inline int LeafPos(int treeNum, int leafIdx) const {
    return treeOrigin[treeNum] + leafIdx;
  }

  
  inline double LeafVal(int treeNum, int leafIdx) const {
    return feNum[LeafPos(treeNum, leafIdx)];
  }
};

#endif
