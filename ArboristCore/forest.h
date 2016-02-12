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

  inline unsigned int &Pred() {
    return pred;
  }

  inline double &Num() {
    return num;
  }


  inline void ScoreAccum(double incr) {
    num += incr;
  }


  inline void ScoreScale(unsigned int sCount) {
    num /= sCount;
  }


  inline double Score() const {
    return num;
  }


  /**
     @brief Accessor for building leaf count.
   */
  inline unsigned int &LeafCount() {
    return pred;
  }

  
  /**
     @brief Accessor for final leaf count;
   */
  inline unsigned int Extent() const {
    return pred;
  }

  
  /**
     @return True iff bump value is nonzero.
   */
  inline bool Nonterminal() const {
    return bump != 0;
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

  std::vector<ForestNode> &forestNode;
  std::vector<unsigned int> &treeOrigin;
  std::vector<unsigned int> &facOrigin;
  std::vector<unsigned int> &facVec;
  //  unsigned int *treeOrigin; // Index of decision tree bases into forestNode[].
  class BVJagged *facSplit; // Consolidation of per-tree values.

  void PredictAcrossNum(int *leaves, unsigned int nRow, const class BitMatrix *bag);
  void PredictAcrossFac(int *leaves, unsigned int nRow, const class BitMatrix *bag);
  void PredictAcrossMixed(int *leaves, unsigned int nRow, const class BitMatrix *bag);
 public:
  //  ForestNode *forestNode;
  void PredictAcross(int *predictLeaves, const class BitMatrix *bag);
  
  void PredictRowNum(unsigned int row, const double rowT[], int leaves[], const class BitMatrix *bag);
  void PredictRowFac(unsigned int row, const int rowT[], int leaves[], const class BitMatrix *bag);
  void PredictRowMixed(unsigned int row, const double rowNT[], const int rowIT[], int leaves[], const class BitMatrix *bag);

  Forest(std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<unsigned int> &_facVec);

  ~Forest();

  
  /**
     @brief Accessor for tree count.
   */
  int NTree() {
    return nTree;
  }

  
  /**
   */
  unsigned int *Origin() {
    return &treeOrigin[0];
  }
  

  void TreeBlock(class PreTree *ptBlock[], int treeBlock, int treeStart);


  inline unsigned Origin(int tIdx) {
    return treeOrigin[tIdx];
  }

  
  unsigned inline int LeafPos(int treeNum, int leafIdx) const {
    return treeOrigin[treeNum] + leafIdx;
  }

  
  inline double LeafVal(int treeNum, int leafIdx) const {
    return forestNode[LeafPos(treeNum, leafIdx)].Score();
  }


  void Reserve(unsigned int nodeEst, unsigned int facEst, double slop);


  /**
     @return current size of (possibly crescent) forest.
   */
  inline unsigned int Height() const {
    return forestNode.size();
  }


  /**
     @brief Computes tree height from either of origin vector or,
     if at the top or growing, the current height.

     @parm tIdx is the tree number.

     @return height of tree.
   */
  inline unsigned int TreeHeight(int tIdx) const {
    if (tIdx < nTree - 1  && treeOrigin[tIdx + 1] > 0)
      return treeOrigin[tIdx + 1] - treeOrigin[tIdx];
    else
      return Height() - treeOrigin[tIdx];
  }

  
  /**
     @return current size of (possibly crescent) splitting vector.
   */
  inline unsigned int SplitHeight() const {
    return facVec.size();
  }


  inline bool Nonterminal(unsigned int idx) const {
    return forestNode[idx].Nonterminal();
  }


  inline bool Nonterminal(int tIdx, unsigned int off) const {
    unsigned int idx = treeOrigin[tIdx] + off;
    return Nonterminal(idx);
  }
  

  inline unsigned int Extent(unsigned int idx) const {
    return forestNode[idx].Extent();
  }


  inline unsigned int Extent(int tIdx, unsigned int off) const {
    unsigned int idx = treeOrigin[tIdx];
    return Extent(idx);
  }


  inline void LeafAccum(int tIdx, unsigned int off) const {
    int idx = treeOrigin[tIdx] + off;
    forestNode[idx].LeafCount()++;
  }


  /**
     @brief Builds score incrementally:  regression.
   */
  inline void ScoreAccum(int tIdx, unsigned int off, double incr) const {
    int idx = treeOrigin[tIdx] + off;
    forestNode[idx].ScoreAccum(incr);
  }


  /**
     @brief Scales accumulated score:  regression.
   */
  inline void ScoreReg(int tIdx, unsigned int off, unsigned int sCount) const {
    int idx = treeOrigin[tIdx] + off;
    forestNode[idx].ScoreScale(sCount);
  }

  int *ExtentPosition(int tIdx = -1) const;

  void NodeProduce(unsigned int _predIdx, unsigned int _bump, double _split);
  void BitProduce(class BV *splitBits, unsigned int bitEnd);
  void Origins(unsigned int tIdx);

  void ScoreCtg(int tIdx, unsigned int off, unsigned int ctg, double weight) const;
  void ScoreUpdate(const class RowRank *rowRank);
};

#endif
