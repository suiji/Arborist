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
  static void TreeExport(const std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_pred, std::vector<unsigned int> &_bump, std::vector<double> &_split, unsigned int treeOff, unsigned int treeHeight);

  /**
     @brief Static determination of individual tree height.

     @return Height of tree.
   */
  static inline unsigned int TreeHeight(const std::vector<unsigned int> &_nodeOrigin, unsigned int height, unsigned int tIdx) {
    unsigned int heightInf = _nodeOrigin[tIdx];
    return tIdx < _nodeOrigin.size() - 1 ? _nodeOrigin[tIdx + 1] - heightInf : height - heightInf;
  }

 public:
  
  void SplitUpdate(const class RowRank *rowRank);
  static void Export(const std::vector<unsigned int> &_nodeOrigin, const std::vector<ForestNode> &_forestNode, std::vector<std::vector<unsigned int> > &_pred, std::vector<std::vector<unsigned int> > &_bump, std::vector<std::vector<double> > &_split);

  inline void Init() {
    pred = bump = 0;
    num = 0.0;
  }


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


  /**
     @brief Accessor for final leaf index.
   */
  inline unsigned int &LeafIdx() {
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
  class Predict *predict;
  class BVJagged *facSplit; // Consolidation of per-tree values.

  void PredictAcrossNum(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const;
  void PredictAcrossFac(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const;
  void PredictAcrossMixed(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const;
 public:

  void SplitUpdate(const class RowRank *rowRank) const;

  void PredictAcross(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const ;
  
  void PredictRowNum(unsigned int row, const double rowT[], unsigned int rowBlock, const class BitMatrix *bag) const;
  void PredictRowFac(unsigned int row, const int rowT[], unsigned int rowBlock, const class BitMatrix *bag) const;
  void PredictRowMixed(unsigned int row, const double rowNT[], const int rowIT[], unsigned int rowBlock, const class BitMatrix *bag) const;

  Forest(std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<unsigned int> &_facVec);
  Forest(std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<unsigned int> &_facVec, class Predict *_predict);
  ~Forest();

  void NodeInit(unsigned int treeHeight);
  
  /**
     @brief Accessor for tree count.
   */
  int NTree() const {
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


  inline unsigned int NodeIdx(unsigned int tIdx, unsigned int nodeOffset) {
    return treeOrigin[tIdx] + nodeOffset;
  }

  
  /**
     @brief Sets looked-up nonterminal node to values passed.

     @return void.
  */
  inline void NonterminalProduce(unsigned int tIdx, unsigned int nodeIdx, unsigned int _predIdx, unsigned int _bump, double _split) {
    forestNode[NodeIdx(tIdx, nodeIdx)].Set(_predIdx, _bump, _split);
  }


  /**
    @brief Sets looked-up leaf node to leaf index passed.

    @return void.

  */
  inline void LeafProduce(unsigned int tIdx, unsigned int nodeIdx, unsigned int _leafIdx) {
    forestNode[NodeIdx(tIdx, nodeIdx)].Set(_leafIdx, 0, 0.0);
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

     @parm tIdx is the tree number, if nonnegative, otherwise an
     indication to return the forest heigght.

     @return height of tree/forest.
   */
  inline unsigned int TreeHeight(int tIdx) const {
    if (tIdx < 0)
      return Height();

    unsigned int heightInf = treeOrigin[tIdx];
    if (tIdx < nTree - 1  && treeOrigin[tIdx + 1] > 0)
      return treeOrigin[tIdx + 1] - heightInf;
    else
      return Height() - heightInf;
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

  
  /**
     @brief

     @param tIdx is a tree index, if nonnegative, else a place holder
     indicating that the offset is absolute.

     @param off is the offset, either absolute or tree-relative.

     @return true iff referenced node is non-terminal.
   */
  inline bool Nonterminal(int tIdx, unsigned int off) const {
    unsigned int idx = tIdx >= 0 ? treeOrigin[tIdx] + off : off;
    return Nonterminal(idx);
  }
  

  inline unsigned int &LeafIdx(unsigned int idx) {
    return forestNode[idx].LeafIdx();
  }


  /**
     @brief Computes the extent of a leaf, that is, the number of
     samples it subsumes.

     @param tIdx is a tree index, if nonnegative, else a place holder
     indicating that the offset is absolute.

     @param off is the offset, either absolute or tree-relative.

     @return referenced leaf index.
   */
  inline unsigned int &LeafIdx(int tIdx, unsigned int off) {
    unsigned int idx = tIdx >= 0 ? treeOrigin[tIdx] + off : off;
    return LeafIdx(idx);
  }


  void NodeProduce(unsigned int _predIdx, unsigned int _bump, double _split);
  void BitProduce(const class BV *splitBits, unsigned int bitEnd);
  void Origins(unsigned int tIdx);
};

#endif
