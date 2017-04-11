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
#include <algorithm>

#include "param.h"


/**
   @brief To replace parallel array access.
 */
class ForestNode {
  static const double *splitQuant;
  unsigned int pred;
  unsigned int bump;
  union {
    double num;
    RankRange rankRange;
  } splitVal;
  
  static void TreeExport(const ForestNode *_forestNode, std::vector<unsigned int> &_pred, std::vector<unsigned int> &_bump, std::vector<double> &_split, unsigned int treeOff, unsigned int treeHeight);

  /**
     @brief Static determination of individual tree height.

     @return Height of tree.
   */
  static inline unsigned int TreeHeight(const unsigned int _nodeOrigin[], unsigned int _nTree, unsigned int height, unsigned int tIdx) {
    unsigned int heightInf = _nodeOrigin[tIdx];
    return tIdx < _nTree - 1 ? _nodeOrigin[tIdx + 1] - heightInf : height - heightInf;
  }

  
 public:
  static void Immutables(const double _splitQuant[]) {
    splitQuant = _splitQuant;
  }

  
  static void DeImmutables() {
    splitQuant = 0;
  }
  
  
  void SplitUpdate(const class PMTrain *pmTrain, const class RowRank *rowRank);
  static void Export(const unsigned int _nodeOrigin[], unsigned int _nTree, const ForestNode *_forestNode, unsigned int nodeEnd, std::vector<std::vector<unsigned int> > &_pred, std::vector<std::vector<unsigned int> > &_bump, std::vector<std::vector<double> > &_split);

  inline void Init() {
    pred = bump = 0;
    splitVal.num = 0.0;
  }


  inline void SetRank(unsigned int _pred, unsigned int _bump, unsigned int _rankLow, unsigned int _rankHigh) {
    pred = _pred;
    bump = _bump;
    splitVal.rankRange.rankLow = _rankLow;
    splitVal.rankRange.rankHigh = _rankHigh;
  }


  inline void SetNum(unsigned int _pred, unsigned int _bump, double _num) {
    pred = _pred;
    bump = _bump;
    splitVal.num = _num;
  }


  inline unsigned int &Pred() {
    return pred;
  }


  inline double &Num() {
    return splitVal.num;
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
    _num = splitVal.num;
  }
};


/**
   @brief The decision forest as a read-only collection.
*/
class Forest {
  const ForestNode *forestNode;
  const unsigned int *treeOrigin;
  const unsigned int nTree;
  class BVJagged *facSplit; // Consolidation of per-tree values.

  class Predict *predict;
  const class PMPredict *predMap;

  void PredictAcrossNum(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const;
  void PredictAcrossFac(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const;
  void PredictAcrossMixed(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const;


  inline unsigned int NTree() const {
    return nTree;
  }


  inline void Ref(unsigned int idx, unsigned int &pred, unsigned int &bump, double &num) const {
    forestNode[idx].Ref(pred, bump, num);
  }
  

 public:
  void PredictAcross(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const ;
   void PredictRowNum(unsigned int row, const double rowT[], unsigned int rowBlock, const class BitMatrix *bag) const;
  void PredictRowFac(unsigned int row, const unsigned int rowT[], unsigned int rowBlock, const class BitMatrix *bag) const;
  void PredictRowMixed(unsigned int row, const double rowNT[], const unsigned int rowIT[], unsigned int rowBlock, const class BitMatrix *bag) const;

  Forest(const ForestNode _forestNode[], const unsigned int _origin[], unsigned int _nTree, unsigned int _facVec[], size_t _facLen, const unsigned int _facOrigin[], unsigned int _nFac, class Predict *_predict);
  ~Forest();
};


class ForestTrain {
  std::vector<ForestNode> &forestNode;
  std::vector<unsigned int> &treeOrigin;
  std::vector<unsigned int> &facOrigin;
  std::vector<unsigned int> &facVec;


  inline unsigned int NodeIdx(unsigned int tIdx, unsigned int nodeOffset) {
    return treeOrigin[tIdx] + nodeOffset;
  }

  
  /**
     @return current size of forest.
   */
  inline unsigned int Height() const {
    return forestNode.size();
  }


  /**
     @return current size of splitting vector.
   */
  inline unsigned int SplitHeight() const {
    return facVec.size();
  }

  
 public:
  ForestTrain(std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<unsigned int> &_facVec);
  ~ForestTrain();
  void BitProduce(const class BV *splitBits, unsigned int bitEnd);
  void Origins(unsigned int tIdx);
  void Reserve(unsigned int nodeEst, unsigned int facEst, double slop);
  void NodeInit(unsigned int treeHeight);
  void SplitUpdate(const class PMTrain *pmTrain, const class RowRank *rowRank) const;


  /**
     @brief Sets looked-up nonterminal node to values passed.

     @return void.
  */
  inline void RankProduce(unsigned int tIdx, unsigned int nodeIdx, unsigned int _predIdx, unsigned int _bump, unsigned int _rankLow, unsigned int _rankHigh) {
    forestNode[NodeIdx(tIdx, nodeIdx)].SetRank(_predIdx, _bump, _rankLow, _rankHigh);
  }


  /**
     @brief Sets looked-up nonterminal node to values passed.

     @return void.
  */
  inline void OffsetProduce(unsigned int tIdx, unsigned int nodeIdx, unsigned int _predIdx, unsigned int _bump, unsigned int offset) {
    forestNode[NodeIdx(tIdx, nodeIdx)].SetNum(_predIdx, _bump, offset);
  }


  /**
    @brief Sets looked-up leaf node to leaf index passed.

    @return void.

  */
  inline void LeafProduce(unsigned int tIdx, unsigned int nodeIdx, unsigned int _leafIdx) {
    forestNode[NodeIdx(tIdx, nodeIdx)].SetNum(_leafIdx, 0, 0.0);
  }
};

#endif
