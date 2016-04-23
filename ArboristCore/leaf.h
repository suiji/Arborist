// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leaf.h

   @brief Class definitions for sample-oriented aspects of training.

   @author Mark Seligman
 */

#ifndef ARBORIST_LEAF_H
#define ARBORIST_LEAF_H

#include "sample.h"
#include <vector>


/**
   @brief Transmits bagged row information for post-training use.
 */
class BagRow {
  unsigned int row;
  unsigned int sCount;
 public:
  void Init() {
    row = sCount = 0;
  }
  

  /**
     @brief Sets both members.
   */
  void Set(unsigned int _row, unsigned int _sCount) {
    row = _row;
    sCount = _sCount;
  }


  /**
     @brief Accessor for row.
   */
  inline unsigned int Row() const {
    return row;
  }


  /**
     @brief Accessor for sample count.
   */
  inline unsigned int SCount() const {
    return sCount;
  }

  
  /**
     @brief Accessor for both members.
   */
  inline void Ref(unsigned int &_row, unsigned int &_sCount) const {
    _row = row;
    _sCount = sCount;
  }

};


class LeafNode {
  double score;
  unsigned int extent; // count of sample-index slots.
  static void TreeExport(const std::vector<LeafNode> &_leafNode, unsigned int treeOff, unsigned int leafCount, std::vector<double> &_score, std::vector<unsigned int> &_extent);


  /**
     @brief All-field accessor.

     @return void, with output reference parameters.
   */
  inline void Ref(double &_score, unsigned int _extent) const {
    _score = score;
    _extent = extent;
  }

 public:
  static void Export(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, std::vector<std::vector<double> > &_score, std::vector<std::vector<unsigned int> > &_extent);

  /**
     @brief Static determination of individual tree height.

     @return Height of tree.
   */
  static inline unsigned int LeafCount(const std::vector<unsigned int> &_origin, unsigned int height, unsigned int tIdx) {
    unsigned int heightInf = _origin[tIdx];
    return tIdx < _origin.size() - 1 ? _origin[tIdx + 1] - heightInf : height - heightInf;
  }


  inline void Init() {
    score = 0.0;
    extent = 0;
  }

  
  /**
     @brief Accessor for fully-accumulated extent value.
   */
  inline unsigned int Extent() const {
    return extent;
  }


  /**
     @brief Reference accessor for accumulating extent.
   */
  inline unsigned int &Count() {
    return extent;
  }


  inline double &Score() {
    return score;
  }

  inline double GetScore() const {
    return score;
  }

};


class Leaf {
  std::vector<unsigned int> &origin; // Starting position, per tree.
  const unsigned int nTree;
  std::vector<LeafNode> &leafNode;
  std::vector<BagRow> &bagRow; // bagged row/count:  per sample.
  static void TreeExport(const std::vector<BagRow> &_bagRow, unsigned int bagOrig, unsigned int bagCount, std::vector<unsigned int> &rowTree, std::vector<unsigned int> &sCountTree);

 protected:
  void RowBag(const class Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx);
  static unsigned int BagCount(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, unsigned int tIdx);
  static void Export(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, const std::vector<BagRow> &_bagRow, std::vector< std::vector<unsigned int> > &rowTree, std::vector< std::vector<unsigned int> >&sCountTree);
  void NodeExtent(const class Sample *sample, std::vector<unsigned int> leafMap, unsigned int leafCount, unsigned int tIdx);

 public:
  Leaf(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow);
  virtual ~Leaf() {}
  
  virtual void Reserve(unsigned int leafEst, unsigned int bagEst);
  virtual void Leaves(const class Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx) = 0;
  virtual void RankInit(unsigned int bagCount, unsigned int init) = 0;
  virtual void RankSet(unsigned int sOff, const class Sample *sample, unsigned int sIdx) = 0;

  class BitMatrix *ForestBag(unsigned int rowTrain);
  
  void SampleOffset(std::vector<unsigned int> &sampleOffset, unsigned int leafBase, unsigned int leafCount, unsigned int sampleBase) const;

  inline unsigned Origin(unsigned int tIdx) {
    return origin[tIdx];
  }

  inline unsigned int NTree() const {
    return nTree;
  }
  

  /**
     @brief computes total number of leaves in forest.

     @return size of leafNode vector.
   */
  inline unsigned int NodeCount() const {
    return leafNode.size();
  }

  
  inline double &Score(unsigned int idx) {
    return leafNode[idx].Score();
  }

  
  inline unsigned int NodeIdx(unsigned int tIdx, unsigned int leafIdx) const {
    return origin[tIdx] + leafIdx;
  }

  
  inline double &Score(int tIdx, unsigned int leafIdx) {
    return Score(NodeIdx(tIdx, leafIdx));
  }

  inline double GetScore(int tIdx, unsigned int leafIdx) const {
    return leafNode[NodeIdx(tIdx, leafIdx)].GetScore();
  }


  /**
    @brief Sets score.
  */
  inline void ScoreSet(unsigned int tIdx, unsigned int leafIdx, double score) {
    Score(tIdx, leafIdx) = score;
  }


  inline unsigned int Extent(unsigned int idx) const {
    return leafNode[idx].Extent();
  }

  
  inline unsigned int Extent(unsigned int tIdx, int leafIdx) const {
    int idx = NodeIdx(tIdx, leafIdx);
    return Extent(idx);
  }

  inline unsigned int SCount(unsigned int idx) const {
    return bagRow[idx].SCount();
  }
};


class LeafReg : public Leaf {
  std::vector<unsigned int> &rank;
  static void TreeExport(const std::vector<unsigned int> &_rank, unsigned int bagOrig, unsigned int bagCount, std::vector<unsigned int> &rankTree);

  void Scores(const class Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx);


  /**
     @brief Accumulates dividend for computation of mean.  Assumes current
     tree is final reference in orgin vector.

     @param leafIdx is the tree-relative leaf index.

     @param incr is the amount by which to increment the accumulating score.

     @return void, with side-effected leaf-node score.
  */
  void ScoreAccum(unsigned int tIdx, unsigned int leafIdx, double incr) {
    Leaf::Score(tIdx, leafIdx) += incr;
  }


  /**
     @brief Scales accumulated response to obtain mean.  Assumes current
     tree is final reference in origin vector.

     @param leafIdx is the tree-relative leaf index.

     @param sCount is the total number of sampled rows subsumed by the leaf.

     @return void, with final leaf-node score.
  */
  inline void ScoreScale(unsigned int tIdx, unsigned int leafIdx, unsigned int sCount) {
    Leaf::Score(tIdx, leafIdx) /= sCount;
  }


 public:
  LeafReg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, std::vector<unsigned int> &_rank);
  ~LeafReg();
  static void Export(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, const std::vector<BagRow> &_bagRow, const std::vector<unsigned int> &_rank, std::vector<std::vector<unsigned int> >&rowTree, std::vector<std::vector<unsigned int> > &sCountTree, std::vector<std::vector<double> > &scoreTree, std::vector<std::vector<unsigned int> >&extentTree, std::vector< std::vector<unsigned int> > &rankTree);
  
  void Reserve(unsigned int leafEst, unsigned int bagEst);
  void Leaves(const class Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx);
  void RankInit(unsigned int bagCount, unsigned int init);
  void RankSet(unsigned int sOff, const class Sample *sample, unsigned int sIdx);

  
  /**
     @brief Accessor for rank.
   */
  inline unsigned int Rank(unsigned int idx) const {
    return rank[idx];
  }

  
  /**
     @brief Computes sum of all bag sizes.

     @return size of information vector, which represents all bagged samples.
   */
  inline unsigned int BagTot() const {
    return rank.size();
  }


};


class LeafCtg : public Leaf {
  std::vector<double> &weight;
  unsigned int ctgWidth;

  static void TreeExport(const std::vector<double> &leafWeight, unsigned int _ctgWidth, unsigned int treeOffset, unsigned int leafCount, std::vector<double> &_weight);
  static unsigned int LeafCount(std::vector<unsigned int> _origin, unsigned int weightLen, unsigned int _ctgWidth, unsigned int tIdx);

  /**
     @brief Looks up info by leaf index and category value.

     @param leafIdx is a tree-relative leaf index.

     @param ctg is a zero-based category value.

     @return reference to info slot.
   */
  double &WeightSlot(unsigned int tIdx, unsigned int leafIdx, unsigned int ctg) {
    unsigned int idx = NodeIdx(tIdx, leafIdx);
    return weight[ctgWidth * idx + ctg];
  }
  
  void Scores(const class SampleCtg *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx);
 public:
  LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, std::vector<double> &_weight, unsigned int _ctgWdith);
  LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, std::vector<double> &_weight);
  ~LeafCtg();

  static void Export(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, const std::vector<BagRow> &_bagRow, const std::vector<double> &_weight, unsigned int _ctgWidth, std::vector<std::vector<unsigned int> > &rowTree, std::vector<std::vector<unsigned int> > &sCountTree, std::vector<std::vector<double> > &scoreTree, std::vector<std::vector<unsigned int> > &extentTree, std::vector<std::vector<double> > &_weightTree);

  void Reserve(unsigned int leafEst, unsigned int bagEst);
  
  void RankInit(unsigned int bagCount, unsigned int init) {}
  void RankSet(unsigned int sOff, const class Sample *sample, unsigned int sIdx) {}


  inline unsigned int CtgWidth() const {
    return ctgWidth;
  }


  inline void WeightAccum(unsigned int tIdx, unsigned int leafIdx, unsigned int ctg, double sum) {
    WeightSlot(tIdx, leafIdx, ctg) += sum;
  }


  inline void WeightInit(unsigned int leafCount) {
    weight.insert(weight.end(), ctgWidth * leafCount, 0.0);
  }


  inline double WeightScale(unsigned int tIdx, unsigned int leafIdx, unsigned int ctg, double scale) {
    WeightSlot(tIdx, leafIdx, ctg) *= scale;

    return WeightSlot(tIdx, leafIdx, ctg);
  }


  inline double WeightCtg(int tIdx, unsigned int leafIdx, unsigned int ctg) const {
    unsigned int idx = NodeIdx(tIdx, leafIdx);
    return weight[ctgWidth * idx + ctg];
  }

  void Leaves(const class Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx);
};


#endif
