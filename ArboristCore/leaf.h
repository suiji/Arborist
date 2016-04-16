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



class RankCount {
  unsigned int rank;
  unsigned int sCount;
 public:

  inline void Init() {
    rank = sCount = 0;
  }


  inline void Set(unsigned int _sCount, unsigned int _rank) {
    rank = _rank;
    sCount = _sCount;
  }


  inline unsigned int SCount() const {
    return sCount;
  }


  inline unsigned int Rank() const {
    return rank;
  }


  inline void Ref(unsigned int &_sCount, unsigned int &_rank) const {
    _rank = rank;
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
  static void Export(const std::vector<unsigned int> &_leafOrigin, const std::vector<LeafNode> &_leafNode, std::vector<std::vector<double> > &_score, std::vector<std::vector<unsigned int> > &_extent);

  /**
     @brief Static determination of individual tree height.

     @return Height of tree.
   */
  static inline unsigned int LeafCount(const std::vector<unsigned int> &_leafOrigin, unsigned int height, unsigned int tIdx) {
    unsigned int heightInf = _leafOrigin[tIdx];
    return tIdx < _leafOrigin.size() - 1 ? _leafOrigin[tIdx + 1] - heightInf : height - heightInf;
  }


  inline void Init() {
    score = 0.0;
    extent = 0;
  }

  inline unsigned int Extent() const {
    return extent;
  }


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
  std::vector<unsigned int> sCount; // bagged count:  per sample.
  std::vector<unsigned int> row; // originating row:  per sample.
 public:
  Leaf(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode);
  void Reserve(unsigned int leafEst);
  virtual ~Leaf() {}
  
  virtual void Reserve(unsigned int leafEst, unsigned int bagEst) = 0;
  virtual void Leaves(const class Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx) = 0;

  
  void NodeExtent(std::vector<unsigned int> leafMap, unsigned int bagCount, unsigned int leafCount, unsigned int tIdx);

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
};


class LeafReg : public Leaf {
  std::vector<class RankCount> &info;
  static void TreeExport(const std::vector<class RankCount> &_leafInfo, unsigned int treeOff, unsigned int leafCount, std::vector<unsigned int> &_rank, std::vector<unsigned int> &_sCount);
  static unsigned int LeafCount(const std::vector<unsigned int> &_origin, unsigned int height, unsigned int tIdx);

  void Scores(const class Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx);

  inline void InfoSet(unsigned int samplePos, unsigned int _sCount, unsigned int _rank) {
    info[samplePos].Set(_sCount, _rank);
  }

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
  LeafReg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<RankCount> &_info);
  ~LeafReg();
  static void Export(const std::vector<unsigned int> &_leafOrigin, const std::vector<RankCount> &_leafInfo, std::vector< std::vector<unsigned int> > &_rank, std::vector< std::vector<unsigned int> > &_sCount);
  
  void Reserve(unsigned int leafEst, unsigned int bagEst);
  void Leaves(const class Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx);
  void SampleOffset(std::vector<unsigned int> &sampleOffset, unsigned int leafBase, unsigned int leafCount, unsigned int sampleBase) const;
  void SampleInfo(const class SampleReg *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx);


  inline void InfoRef(unsigned int infoIdx, unsigned int &_sCount, unsigned int &_rank) const {
    info[infoIdx].Ref(_sCount, _rank);
  }


  inline unsigned int Rank(unsigned int infoIdx) const {
    return info[infoIdx].Rank();
  }

  inline unsigned int SCount(unsigned int infoIdx) const {
    return info[infoIdx].SCount();
  }
  
  /**
     @brief Computes sum of all bag sizes.

     @return size of information vector, which represents all bagged samples.
   */
  inline unsigned int BagTot() const {
    return info.size();
  }


};


class LeafCtg : public Leaf {
  std::vector<double> &info;
  unsigned int ctgWidth;

  static void TreeExport(const std::vector<double> &leafinfo, unsigned int _ctgWidth, unsigned int treeOffset, unsigned int leafCount, std::vector<double> &_weight);
  static unsigned int LeafCount(std::vector<unsigned int> _origin, unsigned int infoLen, unsigned int _ctgWidth, unsigned int tIdx);

  /**
     @brief Looks up info by leaf index and category value.

     @param leafIdx is a tree-relative leaf index.

     @param ctg is a zero-based category value.

     @return reference to info slot.
   */
  double &InfoSlot(unsigned int tIdx, unsigned int leafIdx, unsigned int ctg) {
    unsigned int idx = NodeIdx(tIdx, leafIdx);
    return info[ctgWidth * idx + ctg];
  }
  
  void Scores(const class SampleCtg *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx);
 public:
  LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<double> &_info, unsigned int _ctgWdith);
  LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<double> &_info);
  ~LeafCtg();

  static void Export(const std::vector<unsigned int> &_leafOrigin, const std::vector<double> &_leafInfo, unsigned int _ctgWidth, std::vector< std::vector<double> > &_weight);

  void Reserve(unsigned int leafEst, unsigned int bagEst);
  
  inline unsigned int CtgWidth() const {
    return ctgWidth;
  }


  inline void WeightAccum(unsigned int tIdx, unsigned int leafIdx, unsigned int ctg, double sum) {
    InfoSlot(tIdx, leafIdx, ctg) += sum;
  }


  inline void WeightInit(unsigned int leafCount) {
    info.insert(info.end(), ctgWidth * leafCount, 0.0);
  }


  inline double WeightScale(unsigned int tIdx, unsigned int leafIdx, unsigned int ctg, double scale) {
    InfoSlot(tIdx, leafIdx, ctg) *= scale;

    return InfoSlot(tIdx, leafIdx, ctg);
  }


  inline double WeightCtg(int tIdx, unsigned int leafIdx, unsigned int ctg) const {
    unsigned int idx = NodeIdx(tIdx, leafIdx);
    return info[ctgWidth * idx + ctg];
  }

  void Leaves(const class Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx);
};


#endif
