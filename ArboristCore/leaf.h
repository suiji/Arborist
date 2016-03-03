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
  unsigned int count; // count of sample-index slots.
 public:

  inline void Init() {
    score = 0.0;
    count = 0;
  }

  inline unsigned int Extent() const {
    return count;
  }


  inline unsigned int &Count() {
    return count;
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
 public:
  Leaf(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode);
  void Reserve(unsigned int leafEst);
  void NodeExtent(std::vector<unsigned int> frontierMap, unsigned int bagCount, unsigned int leafCount, unsigned int tIdx);

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
  void Scores(const class SampleReg *sampleReg, const std::vector<unsigned int> &frontierMap, unsigned int leafCount, unsigned int tIdx);

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
  void Reserve(unsigned int leafEst, unsigned int bagEst);
  void Leaves(const class SampleReg *sampleReg, const std::vector<unsigned int> &frontierMap, unsigned int tIdx);
  void SampleOffset(std::vector<unsigned int> &sampleOffset, unsigned int leafBase, unsigned int leafCount, unsigned int sampleBase) const;
  void SampleInfo(const class SampleReg *sampleReg, const std::vector<unsigned int> &frontierMap, unsigned int leafCount, unsigned int tIdx);


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
  
  void Scores(const class SampleCtg *sampleCtg, const std::vector<unsigned int> &frontierMap, unsigned int leafCount, unsigned int tIdx);
 public:
  LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<double> &_info, unsigned int _ctgWdith);
  LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<double> &_info);
  void Reserve(unsigned int leafEst);
  
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

  void Leaves(const class SampleCtg *sampleCtg, const std::vector<unsigned int> &frontierMap, unsigned int tIdx);
};


#endif
