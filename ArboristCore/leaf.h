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

#include "typeparam.h"
#include "sample.h"
#include <vector>


class BagLeaf {
  unsigned int leafIdx;
  unsigned int sCount; // # times bagged:  > 0

 public:
  BagLeaf() {
  }
  
  BagLeaf(unsigned int leafIdx_,
          unsigned int sCount_) : leafIdx(leafIdx_), sCount(sCount_) {
  }

  
  inline unsigned int getLeafIdx() const {
    return leafIdx;
  }

  
  inline unsigned int getSCount() const {
    return sCount;
  }
};


class LeafNode {
  double score;
  unsigned int extent; // count of sample-index slots.

  
 public:

  inline void Init() {
    score = 0.0;
    extent = 0;
  }

  
  /**
     @brief Accessor for fully-accumulated extent value.
   */
  inline unsigned int getExtent() const {
    return extent;
  }


  /**
     @brief Accessor for score value.

     @return void, with output reference parameters.
  */
  inline double Score() const {
    return score;
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


  inline double getScore() const {
    return score;
  }
};


class LeafTrain {
  static bool thinLeaves;
  vector<unsigned int> origin; // Starting position, per tree.
  vector<LeafNode> leafNode;
  vector<BagLeaf> bagLeaf; // bagged row/count:  per sample.


 protected:
  void NodeExtent(const class Sample *sample, vector<unsigned int> leafMap, unsigned int leafCount, unsigned int tIdx);

 public:
  static void Immutables(bool _thinLeaves);
  static void DeImmutables();

  LeafTrain(unsigned int _nTree);
  
  virtual ~LeafTrain();
  virtual void Reserve(unsigned int leafEst, unsigned int bagEst);
  virtual void Leaves(const class Sample *sample, const vector<unsigned int> &leafMap, unsigned int tIdx) = 0;

  size_t getNodeBytes() const;
  size_t getBLBytes() const;
  /** 
    @brief Serializes the internally-typed objects, 'LeafNode', as well
    as the unsigned integer (packed bit) vector, "bagBits".
  */
  void getNodeRaw(unsigned char *leafRaw) const;
  void getBLRaw(unsigned char *blRaw) const;
  
  void bagTree(const class Sample *sample,
               const vector<unsigned int> &leafMap);

  inline const vector<unsigned int>& getTreeOrigin() const {
    return origin;
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


  /**
    @brief Sets score.
  */
  inline void ScoreSet(unsigned int tIdx, unsigned int leafIdx, double score) {
    Score(tIdx, leafIdx) = score;
  }


};


class LeafTrainReg : public LeafTrain {
  
  void Scores(const class Sample *sample,
              const vector<unsigned int> &leafMap,
              unsigned int leafCount,
              unsigned int tIdx);


  /**
     @brief Accumulates dividend for computation of mean.  Assumes current
     tree is final reference in orgin vector.

     @param leafIdx is the tree-relative leaf index.

     @param incr is the amount by which to increment the accumulating score.

     @return void, with side-effected leaf-node score.
  */
  void ScoreAccum(unsigned int tIdx,
                  unsigned int leafIdx,
                  double incr) {
    LeafTrain::Score(tIdx, leafIdx) += incr;
  }


  /**
     @brief Scales accumulated response to obtain mean.  Assumes current
     tree is final reference in origin vector.

     @param leafIdx is the tree-relative leaf index.

     @param sCount is the total number of sampled rows subsumed by the leaf.

     @return void, with final leaf-node score.
  */
  inline void ScoreScale(unsigned int tIdx,
                         unsigned int leafIdx,
                         unsigned int sCount) {
    LeafTrain::Score(tIdx, leafIdx) /= sCount;
  }


 public:
  LeafTrainReg(unsigned int _nTree);
  ~LeafTrainReg();


  
  void Reserve(unsigned int leafEst,
               unsigned int bagEst);
  void Leaves(const class Sample *sample,
              const vector<unsigned int> &leafMap,
              unsigned int tIdx);

};


class LeafTrainCtg : public LeafTrain {
  vector<double> weight; // # leaves x # categories
  const unsigned int nCtg;
  const double weightScale;

  void Scores(const class SampleCtg *sample,
              const vector<unsigned int> &leafMap,
              unsigned int leafCount,
              unsigned int tIdx);
 public:
  LeafTrainCtg(unsigned int _nTree,
               unsigned int rowTrain,
               unsigned int _nCtg);
  ~LeafTrainCtg();


  void Reserve(unsigned int leafEst,
               unsigned int bagEst);

  void getWeight(double weightOut[]) const;
  
  /**
     @brief Looks up info by leaf index and category value.

     @param leafIdx is a tree-relative leaf index.

     @param ctg is a zero-based category value.

     @return reference to info slot.
   */
  inline double &WeightSlot(unsigned int tIdx, unsigned int leafIdx, unsigned int ctg) {
    return weight[nCtg * NodeIdx(tIdx, leafIdx) + ctg];
  }
  

  inline unsigned int CtgWidth() const {
    return nCtg;
  }


  inline void WeightAccum(unsigned int tIdx, unsigned int leafIdx, unsigned int ctg, double sum) {
    WeightSlot(tIdx, leafIdx, ctg) += sum;
  }


  inline void WeightInit(unsigned int leafCount) {
    weight.insert(weight.end(), nCtg * leafCount, 0.0);
  }


  inline double WeightScale(unsigned int tIdx, unsigned int leafIdx, unsigned int ctg, double scale) {
    WeightSlot(tIdx, leafIdx, ctg) *= scale;

    return WeightSlot(tIdx, leafIdx, ctg);
  }


  inline const vector<double> &getWeight() const {
    return weight;
  }
  
  
  void Leaves(const class Sample *sample, const vector<unsigned int> &leafMap, unsigned int tIdx);

};


/**
   @brief Represents leaves for fully-trained forest.
 */
class Leaf {
 protected:
  const unsigned int *origin;
  const class LeafNode *leafNode;
  const class BagLeaf *bagLeaf;
  const unsigned int nTree;
  const unsigned int leafCount;
  const unsigned int bagLeafTot;

  void populate(const class BitMatrix *baggedRows,
              vector< vector<unsigned int> > &rowTree,
              vector< vector<unsigned int> > &sCountTree) const;

  void nodeExport(vector<vector<double> > &_score,
                  vector<vector<unsigned int> > &_extent) const;

  
 public:
  Leaf(const unsigned int *_origin,
       unsigned int _nTree,
       const class LeafNode *_leafNode,
       unsigned int _leafCount,
       const class BagLeaf*_bagLeaf,
       unsigned int _bagTot);

  virtual ~Leaf();
  virtual const unsigned int rowPredict() const = 0;
  virtual void ScoreBlock(const class Predict *predict,
                     unsigned int rowStart,
                     unsigned int rowEnd) = 0;

  inline unsigned int getNTree() const {
    return nTree;
  }

  
  inline unsigned int NodeIdx(unsigned int tIdx, unsigned int leafIdx) const {
    return origin[tIdx] + leafIdx;
  }
  
  inline unsigned int getSCount(unsigned int sIdx) const {
    return bagLeaf[sIdx].getSCount();
  }


  inline unsigned int getExtent(unsigned int nodeIdx) const {
    return leafNode[nodeIdx].getExtent();
  }


  inline double getScore(int tIdx, unsigned int leafIdx) const {
    return leafNode[NodeIdx(tIdx, leafIdx)].getScore();
  }


  /**
     @brief Determines individual tree height.

     @return Height of tree.
   */
  inline unsigned int getHeight(unsigned int tIdx)  const {
    unsigned int heightInf = origin[tIdx];
    return tIdx < nTree - 1 ? origin[tIdx + 1] - heightInf : leafCount - heightInf;
  }

  
  /**
     @brief Computes sum of all bag sizes.

     @return size of information vector, which represents all bagged samples.
   */
  inline unsigned int BagLeafTot() const {
    return bagLeafTot;
  }


  /**
     @brief Determines inattainable leaf index value from leafNode
     vector.  N.B.:  nonsensical if called before training complete.

     @return inattainable leaf index valu.
   */
  inline unsigned int NoLeaf() const {
    return leafCount;
  }


  /**
     @brief computes total number of leaves in forest.

     @return size of leafNode vector.
   */
  inline unsigned int getLeafCount() const {
    return leafCount;
  }

};


class LeafReg : public Leaf {
  const double *yTrain;
  const double meanTrain; // Mean of training response.
  vector<unsigned int> offset; // Accumulated extents.
  double defaultScore;
  vector<double> yPred;
  void Offsets();
  
 public:
  LeafReg(const unsigned int _origin[],
          unsigned int _nTree,
          const class LeafNode _leafNode[],
          unsigned int _leafCount,
          const class BagLeaf _bagLeaf[],
          unsigned int _bagLeafTot,
          const double *_yTrain,
          double _meanTrain,
          unsigned int _rowPredict);

  ~LeafReg() {}

  inline const double *YTrain() const {
    return yTrain;
  }


  const vector<double> &YPred() const {
    return yPred;
  }
  

  const unsigned int rowPredict() const {
    return yPred.size();
  }

  
  inline double MeanTrain() const {
    return meanTrain;
  }


  void ScoreBlock(const class Predict *predict,
                  unsigned int rowStart,
                  unsigned int rowEnd);
  
  /**
   @brief Computes bag index bounds in forest setting.  Only client is Quant.
  */
  void BagBounds(unsigned int tIdx,
                 unsigned int leafIdx,
                 unsigned int &start,
                 unsigned int &end) const {
    auto forestIdx = NodeIdx(tIdx, leafIdx);
    start = offset[forestIdx];
    end = start + getExtent(forestIdx);
  }


  /**
     @param tIdx is the tree index.

     @param bagIdx is the absolute index of a bagged row.

     @return absolute index of leaf containing the bagged row.
   */
  unsigned int getLeafIdx(unsigned int tIdx,
                          unsigned int bagIdx,
                          unsigned int &offset_) const {
    auto leafIdx = origin[tIdx] + bagLeaf[bagIdx].getLeafIdx();
    offset_ = offset[leafIdx];
    return leafIdx;
  }

  
  void populate(const class BitMatrix *baggedRows,
              vector<vector<unsigned int> >&rowTree,
              vector<vector<unsigned int> > &sCountTree,
              vector<vector<double> > &scoreTree,
              vector<vector<unsigned int> >&extentTree) const;
};


class LeafCtg : public Leaf {
  const double *weight;
  const unsigned int ctgTrain;
  vector<unsigned int> yPred;
  unsigned int defaultScore;
  vector<double> defaultWeight;
  unsigned int DefaultScore();
  void DefaultInit();
  double DefaultWeight(double *weightPredict);

 public:
  // Sized to zero by constructor.
  // Resized by bridge and filled in by prediction.
  vector<double> votes;
  vector<unsigned int> census;
  vector<double> prob;

  
  LeafCtg(const unsigned int _origin[],
              unsigned int _nTree,
              const class LeafNode _leafNode[],
              unsigned int _leafCount,
              const class BagLeaf _bagLeaf[],
              unsigned int _bagLeafTot,
              const double _weight[],
          unsigned int _ctgTrain,
          unsigned int _rowPredict,
          bool doProb);

  ~LeafCtg(){}


  const vector<unsigned int> &YPred() {
    return yPred;
  }

  const unsigned int rowPredict() const {
    return yPred.size();
  }


  const unsigned int *Census() const {
    return &census[0];
  }

  const vector<double> &Prob() const {
    return prob;
  }
  

  void ScoreBlock(const class Predict *predict,
                  unsigned int rowStart,
                  unsigned int rowEnd);

  
  void Vote();

  
  void ProbBlock(const class Predict *predict,
                 unsigned int rowStart,
                 unsigned int rowEnd);


  void DefaultWeight(vector<double> &defaultWeight) const;


  inline unsigned int CtgTrain() const {
    return ctgTrain;
  }
  

  inline double WeightCtg(int tIdx,
                          unsigned int leafIdx,
                          unsigned int ctg) const {
    return weight[ctgTrain * NodeIdx(tIdx, leafIdx) + ctg];
  }


  /**
     @brief Derives an index into a matrix having stride equal to the
     number of training categories.

     @return derived strided index.
   */
  unsigned int getTrainIdx(unsigned int row, unsigned int col) const {
    return row * ctgTrain + col;
  }


  void populate(const BitMatrix *baggedRows,
              vector<vector<unsigned int> > &rowTree,
              vector<vector<unsigned int> > &sCountTree,
              vector<vector<double> > &scoreTree,
              vector<vector<unsigned int> > &extentTree,
              vector<vector<double> > &_weightTree) const;
};

#endif
