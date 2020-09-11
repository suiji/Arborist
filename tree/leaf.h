// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leaf.h

   @brief Class definitions for non-terminal representations.

   @author Mark Seligman
 */

#ifndef TREE_LEAF_H
#define TREE_LEAF_H


#include "typeparam.h"

#include <vector>


/**
   @brief The essential contents of a leaf.
 */
class Leaf {
  double score;
  IndexT extent; // # distinct observations mapped to this leaf.
  
 public:

  Leaf() : score(0.0), extent(0) {
  }


  /**
     @brief Getter for fully-accumulated extent value.
   */
  inline auto getExtent() const {
    return extent;
  }


  /**
     @brief Increments extent field.
   */
  inline void incrExtent() {
    extent++;
  }
  

  /**
     @brief Getter for score.
   */
  inline auto getScore() const {
    return score;
  }


  /**
     @brief Setter for score.
   */
  void setScore(double score) {
    this->score = score;
  }

  
  /**
     @brief Increments score.

     @param sum is the quantity by which to increment.
   */
  inline void scoreAccum(double sum) {
    score += sum;
  }


  /**
     @brief Scales score.

     @param scale is the quantity by which to scale.
   */
  inline void scoreScale(double scale) {
    score *= scale;
  }
};


/**
   @brief Leaf block for the crescent frame.
 */
class LBCresc {
  vector<Leaf> leaf;
  vector<size_t> height;
  IndexT leafCount; // Count of leaves in current tree.
  size_t treeFloor; // Block-relative index of current tree floor.

public:
  LBCresc(unsigned int nTree);

  /**
     @brief Leaf count getter.
   */
  auto getLeafCount() const {
    return leafCount;
  }


  const vector<size_t>& getHeight() const {
    return height;
  }

  
  /**
     @brief Allocates and initializes leaves for current tree.

     @param leafMap maps sample indices to containing leaf indices.

     @param tIdx is the block-relative tree index.
   */
  void treeInit(const vector<IndexT> &leafMap,
                unsigned int tIdx);
  

  /**
     @brief Writes the current tree origin and sets per-leaf extents.

     @param leafMap maps sample indices to tree indices.
  */
  void setExtents(const vector<IndexT> &leafMap);


  /**
     @brief Sets regression-mode scores for all leaves in tree.

     @param sample summarizes the sample tree response.

     @param leafMap maps sample indices to leaf indices.
   */
  void setScoresReg(const class Sample* sample,
                    const vector<IndexT>& leafMap);


  void setScoresCtg(const class ProbCresc* probCresc);

  /**
     @brief Accumulates a score for a leaf in the current tree.

     @param leafIdx is a tree-relative leaf index.

     @param sum is the value to add to the leaf's score.
   */
  inline void scoreAccum(IndexT leafIdx,
			 double sum) {
    leaf[treeFloor + leafIdx].scoreAccum(sum);
  }


  /**
     @brief Scales the score of a leaf in the current tree.

     @param leafIdx is a tree-relative leaf index.

     @param recipSum is the value by which to scale the score.
   */
  inline void scoreScale(IndexT leafIdx,
                         double recipSum) {
    leaf[treeFloor + leafIdx].scoreScale(recipSum);
  }


  /**
     @brief Score setter for a leaf in the current tree.

     @param leafIdx is a tree-relative leaf index.

     @param score is the value to set.
   */
  inline void setScore(IndexT leafIdx, double score) {
    leaf[treeFloor + leafIdx].setScore(score);
  }
  
  /**
     @brief Score getter for a leaf in the current tree.

     @param leafIdx is a tree-relative leaf index.
   */
  inline double getScore(IndexT leafIdx) const {
    return leaf[treeFloor + leafIdx].getScore();
  }

  
  /** 
    @brief Serializes the internally-typed objects.

    @param[out] leafRaw outputs the raw content bytes.
  */
  void dumpRaw(unsigned char *leafRaw) const;
};


// EXIT:  sCount should bw subsumable within actual Bag structure.
class BagSample {
  IndexT leafIdx; // Leaf index within tree.
  IndexT sCount; // # times bagged:  > 0

 public:
  BagSample() {
  }
  
  BagSample(IndexT leafIdx_,
            IndexT sCount_) : leafIdx(leafIdx_), sCount(sCount_) {
  }


  inline auto getLeafIdx() const {
    return leafIdx;
  }

  
  inline auto getSCount() const {
    return sCount;
  }
};


/**
   @brief BagSample block for crescent frame.
 */
class BBCresc {
  vector<BagSample> bagSample;
  vector<size_t> height;
  
public:
  BBCresc(unsigned int nTree);

  void treeInit(const Sample* sample,
                unsigned int tIdx);

  const vector<size_t>& getHeight() const {
    return height;
  }

  
  /**
     @brief Records multiplicity and leaf index for bagged samples
     within a tree.  Accessed by bag vector, so sample indices must
     reference consecutive bagged rows.
     @param leafMap maps sample indices to leaves.
     @return void.
  */
  void bagLeaves(const class Sample *sample,
                 const vector<IndexT> &leafMap);


  void dumpRaw(unsigned char blRaw[]) const; 
};


/**
   @brief Crescent LeafPredict implementation for training.
 */
class LFTrain {
protected:

  const double* y;
  unique_ptr<LBCresc> lbCresc; // Leaf block
  unique_ptr<BBCresc> bbCresc; // BagSample block

public:

  LFTrain(const double* y_,
          unsigned int treeChunk);


  virtual ~LFTrain() {}

  /**
     @base Copies front-end vectors and lights off initializations specific to classification.

     @param feCtg is the front end's response vector.

     @param feProxy is the front end's vector of proxy values.

     @return void.
  */
  static unique_ptr<class LFTrainCtg> factoryCtg(const unsigned int* feResponse,
                                                 const double* feProxy,
                                                 unsigned int treeChunk,
                                                 unsigned int nRow,
                                                 unsigned int nCtg,
                                                 unsigned int nTree);

  static unique_ptr<class LFTrainReg> factoryReg(const double* feResponse,
                                                 unsigned int treeChunk);

  
  /**
     @brief Samples (bags) the response to construct the tree root.

     @param frame summarizes the predictor orderings.

     @param treeBag encodes the bagged rows.

     @param tIdx is the block-relative tree index.
   */
  virtual unique_ptr<class Sample> rootSample(const class TrainFrame* frame,
                                              class BitMatrix* bag,
                                              unsigned int tIdx) const = 0;


  virtual void setScores(const class Sample* sample,
                         const vector<IndexT>& leafMap) = 0;

  
  /**
     @brief Allocates and initializes records for each leaf in tree.

     @param tIdx is the block-relative tree index.
   */
  virtual void treeInit(const class Sample* sample,
                        const vector<IndexT>& leafMap,
                        unsigned int tIdx);

  virtual void dumpWeight(double weightOut[]) const = 0;

  virtual size_t getWeightSize() const = 0;
  
  /**
     @brief Appends this tree's leaves to the current block.

     @param sample summarizes the sampling environment of the current tree.

     @param leafMap maps sample indices to the index of the containing leaf.

     @param tIdx is the block-relative tree index.
   */
  void blockLeaves(const class Sample *sample,
                   const vector<IndexT> &leafMap,
                   unsigned int tIdx);


  /** 
    @brief Serializes the internally-typed objects, 'Leaf', as well
    as the unsigned integer (packed bit) vector, "bagBits".
  */
  void cacheLeafRaw(unsigned char *leafRaw) const;
  void cacheBLRaw(unsigned char *blRaw) const;
  

  const vector<size_t>& getLeafHeight() const {
    return lbCresc->getHeight();
  }


  const vector<size_t>& getBagHeight() const {
    return bbCresc->getHeight();
  }
};


class LFTrainReg : public LFTrain {

  /**
     @brief Sets scores for current tree's leaves.

     @param sample summarizes the sample response.

     @param leafMap maps sample indices to leaf indices.
   */
  void setScores(const class Sample* sample,
                 const vector<IndexT>& leafMap);

public:
  /**
     @brief Regression constructor.

     @param y is the training response.

     @param treeChunk is the number of trees in the current block.
   */
  LFTrainReg(const double* y,
             unsigned int treeChunk);

  ~LFTrainReg(){}

  /**
     @brief Samples response of current tree.

     @param frame summarizes the presorted observations.

     @param bag summarizes the in-bag rows for a collection of trees.

     @param tIdx is the block-relative index of the current tree.

     @return summary of sampled response.
   */
  unique_ptr<class Sample> rootSample(const class TrainFrame* frame,
                                      class BitMatrix* bag,
                                      unsigned int tIdx) const;

  /**
     @brief Returns zero, indicating no weight matrix for this class.
   */
  size_t getWeightSize() const {
    return 0;
  }

  /**
     @brief Dummy:  no weight matrix.
   */
  void dumpWeight(double weightOut[]) const {
  }
};


/**
   @brief Container for the crescent categorical probability vector.
 */
class ProbCresc {
  const unsigned int nCtg; // Response cardinality.
  size_t treeFloor; // Running position of start of tree.
  vector<size_t> height; // Height, per tree.
  vector<double> prob; // Raw probability values.
  const double forestScale;  // Forest-wide scaling factor for score.

public:

  ProbCresc(unsigned int treeChunk,
            unsigned int nCtg_,
            double scale_);

  
  /**
     @brief Derives score at a given leaf index.

     @param leafIdx is a block-relative leaf index.

     @return encoded score of leaf:  category + weight.
   */
  double leafScore(IndexT leafIdx) const;

  
  /**
     @brief Allocates and initializes items for the current tree.

     @param leafCount is the number of leaves in this tree.

     @param tIdx is the block-relative tree index.
   */
  void treeInit(unsigned int leafCount,
                unsigned int tIdx);


  /**
     @brief Dumps the probability vector.

     N.B.:  The height vector can be recomputed from that of the
     Leaf container, so need not be dumped.

     @param[out] probOut outputs the probability values.
   */
  void dump(double *probOut) const;


  /**
     @return count of items in the container.
   */
  size_t size() const {
    return height.empty() ? 0 : height.back();
  }
  

  /**
     @brief Computes per-category probabilities for each leaf.

     @param sample summarizes the sampled response.

     @param leafMap assigns samples to terminal node indices.

     @param leafCount is the number of leaves in the tree.
   */
  void probabilities(const Sample* sample,
                     const vector<IndexT>& leafMap,
                     unsigned int leafCount);
  

  /**
     @brief Normalizes the probability at each categorical entry.

     @param leafIdx is the tree-relative leaf index.

     @param sum is the normalization factor.
   */
  void normalize(IndexT leafIdx, double sum);
};


/**
   @brief Training members and methods for categorical response.
 */
class LFTrainCtg : public LFTrain {
  const unsigned int* yCtg; // 0-based factor-valued response.

  unique_ptr<ProbCresc> probCresc; // Crescent probability matrix.


  /**
     @brief Sets the scores for leaves in a tree.

     @param sample summarizes the sampled response.

     @param leafMap maps sample indices into leaf indices.
   */
  void setScores(const class Sample* sample,
                 const vector<IndexT>& leafMap);

  /**
     @brief Initialzes leaf state for current tree.

     @param sample summarizes the sampled response.

     @param leafMap maps sample indices into leaf indices.

     @param tIdx is the block-relative tree index.
   */
  void treeInit(const Sample* sample,
                const vector<IndexT>& leafMap,
                unsigned int tIdx);
public:
  LFTrainCtg(const unsigned int* yCtg_,
             const double* proxy,
             unsigned int treeChunk,
             unsigned int nCtg,
             double scale);

  ~LFTrainCtg(){}

  /**
     @brief Dumps probability matrix.

     @param[out] probOut exports the dumped values.
   */
  void dumpWeight(double probOut[]) const;

  /**
     @brief Gets the size of the probability vector.
   */
  size_t getWeightSize() const {
    return probCresc->size();
  }

  /**
     @brief Samples response of current tree.

     @param frame summarizes the presorted observations.

     @param bag summarizes the in-bag rows for a collection of trees.

     @param tIdx is the block-relative index of the current tree.

     @return summary of sampled response.
   */
  unique_ptr<class Sample> rootSample(const class TrainFrame* frame,
                                      class BitMatrix* bag,
                                      unsigned int tIdx) const;
};


#endif
