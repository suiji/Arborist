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

#ifndef ARBORIST_LEAF_H
#define ARBORIST_LEAF_H

#include "typeparam.h"
#include "jagged.h"
#include "sample.h"

#include <vector>
#include "math.h"

/**
   @brief The essential contents of a leaf.
 */
class Leaf {
  double score;
  unsigned int extent; // # distinct samples mapped to this leaf.
  
 public:

  inline void init() {
    score = 0.0;
    extent = 0;
  }

  
  /**
     @brief Getter for fully-accumulated extent value.
   */
  inline unsigned int getExtent() const {
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


class BagSample {
  unsigned int leafIdx; // Leaf index within tree.
  unsigned int sCount; // # times bagged:  > 0

 public:
  BagSample() {
  }
  
  BagSample(unsigned int leafIdx_,
          unsigned int sCount_) : leafIdx(leafIdx_), sCount(sCount_) {
  }

  
  inline unsigned int getLeafIdx() const {
    return leafIdx;
  }

  
  inline unsigned int getSCount() const {
    return sCount;
  }
};


/**
   @brief Leaf block for the crescent frame.
 */
class LBCresc {
  vector<Leaf> leaf;
  vector<size_t> height;
  unsigned int leafCount; // Count of leaves in current tree.
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
  void treeInit(const vector<unsigned int> &leafMap,
                unsigned int tIdx);
  

  /**
     @brief Writes the current tree origin and sets per-leaf extents.

     @param leafMap maps sample indices to tree indices.
  */
  void setExtents(const vector<unsigned int> &leafMap);


  /**
     @brief Sets regression-mode scores for all leaves in tree.

     @param sample summarizes the sample tree response.

     @param leafMap maps sample indices to leaf indices.
   */
  void setScoresReg(const Sample* sample,
                    const vector<unsigned int>& leafMap);


  void setScoresCtg(const class ProbCresc* probCresc);

  /**
     @brief Accumulates a score for a leaf in the current tree.

     @param leafIdx is a tree-relative leaf index.

     @param sum is the value to add to the leaf's score.
   */
  inline void scoreAccum(unsigned int leafIdx,
                        double sum) {
    leaf[treeFloor + leafIdx].scoreAccum(sum);
  }


  /**
     @brief Scales the score of a leaf in the current tree.

     @param leafIdx is a tree-relative leaf index.

     @param recipSum is the value by which to scale the score.
   */
  inline void scoreScale(unsigned int leafIdx,
                         double recipSum) {
    leaf[treeFloor + leafIdx].scoreScale(recipSum);
  }


  /**
     @brief Score setter for a leaf in the current tree.

     @param leafIdx is a tree-relative leaf index.

     @param score is the value to set.
   */
  inline void setScore(unsigned int leafIdx, double score) {
    leaf[treeFloor + leafIdx].setScore(score);
  }
  
  /**
     @brief Score getter for a leaf in the current tree.

     @param leafIdx is a tree-relative leaf index.
   */
  inline double getScore(unsigned int leafIdx) const {
    return leaf[treeFloor + leafIdx].getScore();
  }

  
  /** 
    @brief Serializes the internally-typed objects.

    @param[out] leafRaw outputs the raw content bytes.
  */
  void dumpRaw(unsigned char *leafRaw) const;
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
                 const vector<unsigned int> &leafMap);


  void dumpRaw(unsigned char blRaw[]) const; 
};


/**
   @brief Crescent LeafFrame implementation for training.
 */
class LFTrain {
protected:

  const double* y;
  unique_ptr<LBCresc> lbCresc; // Leaf block
  unique_ptr<BBCresc> bbCresc; // BagSample block

public:

  LFTrain(const double* y_,
          unsigned int treeChunk);


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

  
  virtual ~LFTrain();


  /**
     @brief Samples (bags) the response to construct the tree root.

     @param rowRank summarizes the predictor orderings.

     @param treeBag encodes the bagged rows.

     @param tIdx is the block-relative tree index.
   */
  virtual shared_ptr<class Sample> rootSample(const class RowRank* rowRank,
                                              class BitMatrix* bag,
                                              unsigned int tIdx) const = 0;


  virtual void setScores(const class Sample* sample,
                         const vector<unsigned int>& leafMap) = 0;

  
  /**
     @brief Allocates and initializes records for each leaf in tree.

     @param tIdx is the block-relative tree index.
   */
  virtual void treeInit(const class Sample* sample,
                        const vector<unsigned int>& leafMap,
                        unsigned int tIdx);

  /**
     @brief Appends this tree's leaves to the current block.

     @param sample summarizes the sampling environment of the current tree.

     @param leafMap maps sample indices to the index of the containing leaf.

     @param tIdx is the block-relative tree index.
   */
  void blockLeaves(const class Sample *sample,
                   const vector<unsigned int> &leafMap,
                   unsigned int tIdx);


  /** 
    @brief Serializes the internally-typed objects, 'Leaf', as well
    as the unsigned integer (packed bit) vector, "bagBits".
  */
  void cacheNodeRaw(unsigned char *leafRaw) const;
  void cacheBLRaw(unsigned char *blRaw) const;
  

  const vector<size_t>& getLeafHeight() const {
    return lbCresc->getHeight();
  }


  const vector<size_t>& getBagHeight() const {
    return bbCresc->getHeight();
  }
};


class LFTrainReg : public LFTrain {

  void setScores(const class Sample* sample,
                 const vector<unsigned int>& leafMap);

public:
  LFTrainReg(const double*y,
             unsigned int treeChunk);

  ~LFTrainReg();

  shared_ptr<class Sample> rootSample(const class RowRank* rowRank,
                                      class BitMatrix* bag,
                                      unsigned int tIdx) const;
};


/**
   @brief Container for the crescent categorical probability vector.
 */
class ProbCresc {
  unsigned int nCtg;
  size_t treeFloor;
  unsigned int leafCount;
  vector<size_t> height;
  vector<double> prob;
  const double forestScale;  // Forest-wide scaling factor for score.
  
public:

  ProbCresc(unsigned int treeChunk,
            unsigned int nCtg_,
            double scale_);

  
  /**
     @brief Accumulates the score at a given coordinate.
   */
  void accum(unsigned int tIdx,
             size_t leafIdx,
             unsigned int ctg,
             double sum);


  /**
     @brief Derives score at a given leaf index.

     @param leafIdx is a block-relative leaf index.

     @param recipSum is a normalizing factor.

     @return final score of leaf.
   */
  double leafScore(unsigned int leafIdx) const;

  /**
     @brief Allocates and initializes items for the current tree.

     @param leafCount is the number of leaves in this tree.
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
                     const vector<unsigned int>& leafMap,
                     unsigned int leafCount);
  

  /**
     @brief Normalizes the probability at each categorical entry.

     @param leafIdx is the tree-relative leaf index.

     @param recipSum is the normalization factor.
   */
  void normalize(unsigned int leafIdx, double recipSum);

  
  /**
     @brief Accumulates (unnormalized) probability at a given coordinate.

     @param leafIdx is the block relative leaf index.

     @param ctg is the category.

     @param incr is the quantity to accumulate.
   */
  inline void accum(unsigned int leafIdx,
                    unsigned int ctg,
                    double incr) {
    prob[treeFloor + leafIdx*nCtg + ctg] += incr;
  }

  
  /**
     @brief Normalizes the probability at a given coordinate.

     @param leafIdx is the block-relative leaf index.

     @param ctg is the category.

     @return final normalized probability at category.
   */
  inline void normalize(unsigned int leafIdx,
                        unsigned int ctg,
                        double recipSum) {
    prob[treeFloor + leafIdx*nCtg + ctg] *= recipSum;
  }
};


class LFTrainCtg : public LFTrain {
  const unsigned int* yCtg; // 0-based factor-valued response.

  unique_ptr<ProbCresc> probCresc;
  const unsigned int nCtg;

  void setScores(const class Sample* sample,
                 const vector<unsigned int>& leafMap);

  void treeInit(const Sample* sample,
                const vector<unsigned int>& leafMap,
                unsigned int tIdx);
public:
  LFTrainCtg(const unsigned int* yCtg_,
             const double* proxy,
             unsigned int treeChunk,
             unsigned int _nCtg,
             double scale);

  ~LFTrainCtg();


  void dumpProb(double probOut[]) const;

  /**
     @brief Gets the size of the probability vector.
   */
  inline size_t getProbSize() const {
    return probCresc->size();
  }


  shared_ptr<class Sample> rootSample(const class RowRank* rowRank,
                                      class BitMatrix* bag,
                                      unsigned int tIdx) const;
};


class LeafBlock {
  const unique_ptr<JaggedArray<const Leaf*, const unsigned int*> > raw;
  const size_t noLeaf;

public:
  LeafBlock(const unsigned int nTree_,
            const unsigned int* height_,
            const Leaf* leaf_);

  size_t size() const {
    return raw->size();
  }


  unsigned int nTree() const {
    return raw->getNMajor();
  };
  

  /**
     @brief Accumulates individual leaf extents across the forest.

     @return forest-wide offset vector.
   */
  vector<size_t> setOffsets() const;


  void regAcross(const unsigned int* predictLeaves,
                 double defaultScore,
                 double* yPred) const;

  void ctgAcross(const unsigned int predictLeaves[],
                 unsigned int ctgDefault,
                 double prediction[]) const;

  /**
     @brief Index-parametrized score getter.

     @param idx is the absolute index of a leaf.

     @return score at leaf.
   */
  const double getScore(unsigned int idx) const {
    return raw->items[idx].getScore();
  }


  const auto absOffset(unsigned int tIdx, unsigned int leafIdx) const {
    return raw->absOffset(tIdx, leafIdx);
  }


  /**
     @return beginning leaf offset for tree.
   */
  const auto treeBase(unsigned int tIdx) const {
    return raw->majorOffset(tIdx);
  }

  /**
     @brief Coordinate-parametrized score getter.

     @param tIdx is the tree index.

     @param idx is the tree-relative index of a leaf.

     @return score at leaf.
   */
  inline double getScore(unsigned int tIdx, unsigned int idx) const {
    auto absOff = raw->absOffset(tIdx, idx);
    return raw->items[absOff].getScore();
  }


  const unsigned int getExtent(unsigned int idx) const {
    return raw->items[idx].getExtent();
  }

  /**
     @brief Dumps leaf members into separate per-tree vectors.

     @param[out] score ouputs per-tree vectors of leaf scores.

     @param[out] extent outputs per-tree vectors of leaf extents.
   */
  void dump(vector<vector<double> >& score,
            vector<vector<unsigned int> >& extent) const;
};


/**
   @brief Jagged vector of bagging summaries.
 */
class BLBlock {
  const unique_ptr<JaggedArray<const BagSample*, const unsigned int*> > raw;

public:
  BLBlock(const unsigned int nTree_,
          const unsigned int*height_,
          const BagSample* bagSample_);


  size_t size() const {
    return raw->size();
  }


  void dump(const class BitMatrix* baggedRows,
            vector<vector<unsigned int> >& rowTree,
            vector<vector<unsigned int> >& sCountTree) const;


  /**
     @brief Index-parametrized sample-count getter.
   */
  const unsigned int getSCount(unsigned int idx) const {
    return raw->items[idx].getSCount();
  };


  /**
     @brief Index-parametrized leaf-index getter.
   */
  const unsigned int getLeafIdx(unsigned int idx) const {
    return raw->items[idx].getLeafIdx();
  };
};


/**
   @brief Block of leaves for fully-trained forest.
 */
class LeafFrame {
protected:
  const unsigned int nTree;
  unique_ptr<LeafBlock> leafBlock;
  unique_ptr<BLBlock> blBlock;

public:
  LeafFrame(const unsigned int* nodeHeight_,
       unsigned int nTree_,
       const class Leaf* leaf_,
       const unsigned int bagHeight_[],
       const class BagSample* bagSample_);

  const size_t noLeaf; // Exit

  virtual ~LeafFrame();
  virtual const unsigned int rowPredict() const = 0;

  /**
     @brief Sets scores for a block of rows.

     @param rowStart is the beginning row index.

     @param rowEnd is the final row index.

     @return
  */
  virtual void scoreBlock(const unsigned int* predictLeaves,
                          unsigned int rowStart,
                          unsigned int rowEnd) = 0;

  /**
     @brief Accessor for # trees in forest.
   */
  inline unsigned int getNTree() const {
    return nTree;
  }

  
  /**
     @brief Accessor for #samples at a given index.
   */
  inline unsigned int getSCount(unsigned int sIdx) const {
    return blBlock->getSCount(sIdx);
  }

  /**
     @brief Computes sum of all bag sizes.

     @return size of information vector, which represents all bagged samples.
  */
  inline unsigned int bagSampleTot() const {
    return blBlock->size();
  }


  /**
     @brief Determines inattainable leaf index value from leaf
     vector.

     @return inattainable leaf index value.
   */
  inline auto getNoLeaf() const {
    return noLeaf;
  }


  /**
     @brief computes total number of leaves in forest.

     @return size of leaf vector.
   */
  inline auto leafCount() const {
    return leafBlock->size();
  }

  
  /**
     @brief Dumps block components into separate tree-based vectors.
   */
  void dump(const class BitMatrix *baggedRows,
            vector< vector<unsigned int> > &rowTree,
            vector< vector<unsigned int> > &sCountTree,
            vector<vector<double> >& scoreTree,
            vector<vector<unsigned int> >& extentTree) const;
};


class LeafFrameReg : public LeafFrame {
  const double *yTrain;
  const double meanTrain; // Mean of training response.
  const vector<size_t> offset; // Accumulated extents.
  double defaultScore;
  vector<double> yPred;
  
 public:
  LeafFrameReg(const unsigned int nodeHeight_[],
          unsigned int nTree_,
          const class Leaf leaf_[],
          const unsigned int bagHeight_[],
          const class BagSample bagSample_[],
          const double *yTrain_,
          double meanTrain_,
          unsigned int rowPredict_);

  ~LeafFrameReg() {}

  inline const double *YTrain() const {
    return yTrain;
  }


  /**
     @brief Getter for prediction.
   */
  const vector<double> &getYPred() const {
    return yPred;
  }
  

  const unsigned int rowPredict() const {
    return yPred.size();
  }

  
  inline double MeanTrain() const {
    return meanTrain;
  }


  void scoreBlock(const unsigned int* predictLeaves,
                  unsigned int rowStart,
                  unsigned int rowEnd);
  
  /**
     @brief Computes bag index bounds in forest setting (Quant only).
  */
  void bagBounds(unsigned int tIdx,
                 unsigned int leafIdx,
                 unsigned int &start,
                 unsigned int &end) const {
    auto absIdx = leafBlock->absOffset(tIdx, leafIdx);
    start = offset[absIdx];
    end = start + leafBlock->getExtent(absIdx);
  }


  /**
     @brief Derives an absolute leaf index for a given tree and
     bag index.
     
     @param tIdx is a tree index.

     @param bagIdx is an absolute index of a bagged row.

     @param[out] offset_ outputs the absolute extent offset.

     @return absolute index of leaf containing the bagged row.
   */
  unsigned int getLeafIdx(unsigned int tIdx,
                          unsigned int bagIdx,
                          unsigned int &offset_) const {
    auto leafIdx = leafBlock->treeBase(tIdx) + blBlock->getLeafIdx(bagIdx);
    offset_ = offset[leafIdx];
    return leafIdx;
  }
};


/**
   @brief Specialization providing a subscript operation.
 */
template<>
class Jagged3<const double*, const unsigned int*> : public Jagged3Base<const double*, const unsigned int*> {
public:
  Jagged3(const unsigned int nCtg_,
          const unsigned int nTree_,
          const unsigned int* height_,
          const double *ctgProb_) :
    Jagged3Base<const double*, const unsigned int*>(nCtg_, nTree_, height_, ctgProb_) {
  }

  double getItem(unsigned int idx) const {
    return items[idx];
  }
};


/**
   @brief Categorical probabilities associated with indivdual leaves.

   Intimately accesses the raw jagged array it contains.
 */
class CtgProb {
  const unsigned int nCtg;
  vector<double> probDefault; // Forest-wide default probability.
  const vector<unsigned int> ctgHeight; // Scaled from Leaf's height vector.
  const unique_ptr<Jagged3<const double*, const unsigned int*> > raw;

  /**
     @brief Scales a vector of offsets by category count.

     @param leafHeight is the leaf-relative height vector.

     @param nTree is the number of trees.

     @return ad-hoc scaled vector.
   */
  vector<unsigned int> scaleHeight(const unsigned int* leafHeight,
                                   unsigned int nTree) const;

public:
  CtgProb(unsigned int ctgTrain,
          unsigned int nTree,
          const unsigned int* leafHeight,
          const double* prob);

  /**
     @brief Accumulates probabilities associated with a leaf.

     @param[in, out] probRow accumulates (unnormalized) probabilities across leaves.

     @param tIdx is the tree index.

     @param leafIdx is the tree-relative leaf index.
   */
  void addLeaf(double* probRow,
               unsigned int tIdx,
               unsigned int leafIdx) const;

  /**
     @brief Predicts probabilities across all trees.
   */
  void probAcross(const unsigned int* predictRow,
                  double* probRow,
                  unsigned int noLeaf) const;


  /**
     @brief Constructs the vector of default probabilities.
 */
  void setDefault();

  
  /**
     @brief Copies default probability vector into argument.

     @param[out] probPredict outputs the default category probabilities.
   */
  void applyDefault(double* probPredict) const;
  

  /**
     @return highest probability category of default vector.
   */
  unsigned int ctgDefault() const;

  
  void dump(vector<vector<double> >& probTree) const;
};

class LeafFrameCtg : public LeafFrame {
  const unsigned int ctgTrain;
  unique_ptr<CtgProb> ctgProb;
  
  vector<unsigned int> yPred;
  unsigned int ctgDefault;

 public:
  // Sized to zero by constructor.
  // Resized by bridge and filled in by prediction.
  vector<double> votes;
  vector<unsigned int> census;
  vector<double> prob;

  LeafFrameCtg(const unsigned int leafHeight_[],
          unsigned int nTree_,
          const class Leaf leaf_[],
          const unsigned int bagHeight_[],
          const class BagSample bagSample_[],
          const double prob_[],
          unsigned int ctgTrain_,
          unsigned int rowPredict_,
          bool doProb);

  ~LeafFrameCtg(){}


  /**
     @brief Getter for prediction.
   */
  const vector<unsigned int> &getYPred() {
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
  

  void scoreBlock(const unsigned int* predictLeaves,
                  unsigned int rowStart,
                  unsigned int rowEnd);

  
  void vote();

  
  inline unsigned int getCtgTrain() const {
    return ctgTrain;
  }
  

  /**
     @brief Derives an index into a matrix having stride equal to the
     number of training categories.

     @return derived strided index.
   */
  unsigned int ctgIdx(unsigned int row, unsigned int col) const {
    return row * ctgTrain + col;
  }


  void dump(const BitMatrix *baggedRows,
            vector<vector<unsigned int> > &rowTree,
            vector<vector<unsigned int> > &sCountTree,
            vector<vector<double> > &scoreTree,
            vector<vector<unsigned int> > &extentTree,
            vector<vector<double> > &_probTree) const;
};

#endif
