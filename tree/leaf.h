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

#ifndef CORE_LEAF_H
#define CORE_LEAF_H

#include "jagged.h"
#include "sample.h"
#include "typeparam.h"

#include <vector>
#include "math.h"

/**
   @brief The essential contents of a leaf.
 */
class Leaf {
  double score;
  unsigned int extent; // # distinct samples mapped to this leaf.
  
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


class BagSample {
  unsigned int leafIdx; // Leaf index within tree.
  unsigned int sCount; // # times bagged:  > 0

 public:
  BagSample() {
  }
  
  BagSample(unsigned int leafIdx_,
            unsigned int sCount_) : leafIdx(leafIdx_), sCount(sCount_) {
  }


  inline auto getLeafIdx() const {
    return leafIdx;
  }

  
  inline auto getSCount() const {
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
  virtual unique_ptr<class Sample> rootSample(const class SummaryFrame* frame,
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

  virtual void dumpWeight(double weightOut[]) const = 0;

  virtual size_t getWeightSize() const = 0;
  
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

  /**
     @brief Sets scores for current tree's leaves.

     @param sample summarizes the sample response.

     @param leafMap maps sample indices to leaf indices.
   */
  void setScores(const class Sample* sample,
                 const vector<unsigned int>& leafMap);

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
  unique_ptr<class Sample> rootSample(const class SummaryFrame* frame,
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
  double leafScore(unsigned int leafIdx) const;

  
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
                     const vector<unsigned int>& leafMap,
                     unsigned int leafCount);
  

  /**
     @brief Normalizes the probability at each categorical entry.

     @param leafIdx is the tree-relative leaf index.

     @param sum is the normalization factor.
   */
  void normalize(unsigned int leafIdx, double sum);
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
                 const vector<unsigned int>& leafMap);

  /**
     @brief Initialzes leaf state for current tree.

     @param sample summarizes the sampled response.

     @param leafMap maps sample indices into leaf indices.

     @param tIdx is the block-relative tree index.
   */
  void treeInit(const Sample* sample,
                const vector<unsigned int>& leafMap,
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
  unique_ptr<class Sample> rootSample(const class SummaryFrame* frame,
                                      class BitMatrix* bag,
                                      unsigned int tIdx) const;
};


class LeafBlock {
  const unique_ptr<JaggedArray<const Leaf*, const unsigned int*> > raw;
  const size_t noLeaf; // Inattainable leaf index value.

public:
  LeafBlock(const unsigned int nTree_,
            const unsigned int* height_,
            const Leaf* leaf_);

  /**
     @brief Accessor for size of raw vector.
   */
  size_t size() const {
    return raw->size();
  }

  /**
     @brief Accessor for tree count.
   */
  unsigned int nTree() const {
    return raw->getNMajor();
  };
  

  /**
     @brief Accumulates individual leaf extents across the forest.

     @return forest-wide offset vector.
   */
  vector<size_t> setOffsets() const;


  /**
     @brief Scores a numerical row across all trees.

     @param predictLeaves holds the leaf predictions for all trees
     over a block of rows.

     @param defaultScore is the default score if all trees in-bag.

     @param[out] yPred[] outputs scores over a block of rows.
   */
  void scoreAcross(const unsigned int* predictLeaves,
                   double defaultScore,
                   double yPred[]) const;

  /**
     @brief Scores a categorical row across all trees.

     @param predictLeaves holds the leaf predictions for a block of rows.

     @param ctgDefault is the default category if all trees in-bag.

     @param[out] yCtg[] outputs per-category scores over a block of rows.
   */
  void scoreAcross(const unsigned int predictLeaves[],
                   unsigned int ctgDefault,
                   double yCtg[]) const;

  /**
     @brief Index-parametrized score getter.

     @param idx is the absolute index of a leaf.

     @return score at leaf.
   */
  const double getScore(unsigned int idx) const {
    return raw->items[idx].getScore();
  }


  /**
     @brief Derives forest-relative offset of tree/leaf coordinate.

     @param tIdx is the tree index.

     @param leafIdx is a tree-local leaf index.

     @return absolute offset of leaf.
   */
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


  /**
     @brief Derives count of samples assigned to a leaf.

     @param leafIdx is the absolute leaf index.

     @return extent value.
   */
  const unsigned int getExtent(unsigned int leafAbs) const {
    return raw->items[leafAbs].getExtent();
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
          const unsigned int* height_,
          const BagSample* bagSample_);

  /**
     @brief Derives size of raw contents.
   */
  size_t size() const {
    return raw->size();
  }


  void dump(const class Bag* bag,
            vector<vector<size_t> >& rowTree,
            vector<vector<unsigned int> >& sCountTree) const;


  /**
     @brief Index-parametrized sample-count getter.
   */
  const unsigned int getSCount(unsigned int absOff) const {
    return raw->items[absOff].getSCount();
  };


  /**
     @brief Index-parametrized leaf-index getter.

     @param absOff is the forest-relative bag offset.

     @return associated tree-relative leaf index.
   */
  const unsigned int getLeafIdx(unsigned int absOff) const {
    return raw->items[absOff].getLeafIdx();
  };
};


/**
   @brief Block of leaves for fully-trained forest.
 */
class LeafFrame {
protected:
  const unsigned int nTree;  // # trees used to train.
  unique_ptr<LeafBlock> leafBlock; // Leaves.
  unique_ptr<BLBlock> blBlock; // Bag-sample summaries.

public:
  LeafFrame(const unsigned int* nodeHeight_,
       unsigned int nTree_,
       const class Leaf* leaf_,
       const unsigned int bagHeight_[],
       const class BagSample* bagSample_);

  const size_t noLeaf; // Inattainable leaf index value.

  virtual ~LeafFrame() {}
  virtual const unsigned int getRowPredict() const = 0;

  /**
     @brief Sets scores for a block of rows.

     @param predictLeaves are the leaf indices predicted at each row/tree pair.

     @param rowStart is the beginning row index.

     @param extent is the number of rows indexed.
  */
  virtual void scoreBlock(const unsigned int* predictLeaves,
                          size_t rowStart,
                          size_t extent) = 0;

  /**
     @brief Accessor for # trees in forest.
   */
  inline unsigned int getNTree() const {
    return nTree;
  }


  /**
     @brief Accessor for #samples at an absolute bag index.
   */
  inline unsigned int getSCount(unsigned int bagIdx) const {
    return blBlock->getSCount(bagIdx);
  }


  /**
     @param absSIdx is an absolute bagged sample index.

     @return tree-relative leaf index of bagged sample.
   */
  inline auto getLeafLoc(unsigned int absSIdx) const {
    return blBlock->getLeafIdx(absSIdx);
  }

  /**
     @brief Accessor for forest-relative leaf index .

     @param tIdx is the tree index.

     @param absSIdx is a forest-relative sample index.

     @return forest-relative leaf index.
   */
  inline unsigned int getLeafAbs(unsigned int tIdx,
                                 unsigned int absSIdx) const {
    return leafBlock->absOffset(tIdx, getLeafLoc(absSIdx));
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
  void dump(const class Bag* bag,
            vector< vector<size_t> >& rowTree,
            vector< vector<unsigned int> >& sCountTree,
            vector<vector<double> >& scoreTree,
            vector<vector<unsigned int> >& extentTree) const;
};


/**
   @brief Rank and sample-counts associated with bagged rows.

   Client:  quantile inference.
 */
struct RankCount {
  IndexT rank; // Training rank of row.
  unsigned int sCount; // # times row sampled.

  void init(unsigned int rank,
            unsigned int sCount) {
    this->rank = rank;
    this->sCount = sCount;
  }
};


class LeafFrameReg : public LeafFrame {
  const double *yTrain;
  const size_t rowTrain;
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
               size_t rowTrain,
               double meanTrain_,
               unsigned int rowPredict_);

  ~LeafFrameReg() {}

  /**
     @brief Accesor for training response, by row.

     @return training value at row.
   */
  inline const double* getYTrain() const {
    return yTrain;
  }


  /**
     @brief Accessor for training response length.
   */
  inline const size_t getRowTrain() const {
    return rowTrain;
  }


  /**
     @brief Getter for predicted values.

     @return pointer to base of prediction vector.
   */
  const vector<double> &getYPred() const {
    return yPred;
  }

  
  inline const double getYPred(unsigned int row) const {
    return yPred[row];
  }
  

  /**
     @brief Getter for number of rows to predict.

     @return size of prediction vector.
   */
  const unsigned int getRowPredict() const {
    return yPred.size();
  }

  
  inline double MeanTrain() const {
    return meanTrain;
  }


  /**
     @brief Description given in virtual declaration.
   */
  void scoreBlock(const unsigned int* predictLeaves,
                  size_t rowStart,
                  size_t extent);
  
  /**
     @brief Computes bag index bounds in forest setting (Quant only).

     @param tIdx is the absolute tree index.

     @param leafIdx is the tree-relative leaf index.

     @param[out] start outputs the staring sample offset.

     @param[out] end outputs the final sample offset. 
  */
  inline void bagBounds(unsigned int tIdx,
                        unsigned int leafIdx,
                        unsigned int &start,
                        unsigned int &end) const {
    auto leafAbs = leafBlock->absOffset(tIdx, leafIdx);
    start = offset[leafAbs];
    end = start + leafBlock->getExtent(leafAbs);
  }
  

  /**
     @brief Builds row-ordered mapping of leaves to rank/count pairs.

     @param baggedRows encodes the forest-wide tree bagging.

     @param row2Rank is the ranked training outcome.

     @return per-leaf vector expressing mapping.
   */
  vector<RankCount> setRankCount(const class BitMatrix* baggedRows,
                                 const vector<IndexT>& row2Rank) const;
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

  ~Jagged3() {
  }

  /**
     @brief Getter for indexed item.

     @param idx is the item index.

     @return indexed item.
   */
  double getItem(unsigned int idx) const {
    return items[idx];
  }
};


/**
   @brief Categorical probabilities associated with indivdual leaves.

   Intimately accesses the raw jagged array it contains.
 */
class CtgProb {
  const unsigned int nCtg; // Training cardinality.
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

  ~CtgProb() {}

  /**
     @brief Accumulates probabilities associated with a leaf.

     @param[in, out] probRow accumulates (unnormalized) probabilities across leaves.

     @param tIdx is the tree index.

     @param leafIdx is the block-relative leaf index.
   */
  void addLeaf(double* probRow,
               unsigned int tIdx,
               unsigned int leafIdx) const;

  /**
     @brief Predicts probabilities across all trees.

     @param predictRow are the categorical predictions, per tree.

     @param noLeaf characterizes an unscored tree.

     @param[out] probRow outputs the per-category probabilities.
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

  /**
     @brief Dumps probability values.

     @param[out] probTree outputs the per-tree tables.
   */  
  void dump(vector<vector<double> >& probTree) const;
};

class LeafFrameCtg : public LeafFrame {
  const unsigned int ctgTrain; // Response training cardinality.
  unique_ptr<CtgProb> ctgProb; // Matrix (row * ctg) of predicted probabilities.
  
  vector<unsigned int> yPred; // Per-row vector of predicted categories.
  unsigned int ctgDefault; // Default score for rows with no out-of-bag trees.

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


  const unsigned int getYPred(size_t row) {
    return yPred[row];
  }
  

  /**
     @brief Getter for number of rows to predict.
   */
  const unsigned int getRowPredict() const {
    return yPred.size();
  }


  /**
     @brief Getter for census.
   */
  const unsigned int* getCensus() const {
    return &census[0];
  }

  /**
     @brief Getter for probability matrix.
   */
  const vector<double>& getProb() const {
    return prob;
  }
  
  /**
     @brief Description given in virtual declartion.
   */
  void scoreBlock(const unsigned int* predictLeaves,
                  size_t rowStart,
                  size_t extent);

  /**
     @brief Fills the vote table using predicted response.
   */  
  void vote();

  /**
     @brief Getter for training cardinality.

     @return value of ctgTrain.
   */
  inline unsigned int getCtgTrain() const {
    return ctgTrain;
  }
  

  /**
     @brief Derives an index into a matrix having stride equal to the
     number of training categories.
     
     @param row is the row coordinate.

     @param col is the column coordinate.

     @return derived strided index.
   */
  unsigned int ctgIdx(unsigned int row, unsigned int col) const {
    return row * ctgTrain + col;
  }


  /**
     @brief Dumps bagging and leaf information into per-tree vectors.
   */
  void dump(const class Bag* bag,
            vector<vector<size_t> > &rowTree,
            vector<vector<unsigned int> > &sCountTree,
            vector<vector<double> > &scoreTree,
            vector<vector<unsigned int> > &extentTree,
            vector<vector<double> > &_probTree) const;
};

#endif
