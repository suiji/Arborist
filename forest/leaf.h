// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leaf.h

   @brief Class definitions for crescent leaf structures.

   @author Mark Seligman
 */

#ifndef FOREST_LEAF_H
#define FOREST_LEAF_H


#include "typeparam.h"
#include "jagged.h"


#include <vector>
#include <numeric>
#include <algorithm>


/**
   @brief Crescent LeafPredict implementation for training.
 */
class Leaf {

protected:

  void setScoreCtg(const vector<IndexT>& ctgCount,
		   const vector<double>& jitters,
		   IndexT leafIdx);

  
  /** 
    @brief Writes out the internal score.

    @param[out] scoreOut outputs the score.
  */
  void dumpScore(double scoreOut[]) const;


public:

  Leaf();


  virtual ~Leaf() {}

  /**
     @base Copies front-end vectors and lights off initializations specific to classification.

     @param yCtg is the zero-indexed response vector.

     @return void.
  */
  static unique_ptr<class LeafCtg> factoryCtg(const vector<unsigned int>& yCtg,
					      PredictorT nCtg);

  
  static unique_ptr<class LeafReg> factoryReg(const vector<double>& yNum);

  
  /**
     @brief Samples (bags) the response to construct the tree root.

     @param frame summarizes the predictor orderings.
   */
  virtual unique_ptr<class Sample> rootSample(const class TrainFrame* frame,
					      const vector<double>& yProxy) const = 0;


  /**
     @brief Appends this tree's leaves to the current block.

     @param sample summarizes the sampling environment of the current tree.

     @param leafMap maps sample indices to the index of the containing leaf.

     @return vector of scores for the leaves in this tree.
   */
  virtual vector<double> scoreTree(const class Sample* sample,
				   const vector<IndexT>& leafMap) = 0;

  
    /** 
    @brief Serializes the internally-typed objects, 'Leaf', as well
    as the unsigned integer (packed bit) vector, "bagBits".
  */
  void cacheScore(double scoreOut[]) const;
};


class LeafReg : public Leaf {
  const vector<double> yTrain; // Training response.

  /**
     @brief Sets scores for current tree's leaves.

     @param sample summarizes the sample response.

     @param leafMap maps sample indices to leaf indices.
   */
  vector<double> scoreTree(const class Sample* sample,
			   const vector<IndexT>& leafMap);

public:
  /**
     @brief Regression constructor.

     @param y is the training response.
   */
  LeafReg(const vector<double>& y);


  ~LeafReg(){}


  const vector<double>& getYTrain() const {
    return yTrain;
  }
  

  /**
     @brief Samples response of current tree.

     @param frame summarizes the presorted observations.

     @return summary of sampled response.
   */
  unique_ptr<class Sample> rootSample(const class TrainFrame* frame,
				      const vector<double>& yProxy) const;


  /**
     @brief Determines mean training value.

     @return mean trainig value.
   */
  double meanTrain() const {
    return yTrain.empty() ? 0.0 : accumulate(yTrain.begin(), yTrain.end(), 0.0) / yTrain.size();
  }
};


/**
   @brief Training members and methods for categorical response.
 */
class LeafCtg : public Leaf {
  const vector<PredictorT> yCtg; // 0-based factor-valued response.
  const PredictorT nCtg;
  
  
  /**
     @brief Counts the categories, by leaf.

     @return map of category counts, by leaf.
   */
  vector<PredictorT> countCtg(const vector<double>& score,
			      const class Sample* sample,
			      const vector<IndexT>& leafMap) const;


  double argMax(IndexT leafIdx,
		const vector<IndexT>& ctgCount,
		const vector<double>& jitters);


  /**
     @brief Sets the scores for leaves in a tree.

     @param sample summarizes the sampled response.

     @param leafMap maps sample indices into leaf indices.
   */
  vector<double> scoreTree(const class Sample* sample,
			   const vector<IndexT>& leafMap);



public:
  LeafCtg(const vector<PredictorT>& yCtg_,
	  PredictorT nCtg);


  ~LeafCtg(){}


  inline auto getCtg(IndexT row) const {
    return yCtg[row];
  }


  auto getNCtg() const {
    return nCtg;
  }
  

  /**
     @brief Samples response of current tree.

     @param frame summarizes the presorted observations.

     @return summary of sampled response.
   */
  unique_ptr<class Sample> rootSample(const class TrainFrame* frame,
				      const vector<double>& yProxy) const;


  /**
     @brief As above, but derives category boundary coordinates of a leaf.
  */
  void ctgBounds(const class Predict* predict,
		 unsigned int tIdx,
		 IndexT leafIdx,
		 size_t& start,
		 size_t& end) const;

  /**
     @brief Builds a height vector scaled for categoricity.

     @return vector of scaled heights.
  */
  vector<size_t> ctgHeight(const Predict* predict) const;
};


/**
   @brief Specialization providing a subscript operation.
 */
template<>
class Jagged3<const IndexT*, const size_t*> : public Jagged3Base<const IndexT*, const size_t*> {
public:
  Jagged3(const PredictorT nCtg_,
          const unsigned int nTree_,
          const size_t* height_,
          const IndexT* ctgProb_) :
    Jagged3Base<const IndexT*, const size_t*>(nCtg_, nTree_, height_, ctgProb_) {
  }

  ~Jagged3() {
  }

  /**
     @brief Getter for indexed item.

     @param idx is the item index.

     @return indexed item.
   */
  IndexT getItem(unsigned int idx) const {
    return items[idx];
  }
};


/**
   @brief Categorical probabilities associated with indivdual leaves.

   Intimately accesses the raw jagged array it contains.
 */
class CtgProb {
  const PredictorT nCtg; // Training cardinality.
  const vector<IndexT> ctgCount; // Category census, per leaf.
  const vector<size_t> ctgHeight; // Scaled from Leaf's height vector.
  const unique_ptr<Jagged3<const IndexT*, const size_t*> > raw;
  const vector<double> probDefault; // Forest-wide default probability.
  vector<double> probs; // Per-row probabilties.

  /**
     @brief Constructs a vector of default probabilities.

     @param leafCount is the number of leaves in the forest.

     @return empircal cdf over training response categories.
  */
  vector<double> ctgECDF(size_t leafCount);

  
  /**
     @brief Copies default probability vector into argument.

     @param[out] probPredict outputs the default category probabilities.
   */
  void applyDefault(double probPredict[]) const;
  

  /**
     @brief Accumulates probabilities associated with a leaf.

     @param[in, out] probRow accumulates (unnormalized) probabilities across leaves.

     @param tIdx is the tree index.

     @param leafIdx is the block-relative leaf index.
   */
  void readLeaf(vector<IndexT>& ctgRow,
		unsigned int tIdx,
		IndexT leafIdx) const;


public:
  CtgProb(const class Predict* predict,
	  const class LeafCtg* leaf,
	  const class Sampler* sampler,
	  bool doProb);

  ~CtgProb() {}


  /**
     @brief Predicts probabilities across all trees.

     @param row is the row number.

     @param[out] probRow outputs the per-category probabilities.
   */
  void predictRow(const class Predict* predict,
		  size_t row);

  
  /**
     @return highest probability category of default vector.
   */
  PredictorT ctgDefault() const {
    return max_element(probDefault.begin(), probDefault.end()) - probDefault.begin();
  }

  
  bool isEmpty() const {
    return probs.empty();
  }

  
  /**
     @brief Getter for probability vector.
   */
  const vector<double>& getProb() {
    return probs;
  }

  
  /**
     @brief Dumps the probability cells.
   */
  void dump() const;
};


#endif
