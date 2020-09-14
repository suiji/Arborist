// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predict.h

   @brief Data structures and methods for prediction.

   @author Mark Seligman
 */

#ifndef CART_PREDICT_H
#define CART_PREDICT_H

#include "block.h"
#include "typeparam.h"
#include "leafpredict.h" // Temporary:  Jagged3Base.

#include <vector>
#include <algorithm>


/**
   @brief Walks the decision forest for each row in a block, collecting
   predictions.
 */
class Predict {
protected:
  static const size_t rowChunk; // Block size.
  
  const class Bag* bag; // In-bag representation.
  const vector<size_t> treeOrigin; // Jagged accessor of tree origins.
  const struct TreeNode* treeNode; // Pointer to base of tree nodes.
  const class BVJagged* facSplit; // Jagged accessor of factor-valued splits.
  struct RLEFrame* rleFrame; // Frame of observations.
  const bool oob; // Whether to ignore in-bag rows.
  const unsigned int nPermute; // # times to permute each predictor.

  vector<IndexT> predictLeaves; // Tree-relative leaf indices.

  size_t blockStart; // Stripmine bound.
  size_t blockEnd; // "" ""
  vector<IndexT> accumNEst;

  size_t nEst; // Total number of estimands.

  /**
     @brief Assigns a true leaf index at the prediction coordinates passed.

     @param row is the row number.

     @param tc is the index of the current tree.

     @param leafIdx is the leaf index to record.
   */
  inline void predictLeaf(size_t row,
                          unsigned int tIdx,
                          IndexT leafIdx) {
    predictLeaves[nTree * (row - blockStart) + tIdx] = leafIdx;
  }


  /**
     @brief Driver for all-row prediction.
   */
  void predictRows();


  /**
     @brief Performs prediction on separately-permuted predictor columns.

     @param permute is the number of times to permute each predictor.
   */
  void predictPermute();
  

  /**
     @brief Drives prediction strip-mining and residual.
   */
  void blocks();
  
  
  /**
     @brief Strip-mines prediction by fixed-size blocks.
   */
  size_t predictBlock(size_t row,
		      size_t extent);


  /**
     @brief Multi-row prediction with predictors of only numeric.

     @param rowStart is the absolute starting row for the block.
  */
  void walkNum(size_t rowStart);

  /**
     @brief Multi-row prediction with predictors of only factor type.

     Parameters as above.
  */
  void walkFac(size_t rowStart);
  

  /**
     @brief Prediction with predictors of both numeric and factor type.
     Parameters as above.
  */
  void walkMixed(size_t rowStart);
  

  virtual void predictBlock() = 0;


  /**
     @brief Accumulates type-based estimates.
   */
  virtual void estAccum();


  virtual void setPermuteTarget(PredictorT predIdx) = 0;

public:

  class LeafBlock* leafBlock; // 
  const PredictorT nPredNum;
  const PredictorT nPredFac;
  const size_t nRow;
  const unsigned int nTree; // # trees used in training.
  const IndexT noLeaf; // Inattainable leaf index value.

  /**
     @brief Aliases a row-prediction method tailored for the frame's
     block structure.
   */
  void (Predict::* walkTree)(size_t);

  vector<unsigned int> trFac; // OTF transposed factor observations.
  vector<double> trNum; // OTF transposed numeric observations.
  vector<size_t> trIdx; // Most recent RLE index accessed by predictor.

  Predict(const class Bag* bag_,
          const class Forest* forest_,
          const class LeafPredict* leaf_,
	  struct RLEFrame* rleFrame_,
	  bool oob_,
	  unsigned int nPredict_);

  
  /**
     @brief Main entry from bridge.

     Distributed prediction will require start and extent parameters.
   */
  void predict();

  
    /**
     @brief Computes block-relative position for a predictor.

     @param[out] thisIsFactor outputs true iff predictor is factor-valued.

     @return block-relative index.
   */
  inline unsigned int getIdx(unsigned int predIdx, bool &predIsFactor) const {
    predIsFactor = isFactor(predIdx);
    return predIsFactor ? predIdx - nPredNum : predIdx;
  }

  
  inline bool isFactor(unsigned int predIdx) const {
    return predIdx >= nPredNum;
  }


  inline unsigned int getNTree() const {
    return nTree;
  }

  
  /**
     @param[out] termIdx is the terminal index of prediction.

     @return true iff the predicted terminal index references a leaf.
   */
  inline bool isLeafIdx(size_t row,
			unsigned int tIdx,
			IndexT& termIdx) const {
    termIdx = predictLeaves[nTree * (row - blockStart) + tIdx];

    // Non-oob scenarios should always see a leaf.
    //    if (!oob) assert(termIdx != noLeaf);
    return termIdx != noLeaf;
  }


  /**
     @brief As above, but outputs leaf score.
   */
  inline bool isLeafIdx(size_t row,
			unsigned int tIdx,
			double& score) const {
    IndexT termIdx = predictLeaves[nTree * (row - blockStart) + tIdx];
    if (termIdx != noLeaf) {
      score = leafBlock->getScore(tIdx, termIdx);
      return true;
    }
    else {
      return false;
    }
    // Non-oob scenarios should always see a leaf.
    //    if (!oob) assert(termIdx != noLeaf);
  }

  
  /**
     @brief Computes pointer to base of row of numeric values.

     @param row is the row number.

     @return base address for numeric values at row.
  */
  const double* baseNum(size_t row) const;


  /**
     @brief As above, but factor varlues.

     @return row is the row number.
   */
  const PredictorT* baseFac(size_t row) const;

  
  /**
     @brief Prediction of single row with mixed predictor types.

     @param row is the absolute row of data over which a prediction is made.

     @param blockRow is the row's block-relative row index.

     @return index of leaf predicted.
  */
  void rowMixed(unsigned int tIdx,
		  const double* rowNT,
		  const unsigned int* rowFT,
		  size_t row);

  
  /**
     @brief Prediction over a single row with factor-valued predictors only.

     Parameters as in mixed case, above.
  */
  void rowFac(unsigned int tIdx,
		const unsigned int* rowT,
		size_t row);


  /**
     @brief Prediction of a single row with numeric-valued predictors only.

     Parameters as in mixed case, above.
   */
  void rowNum(unsigned int tIdx,
		const double* rowT,
		size_t row);
};


class PredictReg : public Predict {
  const double defaultScore;
  const vector<double> yTest;
  vector<double> yPred;
  vector<double> yPermute; // Reused.

  vector<double> accumAbsErr; // One slot per predictor.
  vector<double> accumSSE; // " "

  double saePredict;
  double ssePredict;
  vector<double> saePermute;
  vector<double> ssePermute;

  unique_ptr<class Quant> quant;  // Quantile workplace, as needed.

  vector<double>* yTarg; // Target of current prediction.
  double* saeTarg;
  double* sseTarg;
  
  void testRow(size_t row);


  void scoreRow(size_t row);

public:
  PredictReg(const class Bag* bag_,
	      const class Forest* forest_,
	      const class LeafPredict* leaf_,
	      struct RLEFrame* rleFrame_,
	      vector<double> yTrain,
	      double default_,
	      vector<double> yTest_,
	      bool oob_,
	      unsigned int nPredict_,
	      vector<double> quantile);

  ~PredictReg();


  double getDefault() const {
    return defaultScore;
  }

  
  /**
     @brief Description given in virtual declartion.
   */
  void predictBlock();


  void estAccum();


  void setPermuteTarget(PredictorT predIdx);


  double getSSE() const {
    return ssePredict;
  }


  const vector<double>& getSSEPermute() const {
    return ssePermute;
  }


  
  double getSAE() const {
    return saePredict;
  }


  const vector<double>& getSAEPermute() const {
    return saePermute;
  }
  

  const vector<double>& getYTest() const {
    return yTest;
  }
  
  
  const vector<double>& getYPred() const {
    return yPred;
  }


  inline double getYPred(size_t row) const {
    return yPred[row];
  }
  

 /**
     @return vector of estimated quantile means.
   */
  const vector<double> getQEst() const;


  /**
     @return vector quantile predictions.
  */
  const vector<double> getQPred() const;
};


class PredictCtg : public Predict {
  vector<PredictorT> yTest;
  vector<PredictorT> yPred;
  const PredictorT nCtgTrain; // Cardiality of training response.
  const PredictorT nCtgMerged; // Cardinality of merged test response.
  unique_ptr<class CtgProb> ctgProb; // Matrix (row * ctg) of predicted probabilities.
  const PredictorT ctgDefault; // Default prediction when nothing is out-of-bag.

  vector<PredictorT> yPermute; // Reused.
  vector<double> votes; // Jittered prediction counts.
  vector<PredictorT> census;
  vector<size_t> confusion; // Confusion matrix; saved.
  vector<double> misprediction; // Mispredction, by merged category; saved.
  double oobPredict; // Out-of-bag error:  % mispredicted rows.
  vector<double> prob;
  vector<PredictorT> censusPermute; // Workspace census for permutations.
  vector<size_t> confusionPermute; // Workspace for permutation.
  vector<vector<double>> mispredPermute; // Saved values for permutation.
  vector<double> oobPermute;
  vector<PredictorT>* yTarg; // Target of current prediction.
  vector<size_t>* confusionTarg;
  vector<PredictorT>* censusTarg; // Destination of prediction census.
  vector<double>* mispredTarg;
  double *oobTarg;

  void testRow(size_t row);


  void scoreRow(size_t row);

  
public:

  PredictCtg(const class Bag* bag_,
	      const class Forest* forest_,
	      const class LeafPredict* leaf_,
	      struct RLEFrame* rleFrame_,
	      const unsigned int* leafHeight,
	      const double* leafProbs,
	     unsigned int nCtgTrain_,
	     vector<PredictorT> yTest_,
	      bool oob_,
	      unsigned int nPredict_,
	      bool doProb);

  ~PredictCtg() {}


  /**
     @brief Description given in virtual declartion.
   */
  void predictBlock();

  
  /**
     @Brief Assignes categorical score by plurality vote.
   */
  PredictorT argMax(size_t row);


  /**
     @brief Derives an index into a matrix having stride equal to the
     number of training categories.
     
     @param row is the row coordinate.

     @return derived strided index.
   */
  size_t ctgIdx(size_t row, PredictorT ctg = 0) const {
    return row * nCtgTrain + ctg;
  }


  const vector<PredictorT>& getYPred() const {
    return yPred;
  }


  const vector<size_t>& getConfusion() const {
    return confusion;
  }


  const vector<double>& getMisprediction() const {
    return misprediction;
  }


  const vector<vector<double>>& getMispredPermute() const {
    return mispredPermute;
  }


  double getOOBError() const {
    return oobPredict;
  }    


  const vector<double>& getOOBErrorPermute() const {
    return oobPermute;
  }
  
  
  PredictorT getNCtgTrain() const {
    return nCtgTrain;
  }


  void estAccum();


  void setMisprediction();

  void setPermuteTarget(PredictorT predIdx);

  
  /**
     @brief Getter for census.
   */
  const PredictorT* getCensus() const {
    return &census[0];
  }

  /**
     @brief Getter for probability matrix.
   */
  const vector<double>& getProb() const {
    return prob;
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
  CtgProb(PredictorT ctgTrain,
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
               IndexT leafIdx) const;

  /**
     @brief Predicts probabilities across all trees.

     @param predictRow are the categorical predictions, per tree.

     @param[out] probRow outputs the per-category probabilities.
   */
  void probAcross(const class PredictCtg* predict,
		  size_t row,
                  double* probRow) const;


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
  PredictorT ctgDefault() const;
};


#endif
