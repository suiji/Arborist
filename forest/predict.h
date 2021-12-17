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

#ifndef FOREST_PREDICT_H
#define FOREST_PREDICT_H

#include "block.h"
#include "typeparam.h"
#include "bv.h"
#include "leaf.h"

#include <vector>
#include <algorithm>


/**
   @brief Walks the decision forest for each row in a block, collecting
   predictions.
 */
class Predict {
protected:
  static const size_t scoreChunk; // Score block dimension.
  static const unsigned int seqChunk;  // Effort to minimize false sharing.
  
  const class Sampler* sampler; // In-bag representation.
  const vector<size_t> treeOrigin; // Jagged accessor of tree origins.
  const struct TreeNode* treeNode; // Pointer to base of tree nodes.
  const class BVJagged* facSplit; // Jagged accessor of factor-valued splits.
  struct RLEFrame* rleFrame; // Frame of observations.
  const bool testing; // Whether to compare prediction with test vector.
  const unsigned int nPermute; // # times to permute each predictor.

  vector<IndexT> predictLeaves; // Tree-relative leaf indices.

  size_t blockStart; // Stripmine bound.
  size_t blockEnd; // "" ""
  vector<IndexT> accumNEst;

  size_t nEst; // Total number of estimands.
  
  
  /**
     @return vector of accumulated score heights.
   */
  static vector<size_t> scoreHeights(const vector<vector<double>>& scoreBlock);
  

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

     @param[in, out] trIdx caches RLE index accessed by a predictor.
   */
  size_t predictBlock(size_t row,
		      size_t extent,
		      vector<size_t>& trIdx);


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
  

  /**
     @brief Strip-mines prediction by block.
   */
  void predictBlock();

  
  /**
     @brief Predicts sequentially to minimize false sharing.
   */
  virtual void scoreSeq(size_t rowStart,
			size_t rowEnd) = 0;


  /**
     @brief Accumulates type-based estimates.
   */
  virtual void estAccum();


  virtual void setPermuteTarget(PredictorT predIdx) = 0;

public:

  const vector<vector<double>> scoreBlock; // Scores, by tree.
  const vector<size_t> scoreHeight; // Accumulated heights of scores.
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

  Predict(const class Forest* forest_,
	  const class Sampler* sampler_,
	  struct RLEFrame* rleFrame_,
	  bool testing_,
	  unsigned int nPredict_);
  

  /**
     @brief Main entry from bridge.

     Distributed prediction will require start and extent parameters.
   */
  void predict();


  const class Sampler* getSampler() const {
    return sampler;
  }


  size_t getScoreIdx(unsigned int tIdx,
		     IndexT leafIdx) const {
    return leafIdx + (tIdx == 0 ? 0 : scoreHeight[tIdx-1]);
  }

  
  /**
     @brief Obtains the sample index bounds of a leaf.

     @param[out] start is the starting sample index.

     @param[out] end is the end sample index.
   */
  void sampleBounds(unsigned int tIdx,
		    IndexT leafIdx,
		    size_t& leafStart,
		    size_t& leafEnd) const;


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


  size_t getNRow() const {
    return nRow;
  }
  

  inline unsigned int getNTree() const {
    return nTree;
  }


  size_t getScoreCount() const {
    return scoreHeight.back();
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
      score = scoreBlock[tIdx][termIdx];
      return true;
    }
    else {
      return false;
    }
    // Non-bagging scenarios should always see a leaf.
    //    if (!bagging) assert(termIdx != noLeaf);
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
  const class LeafReg* leaf;
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


  unsigned int scoreRow(size_t row);

public:
  PredictReg(const class Forest* forest_,
	     const class Sampler* sampler_,
	      struct RLEFrame* rleFrame_,
	      const vector<double>& yTest_,
	      unsigned int nPredict_,
	      const vector<double>& quantile);

  //  ~PredictReg(); // Forward declaration:  not specified default.

  
  /**
     @brief Description given in virtual declartion.
   */
  void scoreSeq(size_t rowStart,
		size_t rowEnd);


  void estAccum();


  void setPermuteTarget(PredictorT predIdx);


  double getSSE() const {
    return ssePredict;
  }


  const vector<double>& getSSEPermuted() const {
    return ssePermute;
  }


  
  double getSAE() const {
    return saePredict;
  }


  const vector<double>& getSAEPermuted() const {
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
  const class LeafCtg* leaf;
  const vector<PredictorT> yTest;
  vector<PredictorT> yPred;
  const PredictorT nCtgTrain; // Cardiality of training response.
  const PredictorT nCtgMerged; // Cardinality of merged test response.
  unique_ptr<class CtgProb> ctgProb; // Class prediction probabilities.

  vector<PredictorT> yPermute; // Reused.
  vector<PredictorT> census;
  vector<size_t> confusion; // Confusion matrix; saved.
  vector<double> misprediction; // Mispredction, by merged category; saved.
  double oobPredict; // Out-of-bag error:  % mispredicted rows.
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

  PredictCtg(const class Forest* forest_,
	     const class Sampler* sampler_,
	     struct RLEFrame* rleFrame_,
	     const vector<PredictorT>& yTest_,
	     unsigned int nPredict_,
	     bool doProb);

  //  ~PredictCtg(); // Forward declaration:  not specified default;


  /**
     @brief Description given in virtual declartion.
   */
  void scoreSeq(size_t rowStart,
		size_t rowEnd);


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


  const vector<vector<double>>& getMispredPermuted() const {
    return mispredPermute;
  }


  double getOOBError() const {
    return oobPredict;
  }    


  const vector<double>& getOOBErrorPermuted() const {
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
  const vector<double>& getProb() const;

  
  /**
     @brief Dumps and categorical-specific contents.
   */
  void dump() const;
};


#endif
