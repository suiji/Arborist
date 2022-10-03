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
#include "decnode.h"

#include <vector>
#include <algorithm>


/**
   @brief Categorical probabilities associated with indivdual leaves.

   Intimately accesses the raw jagged array it contains.
 */
class CtgProb {
  const PredictorT nCtg; // Training cardinality.
  const vector<double> probDefault; // Forest-wide default probability.
  vector<double> probs; // Per-row probabilties.

  
  /**
     @brief Copies default probability vector into argument.

     @param[out] probPredict outputs the default category probabilities.
   */
  void applyDefault(double probPredict[]) const;
  

public:
  CtgProb(const class Predict* predict,
	  const class ResponseCtg* response,
	  bool doProb);

  
  /**
     @brief Predicts probabilities across all trees.

     @param row is the row number.

     @param[out] probRow outputs the per-category probabilities.
   */
  void predictRow(const class Predict* predict,
		  size_t row,
		  PredictorT* ctgRow);

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


/**
   @brief Walks the decision forest for each row in a block, collecting
   predictions.
 */
class Predict {
protected:
  static const size_t scoreChunk; // Score block dimension.
  static const unsigned int seqChunk;  // Effort to minimize false sharing.

  const bool trapUnobserved; ///< Whether to trap values not observed during training.
  const class Sampler* sampler; ///< In-bag representation.
  const vector<vector<DecNode>> decNode; ///< Forest-wide decision nodes.
  const vector<unique_ptr<BV>>& factorBits; ///< Splitting bits.
  const vector<unique_ptr<BV>>& bitsObserved; ///< Bits participating in split.
  const bool testing; ///< Whether to compare prediction with test vector.
  const unsigned int nPermute; ///< # times to permute each predictor.

  vector<IndexT> predictLeaves; ///< Tree-relative leaf indices.

  size_t blockStart; ///< Stripmine bound.
  vector<IndexT> accumNEst;

  size_t nEst; ///< Total number of estimands.
  
  
  /**
     @brief Assigns a relative node index at the prediction coordinates passed.

     @param row is the row number.

     @param tc is the index of the current tree.

     @param idx is an absolute node index.
   */
  inline void predictLeaf(size_t row,
                          unsigned int tIdx,
                          IndexT idx) {
    predictLeaves[nTree * (row - blockStart) + tIdx] = idx;
  }


  /**
     @brief Performs prediction on separately-permuted predictor columns.

     @param permute is the number of times to permute each predictor.
   */
  void predictPermute(struct RLEFrame* rleFrame);
  

  /**
     @brief Drives prediction strip-mining and residual.
   */
  void blocks(const struct RLEFrame* rleFrame);
  
  
  /**
     @brief Strip-mines prediction by fixed-size blocks.

     @param[in, out] trIdx caches RLE index accessed by a predictor.
   */
  size_t predictBlock(const struct RLEFrame* rleFrame,
		      size_t row,
		      size_t extent,
		      vector<size_t>& trIdx);

  /**
     @brief Transposes typed blocks, prediction-style.

     @param[in,out] idxTr is the most-recently accessed RLE index, by predictor.

     @param rowStart is the starting source row.

     @param rowExtent is the total number of rows to transpose.
   */
  void transpose(const RLEFrame* rleFrame,
		 vector<size_t>& idxTr,
		 size_t rowStart,
		 size_t rowExtent);


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
  void predictBlock(size_t span);


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
  const PredictorT nPredNum;
  const PredictorT nPredFac;
  const size_t nRow;
  const unsigned int nTree; // # trees used in training.
  const IndexT noNode; // Inattainable leaf index value.

  /**
     @brief Aliases a row-prediction method tailored for the frame's
     block structure.
   */
  void (Predict::* walkTree)(size_t);
  void (Predict::* getWalker())(size_t);

  vector<CtgT> trFac; // OTF transposed factor observations.
  vector<double> trNum; // OTF transposed numeric observations.

  Predict(const class Forest* forest_,
	  const class Sampler* sampler_,
	  struct RLEFrame* rleFrame_,
	  bool testing_,
	  PredictorT nPredict_,
	  bool trapUnobserved_);

  virtual ~Predict() = default;

  /**
     @brief Main entry from bridge.

     Distributed prediction will require start and extent parameters.
   */
  void predict(struct RLEFrame* rleFrame);


  /**
     @brief Indicates whether to exit tree prematurely when an unrecognized
     obervation is encountered.
   */
  bool trapAndBail() const {
    return trapUnobserved;
  }
  

  const class Sampler* getSampler() const {
    return sampler;
  }


  /**
     @brief Computes block-relative position for a predictor.

     @param[out] thisIsFactor outputs true iff predictor is factor-valued.

     @return block-relative index.
   */
  inline unsigned int getIdx(PredictorT predIdx, bool &predIsFactor) const {
    predIsFactor = isFactor(predIdx);
    return predIsFactor ? predIdx - nPredNum : predIdx;
  }

  
  inline bool isFactor(PredictorT predIdx) const {
    return predIdx >= nPredNum;
  }


  size_t getNRow() const {
    return nRow;
  }
  

  inline unsigned int getNTree() const {
    return nTree;
  }


  /**
     @param[out] termIdx is the node index of prediction.

     @return true iff the predicted terminal index references a leaf.
   */
  bool isLeafIdx(size_t row,
		 unsigned int tIdx,
		 IndexT& leafIdx) const;


  /**
     @brief As above, but outputs leaf score.
   */
  inline bool isLeafIdx(size_t row,
			unsigned int tIdx,
			double& score) const {
    IndexT termIdx = predictLeaves[nTree * (row - blockStart) + tIdx];
    if (termIdx != noNode) {
      score = scoreBlock[tIdx][termIdx];
      return true;
    }
    else {
      return false;
    }
    // Non-bagging scenarios should always see a leaf.
    //    if (!bagging) assert(termIdx != noNode);
  }

  
  /**
     @brief As above, but outputs node index.
   */
  bool isNodeIdx(size_t row,
		 unsigned int tIdx,
		 IndexT& nodeIdx) const {
    nodeIdx = predictLeaves[nTree * (row - blockStart) + tIdx];
    return nodeIdx != noNode;
  }

  
  /**
     @brief Computes pointer to base of row of numeric values.

     @param row is the row number.

     @return base address for numeric values at row.
  */
  const double* baseNum(size_t row) const {
    return &trNum[(row - blockStart) * nPredNum];
  }


  /**
     @brief As above, but factor varlues.

     @return row is the row number.
   */
  const CtgT* baseFac(size_t row) const {
    return &trFac[(row - blockStart) * nPredFac];
  }

  
  /**
     @brief Prediction of single row with mixed predictor types.

     @param row is the absolute row of data over which a prediction is made.

     @param blockRow is the row's block-relative row index.

     @return index of leaf predicted.
  */
  void rowMixed(unsigned int tIdx,
		const double* rowNT,
		const CtgT* rowFT,
		size_t row);

  
  /**
     @brief Prediction over a single row with factor-valued predictors only.

     Parameters as in mixed case, above.
  */
  void rowFac(unsigned int tIdx,
	      const CtgT* rowT,
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
  const class ResponseReg* response;
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
	     const struct Leaf* leaf_,
	     struct RLEFrame* rleFrame_,
	     const vector<double>& yTest_,
	     PredictorT nPredict_,
	     const vector<double>& quantile,
	     bool trapUnobserved_);


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
  const class ResponseCtg* response;
  const vector<PredictorT> yTest;
  vector<PredictorT> yPred;
  const PredictorT nCtgTrain; // Cardiality of training response.
  const PredictorT nCtgMerged; // Cardinality of merged test response.
  unique_ptr<CtgProb> ctgProb; // Class prediction probabilities.

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
	     PredictorT nPredict_,
	     bool doProb,
	     bool trapUnobserved_);


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
  const vector<double>& getProb() const {
    return ctgProb->getProb();
  }


  /**
     @brief Dumps and categorical-specific contents.
   */
  void dump() const;
};


#endif
