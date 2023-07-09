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
   @brief Encapsulates options specifying prediction.   
 */
struct PredictOption {
  unsigned int nPermute; ///< 0/1:  # permutations to perform.
  bool trapUnobserved; ///< exits at nonterminal on unobserved value.
  bool indexing; ///< reports index of prediction, typically terminal.

  PredictOption(unsigned int nPermute_,
		bool indexing_,
		bool trapUnobserved_) :
    nPermute(nPermute_),
    trapUnobserved(trapUnobserved_),
    indexing(indexing_) {
  }
};


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
  static const size_t scoreChunk; ///< Score block dimension.
  static const unsigned int seqChunk;  ///< Effort to minimize false sharing.

  const bool trapUnobserved; ///< Whether to trap values not observed during training.
  const class Sampler* sampler; ///< In-bag representation.
  const vector<vector<DecNode>> decNode; ///< Forest-wide decision nodes.
  const vector<unique_ptr<BV>>& factorBits; ///< Splitting bits.
  const vector<unique_ptr<BV>>& bitsObserved; ///< Bits participating in split.
  const bool testing; ///< Whether to compare prediction with test vector.
  const unsigned int nPermute; ///< # times to permute each predictor.

  vector<IndexT> idxFinal; ///< Final walk index, typically terminal.

  size_t blockStart; ///< Stripmine bound.
  vector<IndexT> accumNEst;

  size_t nEst; ///< Total number of estimands.
  
  
  /**
     @brief Assigns a relative node index at the prediction coordinates passed.

     @param row is the row number.

     @param tc is the index of the current tree.
   */
  inline void predictLeaf(unsigned int tIdx,
			  size_t obsIdx) {
    idxFinal[nTree * (obsIdx - blockStart) + tIdx] = (this->*Predict::walkObs)(tIdx, obsIdx);
  }


  /**
     @brief Performs prediction on separately-permuted predictor columns.

     @param permute is the number of times to permute each predictor.
   */
  void predictPermute(struct RLEFrame* rleFrame);
  

  /**
     @brief Drives prediction strip-mining and residual.
xon   */
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
     @brief Predicts along each tree.

     @param rowStart is the absolute starting row for the block.
  */
  void walkTree(size_t rowStart);


  /**
     @brief Predicts a block of observations in parallel.
   */
  void predictObs(size_t span);


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
  IndexT (Predict::* walkObs)(unsigned int, size_t);
  IndexT (Predict::* getObsWalker())(unsigned int, size_t);
  
  vector<CtgT> trFac; ///< OTF transposed factor observations.
  vector<double> trNum; ///< OTF transposed numeric observations.
  vector<size_t> indices; ///< final index of traversal.
  
  Predict(const class Forest* forest_,
	  const class Sampler* sampler_,
	  struct RLEFrame* rleFrame_,
	  bool testing_,
	  const PredictOption& option);


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

  
  /**
     @return handle to cached index vector.
   */
  const vector<size_t>& getIndices() const {
    return indices;
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
     @brief As above, but outputs node score.
   */
  inline bool isNodeIdx(size_t row,
			unsigned int tIdx,
			double& score) const {
    IndexT termIdx = idxFinal[nTree * (row - blockStart) + tIdx];
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
    nodeIdx = idxFinal[nTree * (row - blockStart) + tIdx];
    return nodeIdx != noNode;
  }

  
  /**
     @brief Computes pointer to base of row of numeric values.

     @param row is the row number.

     @return base address for numeric values at row.
  */
  inline const double* baseNum(size_t row) const {
    return &trNum[(row - blockStart) * nPredNum];
  }


  /**
     @brief As above, but factor varlues.

     @return row is the row number.
   */
  inline const CtgT* baseFac(size_t row) const {
    return &trFac[(row - blockStart) * nPredFac];
  }

  
  /**
     @brief Prediction over an observation with mixed predictor types.

     @param obsIdx is the absolute row of data over which a prediction is made.

     @return index of node predicted.
  */
  IndexT obsMixed(unsigned int tIdx,
		  size_t obsIdx);


  /**
     @brief Prediction over an observation with factor-valued predictors only.

     Parameters as in mixed case, above.
  */
  IndexT obsFac(unsigned int tIdx,
		size_t obsIdx);


  /**
     @brief Prediction over an observation with numeric-valued predictors only.

     Parameters as in mixed case, above.
   */
  IndexT obsNum(unsigned int tIdx,
		size_t obsIdx);


  /**
     @brief Computes Meinshausen's weight vectors for a block of predictions.

     @param nPredict is tne number of predictions to weight.

     @param finalIdx is a block of nPredict x nTree prediction indices.
     
     @return prediction-wide vector of response weights.
   */
  static vector<double> forestWeight(const class Forest* forest,
					     const class Sampler* sampler,
					     const struct Leaf* leaf,
					     size_t nPredict,
					     const double finalIdx[],
					     unsigned int nThread);


  static vector<vector<struct IdCount>> obsCounts(const class Forest* forest,
						  const class Sampler* sampler,
						  const struct Leaf* leaf,
						  unsigned int tIdx);


  static void weighNode(const class Forest* forest,
			const double treeIdx[],
			const vector<vector<struct IdCount>>& nodeCount,
			vector<vector<double>>& obsWeight);


  /**
     @brief Normalizes each weight vector passed.

     @return vector of normalized weight vectors.
   */
  static vector<double> normalizeWeight(const class Sampler* sampler,
					const vector<vector<double>>& obsWeight);
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
  
  void testObs(size_t row);

  unsigned int scoreObs(size_t row);

public:
  PredictReg(const class Forest* forest_,
	     const class Sampler* sampler_,
	     const struct Leaf* leaf_,
	     struct RLEFrame* rleFrame_,
	     const vector<double>& yTest_,
	     const PredictOption& option,
	     const vector<double>& quantile);


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

  void testObs(size_t row);

  void scoreObs(size_t row);

  
public:

  PredictCtg(const class Forest* forest_,
	     const class Sampler* sampler_,
	     struct RLEFrame* rleFrame_,
	     const vector<PredictorT>& yTest_,
	     const PredictOption& option,
	     bool doProb);


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
