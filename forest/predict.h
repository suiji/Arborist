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

#include "prediction.h"
#include "predictframe.h"
#include "block.h"
#include "typeparam.h"
#include "bv.h"

#include <vector>
#include <algorithm>


/**
   @brief Regression-specific prediction summary.
 */
struct SummaryReg {
  unique_ptr<ForestPredictionReg> prediction;
  unique_ptr<TestReg> test;
  vector<vector<unique_ptr<TestReg>>> permutationTest;

  SummaryReg(const class Sampler* sampler,
	     const class Predict* predict,
	     class Forest* forest);


  void build(class Predict* predict,
	     const class Sampler* sampler,
	     const vector<double>& yTest);


  static vector<vector<unique_ptr<TestReg>>> permute(const class Predict* predict,
						     const class Sampler* sampler,
						     const vector<double>& yTest);

  size_t getNObs() const {
    return prediction->getNObs();
  }


  /**
     @return handle to cached index vector.
   */
  const vector<size_t>& getIndices() const {
    return prediction->idxFinal;
  }

  
  const vector<double>& getYPred() const;


  /**
     @brief Passes through to TestReg.

     @return SSE if testing, else zero.
   */
  double getSSE() const;


  /**
     @brief Passes through to TestReg.
     
     @return if testing absolute error else zero.
   */
  double getSAE() const;


  /**
     @return vector of estimated quantile means.
   */
  const vector<double>& getQEst() const;


  /**
     @return vector quantile predictions.
  */
  const vector<double>& getQPred() const;


  vector<vector<double>> getSSEPermuted() const;


  vector<vector<double>> getSAEPermuted() const;
};


/**
   @brief Classification-specific prediction summary.
 */
struct SummaryCtg {
  CtgT nCtgTrain;  // Census, prob only accessible by training categories.
  unique_ptr<ForestPredictionCtg> prediction;
  unique_ptr<TestCtg> test;
  vector<vector<unique_ptr<TestCtg>>> permutationTest;

  SummaryCtg(const class Sampler* sampler,
	     const class Predict* predict,
	     class Forest* forest);

  
  void build(class Predict* predict,
	     const class Sampler* sampler,
	     const vector<unsigned int>& yTest);


  static vector<vector<unique_ptr<TestCtg>>> permute(const class Predict* predict,
						     const class Sampler* sampler,
						     const vector<unsigned int>& yTest);


  /**
     @brief Derives an index into a matrix having stride equal to the
     number of training categories.
     
     @param row is the row coordinate.

     @return derived strided index.
   */
  size_t ctgIdx(size_t row, PredictorT ctg = 0) const {
    return row * nCtgTrain + ctg;
  }


  size_t getNObs() const {
    return prediction->getNObs();
  }

  
  /**
     @return handle to cached index vector.
   */
  const vector<size_t>& getIndices() const {
    return prediction->idxFinal;
  }


  const vector<CtgT>& getYPred() const;

  const vector<size_t>& getConfusion() const;

  
  const vector<double>& getMisprediction() const;

  double getOOBError() const;


  /**
     @brief Passes through to scorer.
   */
  const vector<unsigned int>& getCensus() const;


  /**
     @brief Passes through to scorer.

     @return reference to per-category probability predictions.
   */
  const vector<double>& getProb() const;


  vector<vector<vector<double>>> getMispredPermuted() const;


  vector<vector<double>> getOOBErrorPermuted() const;
};


/**
   @brief Invokes virtual prediction methods.
 */
class Predict {
protected:
  static const size_t obsChunk; ///< Observation block dimension.
  static const unsigned int seqChunk;  ///< Effort to minimize false sharing.

  const unique_ptr<class BitMatrix> bag; ///< Nonnull iff bagging.
  unique_ptr<struct RLEFrame> rleFrame;
  const size_t nObs; ///< # observations under prediction.

  // Prediction state:
  unsigned int nTree; ///< Initialized by Forest under prediction.
  IndexT noNode; ///< Initialized by Forest under prediction.
  PredictFrame trFrame; ///< Initialized by RLEFrame, reset per block.
  size_t blockStart; ///< Index of observation heading current block.
  vector<IndexT> idxFinal; ///< Final walk index, typically terminal.

  void predictBlock(ForestPrediction* prediction);


  void predictObs(class ForestPrediction* prediction,
		  size_t span);


  void resetIndices();


  void walkTree(const PredictFrame& frame,
		size_t obsStart,
		size_t obsEnd);


  void setFinalIdx(size_t obsIdx, unsigned int tIdx, IndexT finalIdx) {
    idxFinal[nTree * (obsIdx - blockStart) + tIdx] = finalIdx;
  }


public:

  static bool bagging; ///< True iff bagging.

  
  static unsigned int nPermute; ///< # times to permute each predictor.


  class Forest* forest; ///< Set at prediction.


  /**
     @brief Static initializations per invocation.
   */
  static void init(bool bagging_,
		   unsigned int nPermute_);

  
  /**
     @brief Resets static values.
   */  
  static void deInit();

  
  Predict(const class Sampler* sampler,
	  unique_ptr<struct RLEFrame> rleFrame_);


  virtual ~Predict() = default;


  static unique_ptr<class PredictCtg> makeCtg(const class Sampler* sampler,
					      unique_ptr<struct RLEFrame>);


  static unique_ptr<class PredictReg> makeReg(const class Sampler* sampler,
					      unique_ptr<struct RLEFrame>);


  virtual unique_ptr<SummaryReg> predictReg(const class Sampler* sampler,
					    class Forest* forest,
					    const vector<double>& yTest) {
    return nullptr;
  }


  virtual unique_ptr<SummaryCtg> predictCtg(const class Sampler* sampler,
					    class Forest* forest,
					    const vector<unsigned int>& yTest) {
    return nullptr;
  }


  struct RLEFrame* getFrame() const {
    return rleFrame.get();
  }


  unsigned int getNTree() const {
    return nTree;
  }


  size_t getNObs() const {
    return nObs;
  }


  bool isNodeIdx(size_t obsIdx,
		 unsigned int tIdx,
		 double& score) const;


  bool isLeafIdx(size_t obsIdx,
		 unsigned int tIdx,
		 IndexT& leafIdx) const;


  /**
     @param[out] nodeIdx is the final index of the tree walk.

     @return true iff final index is a valid node.
  */
  bool getFinalIdx(size_t obsIdx, unsigned int tIdx, IndexT& nodeIdx) const {
    nodeIdx = idxFinal[nTree * (obsIdx - blockStart) + tIdx];
    return nodeIdx != noNode;
  }


  /**
     @brief Determines whether a given forest coordinate is bagged.

     @param tIdx is the tree index.

     @param row is the row index.

     @return true iff bagging and the coordinate bit is set.
   */
  inline bool isBagged(unsigned int tIdx, size_t row) const {
    return bagging && bag->testBit(tIdx, row);
  }

  
  static bool permutes() {
    return nPermute > 0;
  }


  void predict(ForestPrediction* prediction);

  
  /**
     @brief Computes Meinshausen's weight vectors for a block of predictions.

     @param nPredict is tne number of predictions to weight.

     @param finalIdx is a block of nPredict x nTree prediction indices.
     
     @return prediction-wide vector of response weights.
   */
  static vector<double> forestWeight(const class Forest* forest,
					     const class Sampler* sampler,
					     size_t nPredict,
				     const double finalIdx[]);


  static vector<vector<struct IdCount>> obsCounts(const class Forest* forest,
						  const class Sampler* sampler,
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


struct PredictReg : public Predict {

  PredictReg(const class Sampler* sampler,
	     unique_ptr<struct RLEFrame> rleFrame_);


  ~PredictReg() = default;

  unique_ptr<SummaryReg> predictReg(const class Sampler* sampler,
				    class Forest* forest,
				    const vector<double>& yTest);
};


struct PredictCtg : public Predict {

  PredictCtg(const class Sampler* sampler,
	     unique_ptr<struct RLEFrame> rleFrame_);

  
  ~PredictCtg() = default;


  unique_ptr<SummaryCtg> predictCtg(const class Sampler* sampler,
				    class Forest* forest,
				    const vector<unsigned int>& yTest);


  /**
     @brief Dumps and categorical-specific contents.
   */
  void dump() const;
};


#endif
