// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file prediction.h

   @brief Type-based structures recording predictions.

   @author Mark Seligman
 */


#ifndef FOREST_PREDICTION_H
#define FOREST_PREDICTION_H

#include "typeparam.h"

#include <functional>
#include <map>
#include <vector>

struct ScoreCount {
  unsigned int nEst; ///< # participating trees.
  union Score {
    double num;
    CtgT ctg;
  } score;


  /**
     @brief Constructs categorical score.
   */
  ScoreCount(unsigned int nEst_,
	      CtgT ctg) :
    nEst(nEst_) {
    score.ctg = ctg;
  }

  
  /**
     @brief Constructs numerical score.
   */
  ScoreCount(unsigned int nEst_,
	      double num) :
    nEst(nEst_) {
    score.num = num;
  }
};


template<class ResponseType>
struct Prediction {
  vector<ResponseType> value; ///< The predicted value.
  vector<size_t> idxFinal; ///< Final index of tree walk.

  
  Prediction(size_t nObs) :
   value(vector<ResponseType>(nObs)) {
  }


  size_t getNObs() const {
    return value.size();
  }

  
  void setScore(size_t obsIdx,  ResponseType val) {
    value[obsIdx] = val;
  }


  void setIndex(size_t obsIdx,  ResponseType val, size_t idx) {
    idxFinal[obsIdx] = idx;
  }
};


struct ForestPrediction {
  static bool reportIndices;
  
  const double baseScore;
  const double nu;

  vector<size_t> idxFinal; ///< Final index of tree walk; auxilliary.
  
  ForestPrediction(const class Predict* predict,
		   const struct ScoreDesc* scoreDesc);

  virtual ~ForestPrediction() = default;

  static void init(bool doProb);


  static void deInit();


  /**
     @brief Caches final tree-walk indices.
   */
  void cacheIndices(vector<IndexT>& indices,
		    size_t span,
		    size_t obsStart);


  virtual void callScorer(const class Predict*, size_t obsStart, size_t obsEnd) = 0;
};



struct ForestPredictionCtg : public ForestPrediction {
  static map<const string, function<void(ForestPredictionCtg*, const class Predict*, size_t)>> scorerTable;

  const function<void(ForestPredictionCtg*, const class Predict*, size_t)> scorer;
  const CtgT nCtg;
  Prediction<CtgT> prediction;
  const CtgT defaultPrediction;

  void callScorer(const class Predict* predict, size_t obsStart, size_t obsEnd) {
    for (size_t obsIdx = obsStart; obsIdx != obsEnd; obsIdx++) {
      scorer(this, predict, obsIdx);
    }
  }

  vector<CtgT> census; ///< # trees per category, per observation.
  unique_ptr<class CtgProb> ctgProb; ///< probability, per category.

  ForestPredictionCtg(const struct ScoreDesc* scoreDesc,
		      const class Sampler* sampler,
		      const class Predict* predict,
		      bool reportAuxiliary);

  
  ~ForestPredictionCtg() = default;

  
  ScoreCount predictLogOdds(const class Predict* predict,
			    size_t obsIdx) const;


  void predictLogistic(const class Predict* predict,
		       size_t obsIdx);


  void predictPlurality(const class Predict* predict,
			size_t obsIdx);


  CtgT argMaxJitter(const vector<double>& numVec) const;


  size_t getNObs() const {
    return prediction.getNObs();
  }

  
  void setScore(size_t obsIdx, ScoreCount score);


  unique_ptr<struct TestCtg> test(const vector<CtgT>& yTest) const;

  
  const vector<double>& getProb() const;
};


struct ForestPredictionReg : public ForestPrediction {
  static map<const string, function<void(ForestPredictionReg*, const class Predict*, size_t)>> scorerTable;

  const function<void(ForestPredictionReg*, const class Predict*, size_t)> scorer;
  Prediction<double> prediction;
  const double defaultPrediction;

  void callScorer(const class Predict* predict, size_t obsStart, size_t obsEnd) {
    for (size_t obsIdx = obsStart; obsIdx != obsEnd; obsIdx++) {
      scorer(this, predict, obsIdx);
    }
  }

  unique_ptr<class Quant> quant; ///< Independent trees only.


  ForestPredictionReg(const struct ScoreDesc* scoreDesc,
		      const class Sampler* sampler,
		      const class Predict* predict,
		      bool reportAuxiliary);

  
  ~ForestPredictionReg() = default;

  
  void predictMean(const class Predict* predict,
		   size_t obsIdx);

  
  void predictSum(const class Predict* predict,
		  size_t obsIdx);


  unique_ptr<struct TestReg> test(const vector<double>& yTest) const;

  
  double getValue(size_t obsIdx) const {
    return prediction.value[obsIdx];
  }
  

  size_t getNObs() const {
    return prediction.getNObs();
  }

  
  void setScore(const class Predict* predict,
		size_t obsIdx,
		ScoreCount score);


  const vector<double>& getQPred() const;
  const vector<double>& getQEst() const;
};


/**
   @brief Categorical probabilities associated with indivdual leaves.
 */
class CtgProb {
  static bool reportProbabilities; ///< Whether to track.
  
  const PredictorT nCtg; ///< Training cardinality.
  const vector<double> probDefault; ///< Forest-wide default probability.
  vector<double> probs; ///< Per-observation probabilties.

  
  /**
     @brief Copies default probability vector into argument.

     @param[out] probPredict outputs the default category probabilities.
   */
  void applyDefault(double probPredict[]) const;
  

public:
  
  /**
     @param reportAuxiliary is false iff caller declines to record.
   */
  CtgProb(const class Sampler* sampler,
	  size_t nObs,
	  bool reportAuxiliary);


  static void init(bool doProb);


  static void deInit();

  
  /**
     @brief Predicts probabilities across all trees.

     @param row is the row number.
   */
  void predictRow(size_t row,
		  const vector<double>& numVec,
		  unsigned int nEst);


  /**
     @brief Binary classification with know probability.

     @param p1 is the probability of category 1;
   */
  void assignBinary(size_t obsIdx,
		    double p1);

  
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


struct TestReg {
  double SSE;
  double absError;

  TestReg() = default;

  
  TestReg(double SSE_,
	  double absError_) :
    SSE(SSE_),
    absError(absError_) {
  }


  static vector<vector<double>> getSSEPermuted(const vector<vector<unique_ptr<TestReg>>>&);


  static vector<vector<double>> getSAEPermuted(const vector<vector<unique_ptr<TestReg>>>&);
};


struct TestCtg {
  CtgT nCtgTrain;
  CtgT nCtgMerged;
  vector<size_t> confusion;
  vector<double> misprediction;
  double oobErr; /// Out-of-bag error:  % mispredicted observations.

  TestCtg() = default;  

  TestCtg(CtgT nCtgTrain_,
	  CtgT nCtgMerged_) :
    nCtgTrain(nCtgTrain_),
    nCtgMerged(nCtgMerged_),
    confusion(vector<size_t>(nCtgTrain * nCtgMerged)),
    misprediction(vector<double>(nCtgMerged)),
    oobErr(0.0) {
  }


  void buildConfusion(const vector<CtgT>& yTest,
		      const vector<CtgT>& yPred);


  void setMisprediction(size_t nObs);


  static vector<vector<vector<double>>> getMispredPermuted(const vector<vector<unique_ptr<TestCtg>>>&);


  static vector<vector<double>> getOOBErrorPermuted(const vector<vector<unique_ptr<TestCtg>>>&);
};


#endif
