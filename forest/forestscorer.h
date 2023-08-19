// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictscorer.h

   @brief Scorer for prediction.

   @author Mark Seligman
 */

#ifndef FOREST_FORESTSCORER_H
#define FOREST_FORESTSCORER_H


#include "typeparam.h"
#include "predict.h"

#include <functional>
#include <map>
#include <vector>


struct ForestScore {
  unsigned int nEst; ///< # participating trees.
  union Score {
    double num;
    CtgT ctg;
  } score;


  /**
     @brief Constructs categorical score.
   */
  ForestScore(unsigned int nEst_,
	      CtgT ctg) :
    nEst(nEst_) {
    score.ctg = ctg;
  }

  
  /**
     @brief Constructs numerical score.
   */
  ForestScore(unsigned int nEst_,
	      double num) :
    nEst(nEst_) {
    score.num = num;
  }
};



class ForestScorer {
  static map<const string, function<ForestScore(ForestScorer*, class Predict*, size_t)>> scorerTable;

  const double nu; ///< Learning rate, possibly vector if adaptive.
  const double baseScore; ///< Pre-training score of sampled root.
  const CtgT nCtg; ///< Categoricity if classifcation, else zero.
  const double defaultPrediction; ///< Obtained from full response.
  const function<ForestScore(ForestScorer*, class Predict*, size_t)> scorer;

  // Classification only:
  vector<unsigned int> census; ///< # trees per category, per observation.
  unique_ptr<class CtgProb> ctgProb; ///< probability, per category.

  // Regression only.
  unique_ptr<class Quant> quant; ///< Independent trees only.

  ForestScore scoreObs(class Predict* predict, size_t obsIdx) {
    return scorer(this, predict, obsIdx);
  }
  

  CtgT argMaxJitter(const unsigned int censusRow[],
		    const vector<double>& ctgJitter) const;


 public:

  ForestScorer(const struct ScoreDesc* scoreDesc,
	       const class ResponseReg* response,
 	       const class Forest* forest,
	       const class Leaf* leaf,
	       const class PredictReg* predict,
	       vector<double> quantile);


  ForestScorer(const struct ScoreDesc* scoreDesc,
	       const class ResponseCtg* response,
	       size_t nObs,
	       bool doPorb);


  unsigned int scoreObs(class PredictReg* predict, size_t obsIdx, vector<double>& yTarg);

  unsigned int scoreObs(class PredictCtg* predict, size_t obsIdx, vector<CtgT>& yTarg);


  /**
     @brief Derives a mean prediction value for an observation.

     Assumes independent trees.
   */
  ForestScore predictMean(class Predict* predict,
		     size_t obsIdx) const;

  
  /**
     @brief Derives a summation.

     @return sum of predicted responses plus rootScore.
   */
  ForestScore predictSum(class Predict* predict,
		    size_t obsIdx) const;


  /**
     @brief Probability of second element:  logistic of sum.

     @return more likely category, of two.
   */
  ForestScore predictLogistic(class Predict* predict,
			      size_t obsIdx);
  
  
  ForestScore predictPlurality(class Predict* predict,
			       size_t obsIdx);

  
  const vector<unsigned int>& getCensus() const {
    return census;
  }


  vector<unsigned int>* getCensusBase() {
    return &census;
  }
  

  const vector<double>& getProb() const;


  const vector<double>& getQPred() const;

  
  const vector<double>& getQEst() const;
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
  CtgProb(size_t nObs,
	  const class ResponseCtg* response,
	  bool doProb);

  
  /**
     @brief Predicts probabilities across all trees.

     @param row is the row number.
   */
  void predictRow(size_t row,
		  const unsigned int censusRow[]);


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


#endif
