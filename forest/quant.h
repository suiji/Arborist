// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file quant.h

   @brief Data structures and methods for predicting and writing quantiles.

   @author Mark Seligman

 */

#ifndef FOREST_QUANT_H
#define FOREST_QUANT_H

#include "typeparam.h"
#include "prediction.h"
#include "valrank.h"
#include "leaf.h"

#include <vector>

class Predict;
class Sampler;
struct ForestPredictReg;

/**
 @brief Quantile signature.
*/
class Quant {
  static const unsigned int binSize; // # slots to track.
  static vector<double> quantile; ///< quantile values over which to predict.
  const Leaf& leaf;
  const bool empty; // if so, leave vectors empty and bail.
  const unsigned int qCount; ///< caches quantile size for quick reference.
  const bool trapAndBail; ///< Whether nonterminal exit permitted.
  const vector<vector<IndexRange>> leafDom;
  const RankedObs<double> valRank;
  const vector<vector<vector<RankCount>>> rankCount; // forest-wide, by sample.
  const unsigned int rankScale; // log2 of scaling factor.
  const vector<double> binMean;
  vector<double> qPred; // predicted quantiles.
  vector<double> qEst; // quantile of response estimates.
  

  /**
     @brief Computes a bin offset for a given rank.

     @param rank is the rank in question.

     @return bin offset.
   */
  unsigned int binRank(unsigned int rank) const {
    return rank >> rankScale;
  }


  /**
     @brief Determines scaling factor for training response.

     @return power-of-two divisor for training response length.
   */
  unsigned int binScale() const;


  /**
n     @brief Bins response means.

     @param valRank contains the ranked response/row pairs.

     @param rankScale is the bin scaling factor.

     @return binned vector of response means.
   */
  vector<double> binMeans(const RankedObs<double>& valRank) const;

  
  /**
     @brief Writes quantile values for a row of predictions.

     @param sCount is a bin of ranked sample counts.

     @param threshold is the sample count threshold for a given quantile.

     @param yPred is the predicted response for the current row.

     @param[out] qRow[] outputs the derived quantiles.
   */
  void quantSamples(const ForestPredictionReg* prediction,
		    const vector<IndexT>& sCount,
		    const vector<double>& threshold,
		    IndexT totSample,
		    size_t obsIdx);
  

  /**
     @brief Accumulates the ranks assocated with predicted leaf.

     @param tIdx is a tree index.

     @param leafIdx is a tree-relative leaf index.

     @param[in,out] sCount counts the number of samples at a (binned) rank.

     @return count of samples subsumed by leaf.
  */
  IndexT sampleLeaf(unsigned int tIdx,
		    IndexT leafIdx,
		    vector<IndexT>& scountBin) const;


public:
  /**
     @brief Constructor for invocation from within core.

     Parameters mirror simililarly-named members.
   */
  Quant(const Sampler* sampler,
	const Predict* predict,
	bool reportAuxiliary);


  ~Quant() = default;
  

  static void init(vector<double> quantile_);


  static void deInit();


  /**
     @brief Determines whether to bail on quantile estimation.
   */
  bool isEmpty() const {
    return empty;
  };


  /**
     @brief Getter for number of quantiles.

     @return qCount value.
   */
  unsigned int getNQuant() const {
    return qCount;
  }


  /**
     @brief Accessor for predicted quantiles.

     @return vector of quantile predictions.
   */
  const vector<double>& getQPred() const {
    return qPred;
  }
  
  
  /**
     @brief Accessor for estimand quantiles.

     @return pointer to base of estimand quantiles.
   */
  const vector<double>& getQEst() const {
    return qEst;
  }
  
  
  /**
     @brief Writes the quantile values for a given row.

     @param row is the row over which to build prediction quantiles.
  */
  void predictRow(const Predict* predict,
		  const ForestPredictionReg* prediction,
		  size_t obsIdx);
};

#endif
