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

#ifndef RF_QUANT_H
#define RF_QUANT_H

#include "typeparam.h"
#include "valrank.h"

#include <vector>


/**
 @brief Quantile signature.
*/
class Quant {
  static const unsigned int binSize; // # slots to track.
  const class LeafFrameReg *leafReg; // Summary of trained terminal nodes.
  const class BitMatrix* baggedRows; // In-bag summary.
  ValRank<double> valRank;
  const vector<struct RankCount> rankCount; // forest-wide, by sample.
  const vector<double> quantile; // quantile values over which to predict.
  const unsigned int qCount; // caches quantile size for quick reference.
  vector<double> qPred; // predicted quantiles.
  vector<double> qEst; // quantile of response estimates.
  unsigned int rankScale; // log2 of scaling factor.
  const vector<double> binMean;

  /**
     @brief Computes a bin offset for a given rank.

     @param rank is the rank in question.

     @return bin offset.
   */
  inline unsigned int binRank(unsigned int rank) const {
    return rank >> rankScale;
  }


  /**
     @brief Determines scaling factor for training response.

     @return power-of-two divisor for training response length.
   */
  unsigned int binScale() const;


  /**
     @brief Bins response means.

     @param valRank contains the ranked response/row pairs.

     @param rankScale is the bin scaling factor.

     @return binned vector of response means.
   */
  vector<double> binMeans(const ValRank<double>& valRank,
                          unsigned int rankScale);

  
  /**
     @brief Writes the quantile values for a given row.

     @param rowBlock is the block-relative row index.

     @param[out] qRow[] outputs the 'qCount' quantile values.
  */
  void predictRow(const class PredictFrame* frame,
                  unsigned int rowBlock,
                  double yPred,
                  double qRow[],
                  double* qEst);


  /**
     @brief Writes quantile values for a row of predictions.

     @param sCount is a bin of ranked sample counts.

     @param threshold is the sample count threshold for a given quantile.

     @param yPred is the predicted response for the current row.

     @param[out] qRow[] outputs the derived quantiles.
   */
  IndexT quantSamples(const vector<PredictorT>& sCount,
                      const vector<double> threshold,
                      double yPred,
                      double qRow[]) const;

  /**
     @brief Accumulates the ranks assocated with predicted leaf.

     @param tIdx is a tree index.

     @param leafIdx is a tree-relative leaf index.

     @param[in,out] sCount counts the number of samples at a (binned) rank.

     @return count of samples subsumed by leaf.
  */
  IndexT leafSample(unsigned int tIdx,
                    IndexT leafIdx,
                    vector<unsigned int> &sampRanks) const;


 public:
  /**
     @brief Constructor for invocation from within core.

     Parameters mirror simililarly-named members.
   */
  Quant(const class LeafFrameReg* leaf,
        const class Bag* bag,
        const vector<double>& quantile_);

  /**
     @brief Getter for number of quantiles.

     @return qCount value.
   */
  unsigned int getNQuant() const {
    return qCount;
  }


  /**
     @brief Getter for number of rows predicted.

     Returns zero if empty bag precludes valRank from initialization.
   */
  unsigned int getNRow() const;

  
  /**
     @brief Accessor for predicted quantiles.

     @return vector of quantile predictions.
   */
  const vector<double> getQPred() const {
    return qPred;
  }
  
  
  /**
     @brief Accessor for estimand quantiles.

     @return pointer to base of estimand quantiles.
   */
  const vector<double> getQEst() const {
    return qEst;
  }
  
  
  /**
     @brief Fills in the quantile leaves for each row within a contiguous block.

     @param rowStart is the first row at which to predict.

     @param extent is the number of rows to predict.
  */
  void predictAcross(const class PredictFrame* frame,
                     size_t rowStart,
                     size_t extent);
};

#endif
