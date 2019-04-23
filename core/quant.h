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

#ifndef ARBORIST_QUANT_H
#define ARBORIST_QUANT_H

#include <vector>

#include "typeparam.h"

/**
   @brief Value and row of ranked response.
 */
struct ValRow {
  double val;
  unsigned int row;

  void init(double val, unsigned int row) {
    this->val = val;
    this->row = row;
  }
};

/**
 @brief Quantile signature.
*/
class Quant {
  static const size_t binSize; // # slots to track.
  const class LeafFrameReg *leafReg; // Summary of trained terminal nodes.
  const class BitMatrix* baggedRows; // In-bag summary.
  const vector<ValRow> yRanked; // ordered version of yTrain, with ranks.
  const vector<class RankCount> rankCount; // forest-wide, by sample.
  const double* quantile; // quantile values over which to predict.
  const unsigned int qCount; // # quantile values, above.
  vector<double> qPred; // predicted quantiles.
  vector<double> qEst; // quantile of response estimates.
  unsigned int rankScale; // log2 of scaling factor.
  const vector<double> binMean;

  /**
     @brief Computes a bin offset for a given rank.

     @param rank is the rank in question.

     @return bin offset.
   */
  inline unsigned int binRank(size_t rank) const {
    return rank >> rankScale;
  }


  /**
     @brief Ranks the training response.

     @param leafReg is the leaf frame.

     @return ranked representation of response.
   */
  static vector<ValRow> rankResponse(const class LeafFrameReg* leafReg);
  

  /**
     @brief Computes the count and rank of every bagged sample in the forest.
  */
  static vector<class RankCount> baggedRanks(const class BitMatrix* baggedRows,
                                             const class LeafFrameReg* leafReg,
                                             const vector<ValRow>& yRanked);
  

  /**
     @brief Determines scaling factor for training response.

     @return power-of-two divisor for training response length.
   */
  unsigned int binScale() const;


  /**
     @brief Bins response means.

     @param yRanked[] contains the ranked response/row pairs.

     @param rankScale is the bin scaling factor.

     @return binned vector of response means.
   */
  static vector<double> binMeans(const vector<ValRow>& yRanked,
                                 unsigned int rankScale);

  
  /**
     @brief Writes the quantile values for a given row.

     @param rowBlock is the block-relative row index.

     @param[out] qRow[] outputs the 'qCount' quantile values.
  */
  void predictRow(const class Predict *predict,
                  unsigned int rowBlock,
                  double yPred,
                  double qRow[],
                  double* qEst);


  /**
     @brief Writes quantile values a row of predictions.

     @param sCount is a bin of ranked sample counts.

     @param threshold is the sample count threshold for a given quantile.

     @param yPred is the predicted response for the current row.

     @param[out] qRow[] outputs the derived quantiles.
   */
  unsigned int quantSamples(const vector<unsigned int>& sCount,
                            const vector<double> threshold,
                            double yPred,
                            double qRow[]) const;

  /**
     @brief Accumulates the ranks assocated with predicted leaf.

     @param tIdx is a tree index.

     @param leafIdx is a tree-relative leaf index.

     @param[in,out] sampRanks counts the number of samples at a (binned) rank.

     @return count of samples subsumed by leaf.
  */
  unsigned int leafSample(unsigned int tIdx,
                          unsigned int leafIdx,
                          vector<unsigned int> &sampRanks) const;


 public:
  /**
     @brief Constructor for invocation from within core.

     Parameters mirror simililarly-named members.
   */
  Quant(const struct PredictBox* box,
        const double* quantile_,
        unsigned int qCount_);

  /**
     @brief Getter for number of quantiles.

     @return qCount value.
   */
  unsigned int getNQuant() const {
    return qCount;
  }


  /**
     @brief Getter for number of rows predicted.

     Returns zero if empty bag precludes yRanked from initialization.
   */
  unsigned int getNRow() const;

  
  /**
     @brief Accessor for predicted quantiles.

     @return pointer to base of quantile predictions.
   */
  const double *QPred() const {
    return &qPred[0];
  }
  
  
  /**
     @brief Accessor for estimand quantiles.

     @return pointer to base of estimand quantiles.
   */
  const vector<double> &getQEst() const {
    return qEst;
  }
  
  
  /**
     @brief Fills in the quantile leaves for each row within a contiguous block.

     @param rowStart is the first row at which to predict.

     @param rowEnd is first row at which not to predict.
  */
  void predictAcross(const class Predict *predict,
                     unsigned int rowStart,
                     unsigned int rowEnd);
};

#endif
