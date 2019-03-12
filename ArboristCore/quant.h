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
   @brief Rank and sample-count values derived from BagSample.  Client:
   quantile inference.
 */
struct RankCount {
  unsigned int rank;
  unsigned int sCount;

  void init(unsigned int rank, unsigned int sCount) {
    this->rank = rank;
    this->sCount = sCount;
  }
};


/**
 @brief Quantile signature.
*/
class Quant {
  const class LeafFrameReg *leafReg; // Summary of trained terminal nodes.
  const class BitMatrix* baggedRows; // In-bag summary.
  const double *yTrain; // Training response.
  vector<ValRow> yRanked; // ordered version of yTrain, with ranks.
  const double* quantile; // quantile values over which to predict.
  const unsigned int qCount; // # quantile values, above.
  vector<double> qPred; // predicted quantiles
  vector<RankCount> rankCount; // forest-wide, by sample.
  unsigned int logSmudge; // log2 of smudging factor, if smudging.
  unsigned int binSize; // Width of binning parameter.
  vector<unsigned int> binTemp; // Helper vector.
  vector<unsigned int> sCountSmudge; // Smudged sample counts.

  /**
     @brief Computes the count and rank of every bagged sample in the forest.

     @param baggedRows encodes whether a tree/row pair is bagged.

     @return void, with side-effected rankCount vector.
  */
  void rankCounts(const class BitMatrix *baggedRows);
  

  /**
     @brief Computes bin size and smudging factor.

     @param nRow is the number of rows used to train.

     @param qBin is the bin size specified by the front end.

     @param[out] logSmudge outputs the log2 of the smudging factor.

     @return bin size.
  */
  unsigned int imputeBinSize(unsigned int nRow,
                             unsigned int qBin,
                             unsigned int &_logSmudge);
  /**
   @brief Builds a vector of binned sample counts for wide leaves.
 */
  void smudgeLeaves();

  /**
     @brief Writes the quantile values for a given row.

     @param rowBlock is the block-relative row index.

     @param qRow[] outputs the 'qCount' quantile values.
  */
  void predictRow(const class Predict *predict,
                  unsigned int rowBlock,
                  double qRow[]);


  /**
     @brief Accumulates the ranks assocated with predicted leaf.

     @param tIdx is a tree index.

     @param leafIdx is a tree-relative leaf index.

     @param[in,out] sampRanks counts the number of samples at a given rank.

     @return count of samples subsumed by leaf.
  */
  unsigned int ranksExact(unsigned int tIdx,
                          unsigned int leafIdx,
                          vector<unsigned int> &sampRanks);


  /**
     @brief Accumulates binned ranks assocated with a predicted leaf.

     @param tIdx is a tree index.

     @param leafIdx is the tree-relative leaf index.

     @param sampRanks[in,out] counts the number of samples at a given rank.
     
     @return count of samples subsumed by leaf.
 */
  unsigned int ranksSmudge(unsigned int tIdx,
                           unsigned int LeafIdx,
                           vector<unsigned int> &sampRanks);

  
 public:
  /**
     @brief Constructor for invocation from within core.

     Parameters mirror simililarly-named members.
   */
  Quant(const struct PredictBox* box,
        const double* quantile_,
        unsigned int qCount_,
        unsigned int qBin);

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
  unsigned int getNRow() const {
    return yRanked.size();
  }

  
  /**
     @brief Accessor for predicted quantiles.

     @return pointer to base of quantile predictions.
   */
  const double *QPred() const {
    return &qPred[0];
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
