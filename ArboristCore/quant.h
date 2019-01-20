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

typedef pair<double, unsigned int> RankedPair;

/**
   @brief Rank and sample-count values derived from BagLeaf.  Client:
   quantile inference.
 */
class RankCount {
 public:
  unsigned int rank;
  unsigned int sCount;

  void Init(unsigned int _rank, unsigned int _sCount) {
    rank = _rank;
    sCount = _sCount;
  }
};


/**
 @brief Quantile signature.
*/
class Quant {
  const class LeafFrameReg *leafReg;
  const double *yTrain;
  vector<RankedPair> yRanked;
  const vector<double> &quantile;
  const unsigned int qCount;
  vector<double> qPred;
  vector<RankCount> rankCount; // forest-wide, by sample.
  unsigned int logSmudge;
  unsigned int binSize;
  vector<unsigned int> binTemp; // Helper vector.
  vector<unsigned int> sCountSmudge;

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

   @return void.
 */
  void smudgeLeaves();

  /**
     @brief Writes the quantile values for a given row.

     @param rowBlock is the block-relative row index.

     @param qRow[] outputs the 'qCount' quantile values.

     @return void, with output vector parameter.
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
  unsigned int ranksExact(unsigned int tIdx, unsigned int leafIdx, vector<unsigned int> &sampRanks);


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
  Quant(const class LeafFrameReg *leafReg_,
        const class BitMatrix *baggedRows,
        const vector<double> &quantile_,
        unsigned int qBin);

  unsigned int NQuant() const {
    return quantile.size();
  }

  
  const double *QPred() const {
    return &qPred[0];
  }
  
  
  /**
     @brief Fills in the quantile leaves for each row within a contiguous block.

     @param rowStart is the first row at which to predict.

     @param rowEnd is first row at which not to predict.

     @return void.
  */
  void predictAcross(const class Predict *predict,
                     unsigned int rowStart,
                     unsigned int rowEnd);
};

#endif
