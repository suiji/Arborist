// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rowrank.h

   @brief Class definitions for maintenance of predictor ordering.

   @author Mark Seligman
 */

#ifndef ARBORIST_ROWRANK_H
#define ARBORIST_ROWRANK_H

#include <vector>

class RRNode {
  unsigned int row;
  unsigned int rank;
 public:
  unsigned int Lookup(unsigned int &_rank) {
    _rank = rank;
    return row;
  }

  void Set(unsigned int _row, unsigned int _rank) {
    row = _row;
    rank = _rank;
  }
};


/**
   @brief Represents ranks for sparsely-expressed predictors.
 */
class BlockRank {
  unsigned int predIdx;
  unsigned int extent;
  unsigned int row; // Starting row.
  unsigned int rank;
};


/**
  @brief Block and rank orderings of predictors.

*/
class RowRank {
  const unsigned int nRow;
  const unsigned int nBlock; // Number of BlockRank objects.
  const unsigned int nPredDense; // Number of non-sparse predictors.
  const unsigned int *feInvNum; // Numeric predictors only:  split assignment.
  RRNode *rowRank;
  BlockRank *blockRank;

  static void Sort(unsigned int _nRow, unsigned int _nPredNum, double numOrd[], unsigned int perm[]);
  static void Sort(unsigned int _nRow, unsigned int _nPredFac, unsigned int facOrd[], unsigned int perm[]);
  static void Ranks(unsigned int _nRow, unsigned int _nPredNum, double _numOrd[], unsigned int _row[], unsigned int _rank[], unsigned int _invRank[]);
  static void Ranks(unsigned int _nRow, unsigned int _nPredFac, unsigned int _facOrd[], unsigned int _rank[]);
  static void Ranks(unsigned int _nRow, const double xCol[], const unsigned int row[], unsigned int rank[], unsigned int invRank[]);
  static void Ranks(unsigned int _nRow, const unsigned int xCol[], unsigned int rank[]);

 public:
  static void PreSortNum(const double _feNum[], unsigned int _nPredNum, unsigned int _nRow, unsigned int _rowOrd[], unsigned int _rank[], unsigned int _feInvNum[]);
  static void PreSortFac(const unsigned int _feFac[], unsigned int _nPredFac, unsigned int _nRow, unsigned int _rowOrd[], unsigned int _rank[]);


  RowRank(const unsigned int _feRow[], const unsigned int _feRank[], const unsigned int _feInvNum[] , unsigned int _nRow, unsigned int _nPredDense);
  ~RowRank();

  /**
     @brief Looks up row/rank using predictor and index.

     @param predIdx is the predictor index.

     @param idx is the index into a RowRank predictor column.

     @param _row outputs the looked-up row.

     @return rank at predictor/row.
   */
  unsigned int inline Lookup(unsigned int predIdx, unsigned int idx, unsigned int &_rank) const {
    return rowRank[predIdx * nRow + idx].Lookup(_rank);
  }


  /**
     @brief asssumes numerical predictor.

     @return a (possibly nonunique) row index at which predictor has rank passed.
   */
  inline unsigned int Rank2Row(unsigned int predIdx, unsigned int _rank) const {
    return feInvNum[predIdx * nRow + _rank];
  }
  
  double MeanRank(unsigned int predIdx, double rkMean) const;
};

#endif

