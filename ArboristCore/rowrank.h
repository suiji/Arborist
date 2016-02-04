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

class RRNode {
  unsigned int row;
  unsigned int rank;
 public:
  unsigned int Lookup(unsigned int &_rank) {
    _rank = rank;
    return row;
  }
  void Set(int _row, int _rank) {
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
  const int *invNum; // Numeric predictors only:  split assignment.
  RRNode *rowRank;
  BlockRank *blockRank;

  static void Sort(int _nRow, int _nPredNum, double numOrd[], int perm[]);
  static void Sort(int _nRow, int _nPredFac, int facOrd[], int perm[]);
  static void Ranks(unsigned int _nRow, unsigned int _nPredNum, double _numOrd[], int _row[], int _rank[], int _invRank[]);
  static void Ranks(unsigned int _nRow, unsigned int _nPredFac, int _facOrd[], int _rank[]);
  static void Ranks(unsigned int _nRow, const double xCol[], const int row[], int rank[], int invRank[]);
  static void Ranks(unsigned int _nRow, const int xCol[], int rank[]);

 public:
  static void PreSortNum(const double _feNum[], unsigned int _nPredNum, unsigned int _nRow, int _row[], int _rank[], int _invNum[]);
  static void PreSortFac(const int _feFac[], unsigned int _nPredNum, unsigned int _nPredFac, unsigned int _nRow, int _row[], int _rank[]);

  RowRank(const int _feRow[], const int _feRank[], const int _feInvNum[], unsigned int _nRow, unsigned int _nPredDense);
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
  inline unsigned int Rank2Row(unsigned int predIdx, int _rank) const {
    return invNum[predIdx * nRow + _rank];
  }
  
  double MeanRank(unsigned int predIdx, double rkMean) const;
};

#endif

