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
  unsigned int Lookup(unsigned int &_rank) const {
    _rank = rank;
    return row;
  }

  void Init(unsigned int _row, unsigned int _rank) {
    row = _row;
    rank = _rank;
  }


  void Ref(unsigned int &_row, unsigned int &_rank) const {
    _row = row;
    _rank = rank;
  }
};


/**
  @brief Rank orderings of predictors.

*/
class RowRank {
  const unsigned int nRow;
  const unsigned int nPred;
  const unsigned int *feInvNum; // Numeric predictors only:  split assignment.
  const unsigned int noRank; // Inattainable rank value.
  static constexpr double plurality = 1.0;// Suppress, for now 0.25;
  
  std::vector<unsigned int> denseRank;
  std::vector<RRNode> rrNode;
  std::vector<unsigned int> rrCount;
  std::vector<unsigned int> rrStart;


  static void FacSort(const unsigned int predCol[], unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rle);
  static void NumSort(const double predCol[], unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, unsigned int invRank[]);

  unsigned int DenseRanks(const std::vector<unsigned int> &feRank, const std::vector<unsigned int> &rle, unsigned int nonCmprTot);
  void Decompress(const std::vector<unsigned int> &feRow, const std::vector<unsigned int> &feRank, const std::vector<unsigned int> &rle, unsigned int nonCmprTot);

 public:
  static void PreSortNum(const double _feNum[], unsigned int _nPredNum, unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, unsigned int _feInvNum[]);
  static void PreSortFac(const unsigned int _feFac[], unsigned int _nPredFac, unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &runLength);


  RowRank(const std::vector<unsigned int> &feRow, const std::vector<unsigned int> &feRank, const unsigned int _feInvNum[], const std::vector<unsigned int> &feRunLength, unsigned int _nRow, unsigned int _nPred);
  ~RowRank();

  inline unsigned int ExplicitCount(unsigned int predIdx) const {
    return rrCount[predIdx];
  }


  inline void Ref(unsigned int predIdx, unsigned int idx, unsigned int &_row, unsigned int &_rank) const {
    rrNode[rrStart[predIdx] + idx].Ref(_row, _rank);
  }

  
  /**
     @brief asssumes numerical predictor.

     @return a (possibly nonunique) row index at which predictor has rank passed.
   */
  inline unsigned int Rank2Row(unsigned int predIdx, unsigned int _rank) const {
    return feInvNum[predIdx * nRow + _rank];
  }

  
  /**
     @brief Accessor for dense rank value associated with a predictor.

     @param predIdx is the predictor index.

     @return dense rank assignment for predictor.
   */
  unsigned int DenseRank(unsigned int predIdx) const{
    return denseRank[predIdx];
  }

  
  double MeanRank(unsigned int predIdx, double rkMean) const;
};

#endif

