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
  static constexpr double plurality = 0.25;

  unsigned int nonCompact;  // Total count of uncompactified predictors.
  unsigned int accumCompact;  // Sum of compactified lengths.
  std::vector<unsigned int> denseRank;
  RRNode *rrNode;
  std::vector<unsigned int> rrCount;
  std::vector<unsigned int> rrStart;
  std::vector<unsigned int> safeOffset; // Either an index or an accumulated count.


  static void FacSort(const unsigned int predCol[], unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rle);
  static void NumSort(const double predCol[], unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, unsigned int invRank[]);

  unsigned int DenseBlock(const std::vector<unsigned int> &feRank, const std::vector<unsigned int> &rle, unsigned int nonCmprTot, unsigned int blockFirst);
  void Decompress(const std::vector<unsigned int> &feRow, const std::vector<unsigned int> &feRank, const std::vector<unsigned int> &rle, unsigned int inIdx, unsigned int blockFirst);

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
     @brief Row-lookup mechanism for numerical splitting values.

     @param predIdx is the index of the splitting predictor.

     @param _rank is a rank bound for the splitting criterion.

     @return a row index at which predictor value has desired rank.
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

  
  /**
     @brief Computes a conservative buffer size, allowing strided access
     for noncompact predictors but full-width access for compact predictors.

     @param stride is the desired strided access length.

     @return buffer size conforming to conservative constraints.
   */
  unsigned int SafeSize(unsigned int stride) const {
    return nPred * stride; // Until starting offsets cached.
    //return nonCompact * stride + accumCompact;
  }

  
  /**
     @brief Computes conservative offset for storing predictor-based
     information.

     @param predIdx is the predictor index.

     @param stride is the multiplier for strided access.

     @return safe offset.
   */
  unsigned int SafeOffset(unsigned int predIdx, unsigned int stride) const {
    return predIdx * stride; // Until starting offsets cached.
    //    return denseRank[predIdx] == noRank ? safeOffset[predIdx] * stride : nonCompact + safeOffset[predIdx];
  }

  
  double MeanRank(unsigned int predIdx, double rkMean) const;
};

#endif

