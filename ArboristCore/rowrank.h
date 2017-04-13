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
#include <tuple>
#include <cmath>

#include "param.h"
//#include <iostream>
//using namespace std;

typedef std::pair<double, unsigned int> ValRowD;
typedef std::tuple<double, unsigned int, unsigned int> RLENum;
typedef std::pair<unsigned int, unsigned int> ValRowI;


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
  const unsigned int noRank; // Inattainable rank value.
  static constexpr double plurality = 0.25;

  // Jagged array holding numerical predictor values for split assignment.
  const unsigned int *numOffset; // Per-predictor starting offsets.
  const double *numVal; // Actual predictor values.

  unsigned int nonCompact;  // Total count of uncompactified predictors.
  unsigned int accumCompact;  // Sum of compactified lengths.
  std::vector<unsigned int> denseRank;
  RRNode *rrNode;
  std::vector<unsigned int> rrCount;
  std::vector<unsigned int> rrStart;
  std::vector<unsigned int> safeOffset; // Either an index or an accumulated count.

  
  static void FacSort(const unsigned int predCol[], unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rle);
  static void NumSortRaw(const double predCol[], unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut, std::vector<double> &numOut);
  static unsigned int NumSortRLE(const double colNum[], unsigned int _nRow, const unsigned int rowStart[], const unsigned int runLength[], std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rlOut, std::vector<double> &numOut);

  static void RankFac(const std::vector<ValRowI> &valRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut);
  static void RankNum(const std::vector<ValRowD> &valRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut, std::vector<double> &numOut);
  static void RankNum(const std::vector<RLENum> &rleNum, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut, std::vector<double> &numOut);
  static void Rank2Row(const std::vector<ValRowD> &valRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut);

  static inline unsigned int RunSlot(const unsigned int feRLE[], const unsigned int feRow[], const unsigned int feRank[], unsigned int rleIdx, unsigned int &row, unsigned int &rank) {
    row = feRow[rleIdx];
    rank = feRank[rleIdx];
    return feRLE[rleIdx];
  };

  
  static inline unsigned int RunSlot(const unsigned int feRLE[], const unsigned int feRank[], unsigned int rleIdx, unsigned int &rank) {
    rank = feRank[rleIdx];
    return feRLE[rleIdx];
  };

  
  void DenseBlock(const unsigned int feRank[], const unsigned int feRLE[], unsigned int feRLELength);
  void DenseMode(unsigned int predIdx, unsigned int denseMax, unsigned int argMax);
  unsigned int ModeOffsets();
  void Decompress(const unsigned int feRow[], const unsigned int feRank[], const unsigned int feRLE[], unsigned int feRLELength);

  inline double NumVal(unsigned int predIdx, unsigned int rk) const {
    return numVal[numOffset[predIdx] + rk];
  }
  
 public:
  static void PreSortNum(const double _feNum[], unsigned int _nPredNum, unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut, std::vector<unsigned int> &valOffOut, std::vector<double> &numOut);

  static void PreSortNumRLE(const double valNum[], const unsigned int rowStart[], const unsigned int runLength[], unsigned int _nPredNum, unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rlOut, std::vector<unsigned int> &valOffOut, std::vector<double> &numOut);
  
  static void PreSortFac(const unsigned int _feFac[], unsigned int _nPredFac, unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &runLength);


  RowRank(const class PMTrain *pmTrain, const unsigned int feRow[], const unsigned int feRank[], const unsigned int _numOffset[], const double _numVal[], const unsigned int feRLE[], unsigned int feRLELength);
  ~RowRank();

  
  inline unsigned int NPred() const {
    return nPred;
  }
  
  
  inline unsigned int ExplicitCount(unsigned int predIdx) const {
    return rrCount[predIdx];
  }


  inline void Ref(unsigned int predIdx, unsigned int idx, unsigned int &_row, unsigned int &_rank) const {
    rrNode[rrStart[predIdx] + idx].Ref(_row, _rank);
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
    return nonCompact * stride + accumCompact; // TODO:  align.
  }

  
  /**
     @brief Computes conservative offset for storing predictor-based
     information.

     @param predIdx is the predictor index.

     @param stride is the multiplier for strided access.

     @param extent outputs the number of slots avaiable for staging.

     @return safe offset.
   */
  unsigned int SafeOffset(unsigned int predIdx, unsigned int stride, unsigned int &extent) const {
    extent = denseRank[predIdx] == noRank ? stride : rrCount[predIdx];
    return denseRank[predIdx] == noRank ? safeOffset[predIdx] * stride : nonCompact * stride + safeOffset[predIdx]; // TODO:  align.
  }


  /**
     @brief Derives split values for a numerical predictor by synthesizing
     a fractional intermediate rank and interpolating.

     @param predIdx is the predictor index.

     @param rankRange is the range of ranks.

     @return predictor value at mean rank, computed by PBTrain method.
  */
  inline double QuantRank(unsigned int predIdx, RankRange rankRange, const double splitQuant[]) const {
  double rankNum = rankRange.rankLow + splitQuant[predIdx] * (rankRange.rankHigh - rankRange.rankLow);
  unsigned int rankFloor = floor(rankNum);
  unsigned int rankCeil = ceil(rankNum);

  return NumVal(predIdx, rankFloor) + (rankNum - rankFloor) * (NumVal(predIdx, rankCeil) - NumVal(predIdx, rankFloor));
 }
};

#endif

