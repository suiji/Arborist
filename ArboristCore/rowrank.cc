// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rowrank.cc

   @brief Methods for predictor-specific training.

   @author Mark Seligman
 */

#include "rowrank.h"
#include "predblock.h"
#include "callback.h"
#include "math.h"

// Testing only:
//#include <iostream>
using namespace std;

// TODO:  Investigate moving Presorts to bridges, cloning.

// Observations are blocked according to type.  Blocks written in separate
// calls from front-end interface.


/**
   @brief Numeric predictor presort to parallel output vectors.

   @param feNum is a block of numeric predictor values.

   @param nPredNum is the number of numeric predictors.

   @param nRow is the number of observation rows. 

   @param rank outputs the tie-classed predictor ranks.

   @param feInvNum outputs a rank-to-row map.

   @output void, with output vector parameters.
 */
void RowRank::PreSortNum(const double _feNum[], unsigned int _nPredNum, unsigned int _nRow, int _row[], int _rank[], int _feInvNum[]) {
  // Builds the ranked numeric block.
  //
  double *numOrd = new double[_nRow * _nPredNum];
  for (unsigned int i = 0; i < _nRow * _nPredNum; i++)
    numOrd[i] = _feNum[i];
  Sort(_nRow, _nPredNum, numOrd, _row);
  Ranks(_nRow, _nPredNum, numOrd, _row, _rank, _feInvNum);
  delete [] numOrd;
}


/**
   @brief Factor predictor presort to parallel output vectors.

   @param feFac is a block of factor predictor values.

   @param _facStart is the starting output position for factors.

   @param nPredFac is the number of factor predictors.

   @param nRow is the number of observation rows. 

   @param row Outputs the (unstably) sorted row indices.

   @param rank Outputs the tie-classed predictor ranks.

   @output void, with output vector parameters.
 */
void RowRank::PreSortFac(const int _feFac[], unsigned int _facStart, unsigned int _nPredFac, unsigned int _nRow, int _row[], int _rank[]) {

  // Builds the ranked factor block.  Assumes 0-justification has been 
  // performed by bridge.
  //
  int *facOrd = new int[_nRow * _nPredFac];
  for (unsigned int i = 0; i < _nRow * _nPredFac; i++)
    facOrd[i] = _feFac[i];
  Sort(_nRow, _nPredFac, facOrd, _row + _facStart * _nRow);
  Ranks(_nRow, _nPredFac, facOrd, _rank + _facStart * _nRow);
  delete [] facOrd;
}


/**
 @brief Sorts each column of predictors, saving value and permutation vectors.

 @param numOrd outputs the sorted numeric values.

 @param perm outputs the permutation vectors.

 @return void, with output vector parameters.
*/
void RowRank::Sort(int _nRow, int _nPredNum, double numOrd[], int perm[]) {
  int colOff = 0;  
  for (int numIdx = 0; numIdx < _nPredNum; numIdx++, colOff += _nRow) {
    /* Sort-with-index requires a vector of rows to permute.*/
    for (int i = 0; i < _nRow; i++)
      perm[colOff + i] = i;
    // TODO:  Replace with thread-safe sort to permit parallel execution.
    // Row consistency does not appear necessary:  unstable sort suffices.
    //
    CallBack::QSortD(numOrd + colOff, perm + colOff, 1, _nRow);
  }
}


/**
   @brief Same as above, but with factor-valued predictors.

   @return void, with output reference parameters.
 */
void RowRank::Sort(int _nRow, int _nPredFac, int facOrd[], int perm[]) {
  int colOff = 0;
  for (int facIdx = 0; facIdx < _nPredFac; facIdx++, colOff += _nRow) {
    for (int i = 0; i < _nRow; i++)
      perm[colOff + i] = i;
    // TODO:  Replace with thread-safe sort to permit parallel execution.
    CallBack::QSortI(facOrd + colOff, perm + colOff, 1, _nRow);
  }
}


/**
   @brief Loops over numerical predictors to compute row, rank and inverse vectors.

   @param rowRank outputs the matrix of predictor-order objects.

   @return void, with output parameter matrix.
*/
void RowRank::Ranks(unsigned int _nRow, unsigned int _nPredNum, double _numOrd[], int _row[], int _rank[], int _invRank[]) {
  unsigned int colOff = 0;
  unsigned int numIdx; 

  //#pragma omp parallel default(shared) private(predIdx)//, baseOff, rankOff)
  {
    //  #pragma omp for schedule(static, 1) nowait
    for (numIdx = 0; numIdx < _nPredNum; numIdx++, colOff += _nRow) {
      Ranks(_nRow, _numOrd + colOff, _row + colOff, _rank + colOff, _invRank + colOff);
    }
  }
}


/**
   @brief As above, but i) looping over factor predictors and ii) no inverse map computed.
 */
void RowRank::Ranks(unsigned int _nRow, unsigned int _nPredFac, int _facOrd[], int _rank[]) {
  unsigned int colOff = 0;
  unsigned int facIdx;
  
  //#pragma omp parallel default(shared) private(predIdx)//, baseOff, rankOff)
  {    // Factors:
    //#pragma omp for schedule(static, 1) nowait
    for (facIdx = 0; facIdx < _nPredFac; facIdx++, colOff += _nRow) {
      Ranks(_nRow, _facOrd + colOff, _rank + colOff);
    }
  }
}


/**
   @brief Walks sorted predictor rows, assigning rank, row and inverse maps.

   @param xCol[] are the sorted values of a numeric predictor.

   @param row is the permutation vector defined by the sort.

   @param rank outputs the predictor-order rank values.  Rank values are dense, with ties identified.

   @param row[] orders row indices so that predictor is nonondecreasing.

   @param invRank[] maps ranks to one (of possibly many) associated row.

   @return void, with output vector parameters.
*/
void RowRank::Ranks(unsigned int _nRow, const double xCol[], const int row[], int rank[], int invRank[]) {
  unsigned int rk = 0;
  double prevX = xCol[0];
  for (unsigned int rw = 0; rw < _nRow; rw++) {
    double curX = xCol[rw];
    rk = curX == prevX ? rk : rk + 1;
    rank[rw] = rk;
    invRank[rk] = row[rw];  // Assignment of row within a run is arbitrary.
    prevX = curX;
  }
  // Values of invRank[] at indices beyond final 'rk' value are undefined.
}


/**
   @brief Same as above, but for factors.  The zero-based internal
   values of factors are used as proxy "ranks":  gaps are not filled in.

   @param xCol[] are the sorted (nonnegative) predictor values.

   @param rank[] outputs a proxy rank equal to the predictor value.  Rank values need not be dense, as factors may be missing.

   @return void, with output vector parameter.
*/
void RowRank::Ranks(unsigned int _nRow, const int xCol[], int rank[]) {
  for (unsigned int rw = 0; rw < _nRow; rw++) {
    rank[rw] = xCol[rw];
  }
}


/**
   @brief Constructor for row, rank passed from front end as parallel arrays.

   @param _feRow is the vector of rows allocated by the front end.

   @param _feRank is the vector of ranks allocated by the front end.

   @param _feInvNum is the rank-to-row mapping for numeric predictors.
 */
RowRank::RowRank(const int _feRow[], const int _feRank[], const int _feInvNum[], unsigned int _nRow, unsigned int _nPredDense) : nRow(_nRow), nBlock(0), nPredDense(_nPredDense), invNum(_feInvNum) {
  int dim = nRow * nPredDense;

  rowRank = new RRNode[dim];
  for (int i = 0; i < dim; i++) {
    rowRank[i].Set(_feRow[i], _feRank[i]);
  }

  //  blockRank = new BlockRank[nBlock];
}


/**
   @brief Deallocates and resets.

   @return void.
 */
RowRank::~RowRank() {
  delete [] rowRank;
}


/**
  @brief Derives split values for a numerical predictor.

  @param predIdx is the preditor index.

  @param rkMedian is the median splitting rank:  interpolates if fractional.

  @return predictor value at mean rank, computed by PredBlock method.
*/
double RowRank::MeanRank(unsigned int predIdx, double rkMean) const {
  unsigned int rankLow = floor(rkMean);
  unsigned int rankHigh = ceil(rkMean);
  return PBTrain::MeanVal(predIdx, Rank2Row(predIdx, rankLow), Rank2Row(predIdx, rankHigh));
}
