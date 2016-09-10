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
#include "math.h"

#include <algorithm>

// Testing only:
//#include <iostream>
//using namespace std;

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
void RowRank::PreSortNum(const double _feNum[], unsigned int _nPredNum, unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, unsigned int _feInvNum[]) {
  int numIdx, colOff;

  for (numIdx = 0; numIdx < int(_nPredNum); numIdx++) {
    colOff = _nRow * numIdx;
    NumSort(&_feNum[colOff], _nRow, rowOut, rankOut, &_feInvNum[colOff]);
  }
}


/**
   @brief Factor predictor presort to parallel output vectors.

   @param feFac is a block of factor predictor values.

   @param _facStart is the starting output position for factors.

   @param nPredFac is the number of factor predictors.

   @param nRow is the number of observation rows. 

   @param rowOrd outputs the (unstably) sorted row indices.

   @param rank Outputs the tie-classed predictor ranks.

   @output void, with output vector parameters.
 */
void RowRank::PreSortFac(const unsigned int _feFac[], unsigned int _nPredFac, unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &runLength) {
  // Builds the ranked factor block.  Assumes 0-justification has been 
  // performed by bridge.
  //
  for (unsigned int facIdx = 0; facIdx < _nPredFac; facIdx++) {
    FacSort(&_feFac[facIdx * _nRow], _nRow, rowOut, rankOut, runLength);
  }
}


void RowRank::FacSort(const unsigned int predCol[], unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rle) {
typedef std::pair<unsigned int, unsigned int> ValRowI;
  std::vector<ValRowI> valRow(_nRow);
  for (unsigned int row = 0; row < _nRow; row++) {
    valRow[row].first = predCol[row];
    valRow[row].second = row;
  }
  std::sort(valRow.begin(), valRow.end()); // Stable sort.

  // Final "rank" values are the internal factor codes and may contain
  // gaps.  A dense numbering scheme would entail backmapping at LH bit
  // assignment following splitting (q.v.):  prediction and training
  // must map to the same factor levels.
  unsigned int rankPrev = valRow[0].first;
  unsigned int rowPrev = valRow[0].second;
  rle.push_back(1);
  rankOut.push_back(rankPrev);
  rowOut.push_back(rowPrev);
  for (unsigned int row = 1; row < _nRow; row++) {
    unsigned int rankThis = valRow[row].first;
    unsigned int rowThis = valRow[row].second;
    if (rankThis == rankPrev && rowThis == (rowPrev + 1)) {
      rle.back()++;
    }
    else {
      rle.push_back(1);
      rankOut.push_back(rankThis);
      rowOut.push_back(rowThis);
    }
    rankPrev = rankThis;
    rowPrev = rowThis;
  }
}



/**
   @brief

   @return void.
 */
void RowRank::NumSort(const double predCol[], unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, unsigned int invRank[]) {
typedef std::pair<double, unsigned int> ValRowD;
  std::vector<ValRowD> valRow(_nRow);
  for (unsigned int row = 0; row < _nRow; row++) {
    valRow[row].first = predCol[row];
    valRow[row].second = row;
  }
  std::sort(valRow.begin(), valRow.end());  // Stable sort.

  unsigned int rk = 0;
  double prevX = valRow[0].first;
  for (unsigned int row = 0; row < _nRow; row++) {
    double curX = valRow[row].first;
    rk += curX == prevX ? 0 : 1;
    prevX = curX;
    rankOut.push_back(rk);
    rowOut.push_back(valRow[row].second);
    invRank[rk] = valRow[row].second;
  }
  // Under the current stable sorting scheme, invRank[] is assigned the
  // highest-valued row index within a run of identical rank values.
  // Furthermore, as the presence of ties causes the loop to terminate
  // with 'rk < row', invRank[] may be undefined beyond some index.
  // 
}


/**
   @brief Constructor for row, rank passed from front end as parallel arrays.

   @param feRow is the vector of rows allocated by the front end.

   @param feRank is the vector of ranks allocated by the front end.

   @param feInvNum is the rank-to-row mapping for numeric predictors.
 */
RowRank::RowRank(const std::vector<unsigned int> &feRow, const std::vector<unsigned int> &feRank, const unsigned int _feInvNum[], const std::vector<unsigned int> &feRunLength, unsigned int _nRow, unsigned int _nPred) : nRow(_nRow), nPred(_nPred), feInvNum(_feInvNum), noRank(_nRow) {
  std::vector<unsigned int> _rrCount(nPred);
  std::vector<unsigned int> _rrStart(nPred);
  rrCount = std::move(_rrCount);
  rrStart = std::move(_rrStart);
  // Default initialization, overwritten by RLE processing.
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    rrCount[predIdx] = nRow;
    rrStart[predIdx] = nRow * predIdx;
  }
  // By convention, non-compressed row/rank pairs precede any rle-compressed
  // pairs.
  //
  unsigned int nonCmprTot = feRank.size() - feRunLength.size();
  unsigned int cmprTot = DenseRanks(feRank, feRunLength, nonCmprTot);
  std::vector<RRNode> _rrNode(nonCmprTot + cmprTot);
  rrNode = std::move(_rrNode);
  for (unsigned int i = 0; i < nonCmprTot; i++) {
    rrNode[i].Init(feRow[i], feRank[i]);
  }
  Decompress(feRow, feRank, feRunLength, nonCmprTot);
}


/**
   @brief Counts the number of rows to be decompressed and sets dense ranks.

   @param rle records the run lengths of the remaining entries.

   @param nonCmprTot is the total number of noncompressed rows.

   @return total number of rows to be decompressed.
 */
void RowRank::Decompress(const std::vector<unsigned int> &feRow, const std::vector<unsigned int> &feRank, const std::vector<unsigned int> &rle, unsigned int nonCmprTot) {
  unsigned int predCmpr = nonCmprTot / nRow; // First compressed predictor.
  unsigned int outIdx = nonCmprTot;
  unsigned int inIdx = nonCmprTot;
  unsigned int rleIdx = 0; // To change.
  for (unsigned int predIdx = predCmpr; predIdx < nPred; predIdx++) {
    unsigned int rowTot = 0;
    while (rowTot < nRow) {
      unsigned int rank = feRank[inIdx];
      unsigned int row = feRow[inIdx];
      unsigned int runLength = rle[rleIdx++];
      rowTot += runLength;
      if (rank != denseRank[predIdx]) { // Omits dense ranks.
	for (unsigned int i = 0; i < runLength; i++) { // Expands non-dense ranks.
	  rrNode[outIdx++].Init(row + i, rank);
	}
      }
      inIdx++;
    }
  }
}


/**
   @brief Counts the number of rows to be decompressed and sets dense ranks.

   @param rle records the run lengths of the remaining entries.

   @param nonCmprTot is the total number of noncompressed rows.

   @return total number of rows to be decompressed.
 */
unsigned int RowRank::DenseRanks(const std::vector<unsigned int> &feRank, const std::vector<unsigned int> &rle, unsigned int nonCmprTot) {
  std::vector<unsigned int> _denseRank(nPred);
  std::fill(_denseRank.begin(), _denseRank.end(), noRank);
  denseRank = std::move(_denseRank);

  unsigned int cmprTot = 0;
  unsigned int predCmpr = nonCmprTot / nRow; // First compressed predictor.
  unsigned int rleIdx = 0; // Will change.
  unsigned int inIdx = nonCmprTot;
  for (unsigned int predIdx = predCmpr; predIdx < nPred; predIdx++) {
    unsigned int rowTot = 0;
    unsigned int denseCount = 0; // Running maximum of run counts.
    unsigned int argMax = noRank;
    unsigned int runCount = 0; // Runs across adjacent rle entries.
    unsigned int rankPrev = noRank;
    while (rowTot < nRow) {
      unsigned int runLength = rle[rleIdx++];
      unsigned int rankThis = feRank[inIdx++];
      rowTot += runLength;
      if (rankThis == rankPrev) {
	runCount += runLength;
      }
      else {
	runCount = runLength;
	rankPrev = rankThis;
      }
      if (runCount > denseCount) {
	denseCount = runCount;
	argMax = rankThis;
      }
    }

    if (denseCount > plurality * nRow) {
      denseRank[predIdx] = argMax;
      cmprTot += nRow - denseCount;
    }
    else {
      cmprTot += nRow;
    }
  }
  
  return cmprTot;
}


/**
   @brief Deallocates and resets.

   @return void.
 */
RowRank::~RowRank() {
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
