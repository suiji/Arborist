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
RowRank::RowRank(const std::vector<unsigned int> &feRow, const std::vector<unsigned int> &feRank, const unsigned int _feInvNum[], const std::vector<unsigned int> &feRunLength, unsigned int _nRow, unsigned int _nPred) : nRow(_nRow), nPred(_nPred), feInvNum(_feInvNum), noRank(std::max(nRow, PBTrain::CardMax())), nonCompact(0), accumCompact(0) {
  // Default initialization to uncompressed values.
  rrCount.reserve(nPred);
  rrStart.reserve(nPred);
  safeOffset.reserve(nPred);
  denseRank.reserve(nPred);
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    safeOffset[predIdx] = predIdx;
    rrCount[predIdx] = nRow;
    rrStart[predIdx] = nRow * predIdx;
    denseRank[predIdx] = noRank;
  }

  // By convention, rle-compressed pairs given highest offsets.
  //
  unsigned int nonCmprTot = feRank.size() - feRunLength.size();
  nonCompact = nonCmprTot / nRow;
  unsigned int blockFirst = nonCompact;
  unsigned int blockTot = DenseBlock(feRank, feRunLength, nonCmprTot, blockFirst);
  rrNode = new RRNode[nonCmprTot + blockTot];
  for (unsigned int i = 0; i < nonCmprTot; i++) {
    RRNode rr;
    rr.Init(feRow[i], feRank[i]);
    rrNode[i] = rr;
  }

  Decompress(feRow, feRank, feRunLength, nonCmprTot, blockFirst);
}


/**
   @brief Counts the number of rows to be decompressed and sets dense ranks.

   @param rle records the run lengths of the remaining entries.

   @param nonCmprTot is the total number of noncompressed rows.

   @return total number of rows to be decompressed.
 */
unsigned int RowRank::DenseBlock(const std::vector<unsigned int> &feRank, const std::vector<unsigned int> &rle, unsigned int blockIn, unsigned int blockFirst) {
  unsigned int rleIdx = 0; // Will change.
  unsigned int inIdx = blockIn;
  for (unsigned int predIdx = blockFirst; predIdx < nPred; predIdx++) {
    unsigned int denseMax = 0; // Running maximum of run counts.
    unsigned int argMax = noRank;
    unsigned int runCount = 0; // Runs across adjacent rle entries.
    unsigned int rankPrev = noRank;
    
    for (unsigned int rowTot = rle[rleIdx]; rowTot <= nRow; rowTot += rle[rleIdx]) {
      unsigned int runLength = rle[rleIdx];
      unsigned int rankThis = feRank[inIdx++];
      if (rankThis == rankPrev) {
	runCount += runLength;
      }
      else {
	runCount = runLength;
	rankPrev = rankThis;
      }
      if (runCount > denseMax) {
	denseMax = runCount;
	argMax = rankThis;
      }
      rleIdx++;
      if (rleIdx == rle.size())
	break;
    }
    // Post condition:  rowTot == nRow.

    unsigned int rowCount;
    if (denseMax > plurality * nRow) {
      denseRank[predIdx] = argMax;
      safeOffset[predIdx] = accumCompact; // Accumulated offset:  dense.
      rowCount = nRow - denseMax;
      accumCompact += rowCount;
    }
    else {
      denseRank[predIdx] = noRank;
      safeOffset[predIdx] = nonCompact++; // Index:  non-dense storage.
      rowCount = nRow;
    }
    rrCount[predIdx] = rowCount;
  }

  // Assigns rrNode[] offsets ut noncompressed predictors stored first,
  // as with staging offsets.
  //
  unsigned int blockTot = 0;
  for (unsigned int predIdx = blockFirst; predIdx < nPred; predIdx++) {
    unsigned int offSafe = safeOffset[predIdx];
    if (denseRank[predIdx] != noRank) {
      rrStart[predIdx] = nonCompact * nRow + offSafe;
    }
    else {
      rrStart[predIdx] = offSafe * nRow;
    }
    blockTot += rrCount[predIdx];
  }
  
  return blockTot;
}


/**
   @brief Decompresses a block of predictors having compressed encoding.

   @param rle records the run lengths of the remaining entries.

   @param predStart is the first predictor in the block.

   @return void.
 */
void RowRank::Decompress(const std::vector<unsigned int> &feRow, const std::vector<unsigned int> &feRank, const std::vector<unsigned int> &rle, unsigned int inIdx, unsigned int blockFirst) {
  unsigned int rleIdx = 0; // To change.
  for (unsigned int predIdx = blockFirst; predIdx < nPred; predIdx++) {
    unsigned int outIdx = rrStart[predIdx];
    for (unsigned int rowTot = rle[rleIdx]; rowTot <= nRow; rowTot += rle[rleIdx]) {
      unsigned int runLength = rle[rleIdx];
      unsigned int rank = feRank[inIdx];
      if (rank != denseRank[predIdx]) { // Omits dense ranks.
	for (unsigned int i = 0; i < runLength; i++) { // Expands runs.
	  RRNode rr;
	  rr.Init(feRow[inIdx] + i, rank);
	  rrNode[outIdx++] = rr;
	}
      }
      inIdx++;
      rleIdx++;
      if (rleIdx == rle.size())
	break;
    }
    //    if (outIdx - rrStart[predIdx] != rrCount[predIdx])
    //cout << "Dense count mismatch" << endl;
  }
}


/**
   @brief Deallocates and resets.

   @return void.
 */
RowRank::~RowRank() {
  delete [] rrNode;
}


/**
  @brief Derives split values for a numerical predictor.

  @param predIdx is the predictor index.

  @param rkMean is the mean splitting rank:  interpolates if fractional.

  @return predictor value at mean rank, computed by PBTrain method.
*/
double RowRank::MeanRank(unsigned int predIdx, double rkMean) const {
  unsigned int rankLow = floor(rkMean);
  unsigned int rankHigh = ceil(rkMean);
  return PBTrain::MeanVal(predIdx, Rank2Row(predIdx, rankLow), Rank2Row(predIdx, rankHigh));
}
