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

   @output void, with output vector parameters.
 */
void RowRank::PreSortNum(const double _feNum[], unsigned int _nPredNum, unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut, std::vector<unsigned int> &numOffOut, std::vector<double> &numOut) {
  for (unsigned int numIdx = 0; numIdx < _nPredNum; numIdx++) {
    numOffOut[numIdx] = numOut.size();
    NumSortRaw(&_feNum[numIdx * _nRow], _nRow, rowOut, rankOut, rleOut, numOut);
  }
}


void RowRank::PreSortNumRLE(const double valNum[], const unsigned int rowStart[], const unsigned int runLength[], unsigned int _nPredNum, unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut, std::vector<unsigned int> &numOffOut, std::vector<double> &numOut) {
  unsigned int colOff = 0;
  for (unsigned int numIdx = 0; numIdx < _nPredNum; numIdx++) {
    numOffOut[numIdx] = numOut.size();
    unsigned int idxCol = NumSortRLE(&valNum[colOff], _nRow, &rowStart[colOff], &runLength[colOff], rowOut, rankOut, rleOut, numOut);
    colOff += idxCol;
  }
}


/**
   @brief Sorts a column of numerical predictor values compressed with
   run-length encoding.

   @return Count of input vector elements read for the column.
 */
unsigned int RowRank::NumSortRLE(const double colNum[], unsigned int _nRow, const unsigned int rowStart[], const unsigned int runLength[], std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut, std::vector <double> &numOut) {
  std::vector<RLENum> rleNum;
  for (unsigned int rleIdx = 0, rowTot = 0; rowTot < _nRow; rowTot += runLength[rleIdx++]) {
    rleNum.push_back(std::make_tuple(colNum[rleIdx], rowStart[rleIdx], runLength[rleIdx]));
  }

  std::sort(rleNum.begin(), rleNum.end()); // runlengths silent, as rows unique.
  RankNum(rleNum, rowOut, rankOut, rleOut, numOut);

  return rleNum.size();
}


/**
   @brief

   @return void.
 */
void RowRank::NumSortRaw(const double colNum[], unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut, std::vector<double> &numOut) {
  std::vector<ValRowD> valRow(_nRow);
  for (unsigned int row = 0; row < _nRow; row++) {
    valRow[row] = std::make_pair(colNum[row], row);
  }

  std::sort(valRow.begin(), valRow.end());  // Stable sort.
  RankNum(valRow, rowOut, rankOut, rleOut, numOut);
}


/**
   @brief Stores ordered predictor column, entering uncompressed.

   @param numOut outputs the rank-ordered predictor values.

   @return void.
 */
void RowRank::RankNum(const std::vector<ValRowD> &valRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut, std::vector<double> &numOut) {
  unsigned int rk = 0;
  rleOut.push_back(1);
  rowOut.push_back(valRow[0].second);
  numOut.push_back(valRow[0].first);
  rankOut.push_back(rk);
  for (unsigned int idx = 1; idx < valRow.size(); idx++) {
    double valThis = valRow[idx].first;
    unsigned int rowThis = valRow[idx].second;

    if (valThis == numOut.back() && rowThis == rowOut.back() + rleOut.back()) {
      rleOut.back()++;
    }
    else { // New RLE, row and rank entries regardless whether tied.
      if (valThis != numOut.back()) {
	rk++;
	numOut.push_back(valThis);
      }
      rankOut.push_back(rk);
      rleOut.push_back(1);
      rowOut.push_back(rowThis);
    }
  }
}


/**
   @brief Stores ordered predictor column compresed by external RLE.

   @return void.
 */
void RowRank::RankNum(const std::vector<RLENum> &rleNum, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut, std::vector<double> &numOut) {
  RLENum elt = rleNum[0];
  unsigned int rk = 0;
  rankOut.push_back(rk);
  numOut.push_back(std::get<0>(elt));
  rowOut.push_back(std::get<1>(elt));
  rleOut.push_back(std::get<2>(elt));
  for (unsigned int idx = 1; idx < rleNum.size(); idx++) {
    elt = rleNum[idx];
    double valThis = std::get<0>(elt);
    unsigned int rowThis = std::get<1>(elt);
    unsigned int runCount = std::get<2>(elt);
    if (valThis == numOut.back() && rowThis == rowOut.back() + rleOut.back()) {
      rleOut.back() += runCount;
    }
    else { // New RLE, rank entries regardless whether tied.
      if (valThis != numOut.back()) {
	rk++;
	numOut.push_back(valThis);
      }
      rankOut.push_back(rk);
      rowOut.push_back(rowThis);
      rleOut.push_back(runCount);
    }
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


/**
   @brief Sorts factors and stores as rank-ordered run-length encoding.

   @return void.
 */
void RowRank::FacSort(const unsigned int predCol[], unsigned int _nRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut) {
  std::vector<ValRowI> valRow(_nRow);
  for (unsigned int row = 0; row < _nRow; row++) {
    valRow[row] = std::make_pair(predCol[row], row);
  }
  std::sort(valRow.begin(), valRow.end()); // Stable sort.
  RankFac(valRow, rowOut, rankOut, rleOut);
}


/**
   @brief Builds rank-ordered run-length encoding to hold factor values.

   Final "rank" values are the internal factor codes and may contain
   gaps.  A dense numbering scheme would entail backmapping at LH bit
   assignment following splitting (q.v.):  prediction and training
   must map to the same factor levels.

   @return void.
*/ 
void RowRank::RankFac(const std::vector<ValRowI> &valRow, std::vector<unsigned int> &rowOut, std::vector<unsigned int> &rankOut, std::vector<unsigned int> &rleOut) {
  unsigned int rankPrev = valRow[0].first;
  unsigned int rowPrev = valRow[0].second;
  rleOut.push_back(1);
  rankOut.push_back(rankPrev);
  rowOut.push_back(rowPrev);
  for (unsigned int row = 1; row < valRow.size(); row++) {
    unsigned int rankThis = valRow[row].first;
    unsigned int rowThis = valRow[row].second;

    if (rankThis == rankPrev && rowThis == (rowPrev + 1)) {
      rleOut.back() ++;
    }
    else {
      rleOut.push_back(1);
      rankOut.push_back(rankThis);
      rowOut.push_back(rowThis);
    }
    rankPrev = rankThis;
    rowPrev = rowThis;
  }
}



/**
   @brief Constructor for row, rank passed from front end as parallel arrays.

   @param feRow is the vector of rows allocated by the front end.

   @param feRank is the vector of ranks allocated by the front end.

 */
RowRank::RowRank(const PMTrain *pmTrain, const unsigned int feRow[], const unsigned int feRank[], const unsigned int *_numOffset, const double *_numVal, const unsigned int feRLE[], unsigned int rleLength, double _autoCompress) : nRow(pmTrain->NRow()), nPred(pmTrain->NPred()), noRank(std::max(nRow, pmTrain->CardMax())), nPredDense(0), denseIdx(std::vector<unsigned int>(nPred)), numOffset(_numOffset), numVal(_numVal), nonCompact(0), accumCompact(0), denseRank(std::vector<unsigned int>(nPred)), explicitCount(std::vector<unsigned int>(nPred)), rrStart(std::vector<unsigned int>(nPred)), safeOffset(std::vector<unsigned int>(nPred)), autoCompress(_autoCompress) {
  DenseBlock(feRank, feRLE, rleLength);
  unsigned int rrSlots = ModeOffsets();
  rrNode = new RRNode[rrSlots];

  Decompress(feRow, feRank, feRLE, rleLength);
}


/**
   @brief Walks the design matrix as RLE entries, merging adjacent
   entries with identical ranks.

   @brief feRank are the ranks corresponding to runlength-encoding (RLE)
   entries.

   @param feRLE are the run lengths corresponding to RLE entries.

   @param rleLength is the count of RLE entries.

   @return void.
 */
void RowRank::DenseBlock(const unsigned int feRank[], const unsigned int feRLE[], unsigned int rleLength) {
  unsigned int rleIdx = 0;
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    unsigned int denseMax = 0; // Running maximum of run counts.
    unsigned int argMax = noRank;
    unsigned int runCount = 0; // Runs across adjacent rle entries.
    unsigned int rankPrev = noRank;
    unsigned int rank;
    unsigned int runLength = RunSlot(feRLE, feRank, rleIdx, rank);

    for (unsigned int rowTot = runLength; rowTot <= nRow; rowTot += runLength) {
      if (rank == rankPrev) {
	runCount += runLength;
      }
      else {
	runCount = runLength;
	rankPrev = rank;
      }
      if (runCount > denseMax) {
	denseMax = runCount;
	argMax = rank;
      }
      if (++rleIdx == rleLength)
	break;
      runLength = RunSlot(feRLE, feRank, rleIdx, rank);
    }
    // Post condition:  rowTot == nRow.

    DenseMode(predIdx, denseMax, argMax);
  }
}


/**
   @brief Determines whether predictor to be stored densely and updates
   storage accumulators accordingly.

   @param predIdx is the predictor under consideration.

   @param denseMax is the highest run length encountered for the predictor:
   must lie within [1, nRow].

   @param argMax is an argmax rank value corresponding to denseMax.

   @return void.
 */
void RowRank::DenseMode(unsigned int predIdx, unsigned int denseMax, unsigned int argMax) {
  unsigned int rowCount;
  if (denseMax > autoCompress * nRow) { // Sufficiently long run found.
    denseRank[predIdx] = argMax;
    safeOffset[predIdx] = accumCompact; // Accumulated offset:  dense.
    rowCount = nRow - denseMax;
    accumCompact += rowCount;
    denseIdx[predIdx] = nPredDense++;
  }
  else {
    denseRank[predIdx] = noRank;
    denseIdx[predIdx] = nPred; // Inattainable index. 
    safeOffset[predIdx] = nonCompact++; // Index:  non-dense storage.
    rowCount = nRow;
  }
  explicitCount[predIdx] = rowCount;
}


/**
   @brief Assigns predictor offsets according to storage mode:
   noncompressed predictors stored first, as with staging offsets.

   @return total number of non-compressed slots.
 */
unsigned int RowRank::ModeOffsets() {
  unsigned int rrSlots = 0;
  unsigned int denseBase = nonCompact * nRow;
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    unsigned int offSafe = safeOffset[predIdx];
    rrStart[predIdx] = denseRank[predIdx] != noRank ? denseBase + offSafe :
      offSafe * nRow;
    rrSlots += explicitCount[predIdx];
  }

  return rrSlots;
}


/**
   @brief Decompresses a block of predictors deemed not to be storable
   densely.

   @param feRow[] are the rows corresponding to distinct runlength-
   encoded (RLE) entries.

   @param feRank[] are the ranks corresponing to RLE entries.

   @param feRLE records the run lengths spanning the original design
   matrix.

   @param rleLength is the total count of RLE entries.

   @return void.
 */
void RowRank::Decompress(const unsigned int feRow[], const unsigned int feRank[], const unsigned int feRLE[], unsigned int rleLength) {
  unsigned int rleIdx = 0;
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    unsigned int outIdx = rrStart[predIdx];
    unsigned int row, rank;
    unsigned int runLength = RunSlot(feRLE, feRow, feRank, rleIdx, row, rank);
    for (unsigned int rowTot = runLength; rowTot <= nRow; rowTot += runLength) {
      if (rank != denseRank[predIdx]) { // Non-dense runs expanded.
	for (unsigned int i = 0; i < runLength; i++) {
	  rrNode[outIdx++].Init(row + i, rank);
	}
      }
      if (++rleIdx == rleLength)
	break;
      runLength = RunSlot(feRLE, feRow, feRank, rleIdx, row, rank);
    }
    //    if (outIdx - rrStart[predIdx] != explicitCount[predIdx])
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
