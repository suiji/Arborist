// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file block.cc

   @brief Methods for blocks of similarly-typed predictors.

   @author Mark Seligman
 */

#include "block.h"
#include "predict.h"
#include <algorithm>



/**
   @brief RLE variant NYI.
 */
BlockFac *BlockFac::Factory(unsigned int *_feFacT, unsigned int nCol) {
  return new BlockFac(_feFacT, nCol);
}


/**
     @brief Sparse constructor for prediction frame.
   */
BlockSparse::BlockSparse(const double *_val,
			 const unsigned int *_rowStart,
			 const unsigned int *_runLength,
			 const unsigned int *_predStart,
			 unsigned int _nCol) :
  BlockNum(_nCol),
  val(_val),
  rowStart(_rowStart),
  runLength(_runLength),
  predStart(_predStart) {

  // Both 'blockNumT' and 'valPrev' are updated before the next use, so
  // need not be initialized.
  blockNumT = new double[Predict::rowBlock * nCol];
  transVal = new double[nCol];

  rowNext = new unsigned int[nCol];
  idxNext = new unsigned int[nCol];
  for (unsigned int predIdx = 0; predIdx < nCol; predIdx++) {
    rowNext[predIdx] = 0; // Position of first update.
    idxNext[predIdx] = predStart[predIdx]; // Current starting offset.
  }
}


BlockSparse::~BlockSparse() {
  delete [] blockNumT;
  delete [] rowNext;
  delete [] idxNext;
  delete [] transVal;
}


void BlockSparse::transpose(unsigned int rowBegin, unsigned int rowEnd) {
  for (unsigned int row = rowBegin; row < rowEnd; row++) {
    for (unsigned int predIdx = 0; predIdx < nCol; predIdx++) {
      if (row == rowNext[predIdx]) { // Assignments persist across invocations:
	unsigned int vecIdx = idxNext[predIdx];
	transVal[predIdx] = val[vecIdx];
	rowNext[predIdx] = rowStart[vecIdx] + runLength[vecIdx];
	idxNext[predIdx] = ++vecIdx;
      }
      blockNumT[(row - rowBegin) * nCol + predIdx] = transVal[predIdx];
    }
  }
}


BSCresc::BSCresc(unsigned int nRow_,
                 unsigned int nCol_) :
  nRow(nRow_),
  nPred(nCol_),
  predStart(vector<unsigned int>(nPred)) {
}


void BSCresc::nzRow(const double eltsNZ[],
                    const int nz[],
                    const int p[]) {
  // Pre-scans column heights.
  const double zero = 0.0;
  vector<unsigned int> nzHeight(nPred + 1);
  unsigned int idxStart = p[0];
  for (unsigned int colIdx = 1; colIdx <= nPred; colIdx++) {
    nzHeight[colIdx - 1] = p[colIdx] - idxStart;
    idxStart = p[colIdx];
  }

  for (unsigned int colIdx = 0; colIdx < predStart.size(); colIdx++) {
    unsigned int colHeight = nzHeight[colIdx]; // # nonzero values in column.
    predStart[colIdx] = valNum.size();
    if (colHeight == 0) { // No nonzero values for predictor.
      pushRun(zero, nRow, 0);
    }
    else {
      unsigned int nzPrev = nRow; // Inattainable row value.
      // Row indices into 'i' and 'x' are zero-based.
      unsigned int idxStart = p[colIdx];
      unsigned int idxEnd = idxStart + colHeight;
      for (unsigned int rowIdx = idxStart; rowIdx < idxEnd; rowIdx++) {
        unsigned int nzRow = nz[rowIdx]; // row # of nonzero element.
        if (nzPrev == nRow && nzRow > 0) { // Zeroes lead.
          pushRun(zero, nzRow, 0);
        }
        else if (nzRow > nzPrev + 1) { // Zeroes precede.
          pushRun(zero, nzRow - (nzPrev + 1), nzPrev + 1);
        }
        pushRun(eltsNZ[rowIdx], 1, nzRow);
        nzPrev = nzRow;
      }
      if (nzPrev + 1 < nRow) { // Zeroes trail.
        pushRun(zero, nRow - (nzPrev + 1), nzPrev + 1);
      }
    }
  }
}
