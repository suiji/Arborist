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


BlockSet::BlockSet(Block<double>* blockNum_,
                   BlockDense<unsigned int>* blockFac_,
                   unsigned int nRow_) :
  blockNum(blockNum_),
  blockFac(blockFac_),
  nRow(nRow_) {
}

void BlockSet::transpose(unsigned int rowStart,
                         unsigned int rowEnd,
                         unsigned int rowBlock) const {
  blockNum->transpose(rowStart, rowEnd, rowBlock);
  blockFac->transpose(rowStart, rowEnd, rowBlock);
}


/**
   @return base address for (transposed) numeric values at row.
*/
const double* BlockSet::baseNum(unsigned int rowOff) const {
  return blockNum->rowBase(rowOff);
}


  /**
     @return base address for (transposed) factor values at row.
   */
const unsigned int* BlockSet::baseFac(unsigned int rowOff) const {
  return blockFac->rowBase(rowOff);
}
