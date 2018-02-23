// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file frameblock.cc

   @brief Methods for blocks of similarly-typed predictors.

   @author Mark Seligman
 */

#include "frameblock.h"
#include <vector>
#include <algorithm>


/**
   @brief

   @return void.
 */
FrameTrain::FrameTrain(const vector<unsigned int> &_feCard,
		 unsigned int _nPred,
		 unsigned int _nRow) :
  FrameMap(_nRow, _nPred - _feCard.size(), _feCard.size()),
  feCard(_feCard),
  cardMax(nPredFac > 0 ? *max_element(feCard.begin(), feCard.end()) : 0) {
}


/**
   @brief Static initialization for prediction.

   @return void.
 */
FramePredict::FramePredict(/*
			   const vector<double> &_valNum,
		     const vector<unsigned int> &_rowStart,
		     const vector<unsigned int> &_runLength,
		     const vector<unsigned int> &_predStart,
		     double *_feNumT,
		     unsigned int *_feFacT,*/
			   BlockNum *_blockNum,
			   BlockFac *_blockFac,
		     unsigned int _nPredNum,
		     unsigned int _nPredFac,
		     unsigned int _nRow) :
  FrameMap(_nRow, _nPredNum, _nPredFac),
  blockNum(_blockNum),
  blockFac(_blockFac) {
  //blockNum(BlockNum::Factory(_valNum, _rowStart, _runLength,_predStart, _feNumT, nPredNum)),
  //  blockFac(BlockFac::Factory(_feFacT, nPredFac)) {
}


FramePredict::~FramePredict() {
  delete blockNum;
  delete blockFac;
}


/*
BlockNum *BlockNum::Factory(const vector<double> &_valNum,
			    const vector<unsigned int> &_rowStart,
			    const vector<unsigned int> &_runLength,
			    const vector<unsigned int> &_predStart,
			    double *_feNumT,
			    unsigned int _nPredNum) {
  if (_valNum.size() > 0) {
    return new BlockSparse(_valNum, _rowStart, _runLength, _predStart);
  }
  else {
    return new BlockDense(_feNumT, _nPredNum);
  }
}
*/

/**
   @brief RLE variant NYI.
 */
BlockFac *BlockFac::Factory(unsigned int *_feFacT, unsigned int nPredFac) {
  return new BlockFac(_feFacT, nPredFac);
}


/**
     @brief Sparse constructor.
   */
BlockSparse::BlockSparse(const vector<double> &_valNum,
			 const vector<unsigned int> &_rowStart,
			 const vector<unsigned int> &_runLength,
			 const vector<unsigned int> &_predStart) :
  BlockNum(_predStart.size()),
  valNum(_valNum),
  rowStart(_rowStart),
  runLength(_runLength),
  predStart(_predStart) {

  // Both 'blockNumT' and 'valPrev' are updated before the next use, so
  // need not be initialized.
  blockNumT = new double[FramePredict::rowBlock * nPredNum];
  val = new double[nPredNum];

  rowNext = new unsigned int[nPredNum];
  idxNext = new unsigned int[nPredNum];
  for (unsigned int predIdx = 0; predIdx < nPredNum; predIdx++) {
    rowNext[predIdx] = 0; // Position of first update.
    idxNext[predIdx] = predStart[predIdx]; // Current starting offset.
  }
}


BlockSparse::~BlockSparse() {
  delete [] blockNumT;
  delete [] rowNext;
  delete [] idxNext;
  delete [] val;
}


/**
   @brief Requires sequential update by row, but could be parallelized by
   chunking predictors independently.

   @return void.
 */
void BlockSparse::Transpose(unsigned int rowBegin, unsigned int rowEnd) {
  for (unsigned int row = rowBegin; row < rowEnd; row++) {
    for (unsigned int predIdx = 0; predIdx < nPredNum; predIdx++) {
      if (row == rowNext[predIdx]) { // Assignments persist across invocations:
	unsigned int vecIdx = idxNext[predIdx];
	val[predIdx] = valNum[vecIdx];
	rowNext[predIdx] = rowStart[vecIdx] + runLength[vecIdx];
	idxNext[predIdx] = ++vecIdx;
      }
      blockNumT[(row - rowBegin) * nPredNum + predIdx] = val[predIdx];
    }
  }
}

  
