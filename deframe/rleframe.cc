// This file is part of deframe.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rleframe.cc

   @brief Methods for manipulating observation frames in RLE form.

   @author Mark Seligman
 */

#include "rleframe.h"


RLEFrame::RLEFrame(size_t nRow_,
		   const vector<PredictorForm>& predForm_,
		   const vector<size_t>& runVal,
		   const vector<size_t>& runLength,
		   const vector<size_t>& runRow,
		   const vector<size_t>& rleHeight,
		   const vector<double>& numVal,
		   const vector<size_t>& numHeight,
		   const vector<unsigned int>& facVal,
		   const vector<size_t>& facHeight) :
  nRow(nRow_),
  predForm(predForm_),
  rlePred(vector<vector<RLEVal<unsigned int>>>(rleHeight.size())),
  numRanked(vector<vector<double>>(numHeight.size())),
  facRanked(vector<vector<unsigned int>>(facHeight.size())) {
  size_t off = 0;
  unsigned int predIdx = 0;
  for (auto height : rleHeight) {
    for (; off < height; off++) {
      rlePred[predIdx].emplace_back(runVal[off], runRow[off], runLength[off]);
    }
    predIdx++;
  }
  off = predIdx = 0;
  for (auto height : numHeight) {
    for (; off < height; off++) {
      numRanked[predIdx].emplace_back(numVal[off]);
    }
    predIdx++;
  }
  off = predIdx = 0;
  for (auto height : facHeight) {
    for (; off < height; off++) {
      facRanked[predIdx].emplace_back(facVal[off]);
    }
    predIdx++;
  }
}


void RLEFrame::reorderRow() {
  for (auto & rleVal : rlePred) {
    sort(rleVal.begin(), rleVal.end(), RLECompareRow<unsigned int>);
  }
}


void RLEFrame::transpose(vector<size_t>& idxTr,
			 size_t rowStart,
			 size_t rowExtent,
			 vector<unsigned int>& trFac,
			 vector<double>& trNumeric) const {
  size_t rowOff = 0;
  size_t rowEnd = min(nRow, rowStart + rowExtent);
  for (size_t row = rowStart; row != rowEnd; row++) {
    unsigned int numIdx = 0;
    unsigned int facIdx = 0;
    for (unsigned int predIdx = 0; predIdx < idxTr.size(); predIdx++) {
      unsigned int rank = idxRank(rlePred[predIdx], idxTr[predIdx], row);
      if (predForm[predIdx] == PredictorForm::numeric) {
	trNumeric[rowOff * getNPredNum() + numIdx] = numRanked[numIdx][rank];
	numIdx++;
      }
      else {// TODO:  Replace subtraction with (front end)::fac2Rank()
	trFac[rowOff * getNPredFac() + facIdx] = facRanked[facIdx][rank] - 1;
	facIdx++;
      }
    }
    rowOff++;
  }
}


vector<RLEVal<unsigned int>> RLEFrame::permute(unsigned int predIdx,
					       const vector<size_t>& idxPerm) const {
  vector<size_t> row2Rank(nRow);
  for (auto rle : rlePred[predIdx]) {
    for (size_t row = rle.row; row != rle.row + rle.extent; row++) {
      row2Rank[row] = rle.val;
    }
  }

  vector<RLEVal<unsigned int>> rleOut;
  size_t rankPrev = nRow; // Inattainable.  Forces new RLE on first iteration.
  size_t row = 0;
  for (auto idx : idxPerm) {
    auto rankThis = row2Rank[idx];
    if (rankThis == rankPrev) {
      rleOut.back().extent++;
    }
    else {
      rleOut.emplace_back(rankThis, row, 1);
      rankPrev = rankThis;
    }
    row++;
  }

  return rleOut;
}
