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
#include <cmath>


RLEFrame::RLEFrame(size_t nRow_,
		   const vector<unsigned int>& factorTop_,
		   const vector<size_t>& runVal,
		   const vector<size_t>& runLength,
		   const vector<size_t>& runRow,
		   const vector<size_t>& rleHeight,
		   const vector<double>& numVal,
		   const vector<size_t>& numHeight,
		   const vector<unsigned int>& facVal,
		   const vector<size_t>& facHeight) :
  nObs(nRow_),
  factorTop(factorTop_),
  noRank(max(nObs, static_cast<size_t>(*max_element(factorTop.begin(), factorTop.end())))),
  rlePred(packRLE(rleHeight, runVal, runRow, runLength)),
  numRanked(vector<vector<double>>(numHeight.size())),
  facRanked(vector<vector<unsigned int>>(facHeight.size())),
  blockIdx(vector<unsigned int>(rleHeight.size())) {

  unsigned int numIdx = 0;
  unsigned int factorIdx = 0;
  size_t numOff = 0;
  size_t facOff = 0;
  for (unsigned predIdx = 0; predIdx != blockIdx.size(); predIdx++) {
    if (factorTop[predIdx] == 0) {
      for (; numOff < numHeight[numIdx]; numOff++) {
	numRanked[numIdx].emplace_back(numVal[numOff]);
      }
      blockIdx[predIdx] = numIdx++;
    }
    else {
      unsigned int maxVal = factorTop[predIdx] + 1;
      for (; facOff < facHeight[factorIdx]; facOff++) {
	facRanked[factorIdx].emplace_back(min(maxVal, facVal[facOff]));
      }
      blockIdx[predIdx] = factorIdx++;
    }
  }
}


vector<vector<RLEVal<szType>>> RLEFrame::packRLE(const vector<size_t>& rleHeight,
		       const vector<size_t>& runVal,
		       const vector<size_t>& runRow,
		       const vector<size_t>& runLength) {
  vector<vector<RLEVal<szType>>> rlePred(rleHeight.size());
  size_t rleOff = 0;
  for (unsigned int predIdx = 0; predIdx < rleHeight.size(); predIdx++) {
    for (; rleOff < rleHeight[predIdx]; rleOff++) {
      rlePred[predIdx].emplace_back(runVal[rleOff], runRow[rleOff], runLength[rleOff]);
    }
  }

  return rlePred;
}



size_t RLEFrame::findRankMissing(unsigned int predIdx) const {
  size_t rankMissing = noRank;
  unsigned int idx = blockIdx[predIdx];
  if (factorTop[predIdx] > 0) { // Factor
    if (facRanked[idx].back() > factorTop[predIdx]) {
      rankMissing = rlePred[predIdx].back().val;
    }
  }
  else { // Numeric.
    if (isnan(numRanked[idx].back())) {
      rankMissing = rlePred[predIdx].back().val;
    }
  }
  
  return rankMissing;
}


void RLEFrame::reorderRow() {
  for (auto & rleVal : rlePred) {
    sort(rleVal.begin(), rleVal.end(), RLECompareRow<szType>);
  }
}


vector<RLEVal<szType>> RLEFrame::permute(unsigned int predIdx,
					 const vector<size_t>& idxPerm) const {
  vector<size_t> row2Rank(nObs);
  for (auto rle : rlePred[predIdx]) {
    for (size_t row = rle.row; row != rle.row + rle.extent; row++) {
      row2Rank[row] = rle.val;
    }
  }

  vector<RLEVal<szType>> rleOut;
  size_t rankPrev = nObs; // Inattainable.  Forces new RLE on first iteration.
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
