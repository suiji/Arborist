// This file is part of ArboristCore.

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

  /**
     @brief Constructor from packed representation.
   */
RLEFrame::RLEFrame(size_t nRow_,
		   const vector<unsigned int>& cardinality_,
		   const RLEVal<unsigned int>* rle,
		   const vector<size_t>& rleHeight,
		   const vector<double>& numVal_,
		   const vector<size_t>& numOff_) :
  nRow(nRow_),
  cardinality(cardinality_),
  nPred(rleHeight.size()),
  rlePred(vector<vector<RLEVal<unsigned int>>>(nPred)),
  nPredNum(numOff_.size()),
  numVal(numVal_),
  numOff(numOff_) {
  size_t heightPrev = 0;
  size_t off = 0;
  unsigned int predIdx = 0;
  for (auto height : rleHeight) {
    for (size_t i = heightPrev; i < height; i++) {
      RLEVal<unsigned int> rleElt = rle[off++];
      rlePred[predIdx].emplace_back(rleElt.val, rleElt.row, rleElt.extent);
    }
    heightPrev = height;
    predIdx++;
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
