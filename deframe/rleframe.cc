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

vector<RLEVal<unsigned int>> RLEFrame::permute(unsigned int predIdx,
					       const vector<size_t>& idxPerm) const {
  vector<size_t> row2Rank(nRow);
  for (size_t idx = idxStart(predIdx); idx != idxEnd(predIdx); idx++) {
    auto rleThis = rle[idx];
    for (size_t row = rleThis.row; row != rleThis.row + rleThis.extent; row++) {
      row2Rank[row] = rleThis.val;
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
