// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rankedframe.cc

   @brief Methods for presorting and accessing predictors by rank.

   @author Mark Seligman
 */

#include "rankedframe.h"
#include "framemap.h"
#include "valrank.h"

#include <algorithm>

// Observations are blocked according to type.  Blocks written in separate
// calls from front-end interface.

/**
   @brief Constructor for row, rank passed from front end as parallel arrays.

   @param feRow is the vector of rows allocated by the front end.

   @param feRank is the vector of ranks allocated by the front end.

 */
RankedFrame::RankedFrame(unsigned int nRow_,
                         const vector<unsigned int>& cardinality,
                         unsigned int nPred_,
                         const unsigned int feRow[],
                         const unsigned int feRank[],
                         const unsigned int feRLE[],
                         unsigned int rleLength,
                         double autoCompress) :
  nRow(nRow_),
  nPred(nPred_),
  noRank(cardinality.size() == 0 ? nRow : max(nRow, *max_element(cardinality.begin(), cardinality.end()))),
  nPredDense(0),
  denseIdx(vector<unsigned int>(nPred)),
  nonCompact(0),
  accumCompact(0),
  denseRank(vector<unsigned int>(nPred)),
  explicitCount(vector<unsigned int>(nPred)),
  rrStart(vector<unsigned int>(nPred)),
  safeOffset(vector<unsigned int>(nPred)),
  denseThresh(autoCompress * nRow) {
  // Default initializations:
  fill(denseIdx.begin(), denseIdx.end(), nPred);
  fill(denseRank.begin(), denseRank.end(), noRank);
  fill(explicitCount.begin(), explicitCount.end(), nRow);
  unsigned int explCount = denseBlock(feRank, feRLE, rleLength);
  modeOffsets();

  rrNode = vector<RowRank>(explCount);
  decompress(feRow, feRank, feRLE, rleLength);
}


unsigned int RankedFrame::denseBlock(const unsigned int feRank[], const unsigned int feRLE[], unsigned int rleLength) {
  unsigned int explCount = 0;
  unsigned int rleIdx = 0;
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    unsigned int denseMax = 0; // Running maximum of run counts.
    unsigned int argMax = noRank;
    unsigned int runCount = 0; // Runs across adjacent rle entries.
    unsigned int rankPrev = noRank;
    unsigned int rank;
    unsigned int runLength = runSlot(feRLE, feRank, rleIdx, rank);

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
      runLength = runSlot(feRLE, feRank, rleIdx, rank);
    }
    // Post condition:  rowTot == nRow.

    explCount += denseMode(predIdx, denseMax, argMax);
  }

  return explCount;
}


unsigned int RankedFrame::denseMode(unsigned int predIdx, unsigned int denseMax, unsigned int argMax) {
  if (denseMax <= denseThresh) {
    safeOffset[predIdx] = nonCompact++; // Index:  non-dense storage.
    return nRow; // All elements explicit.
  }

  // Sufficiently long run found:
  denseRank[predIdx] = argMax;
  safeOffset[predIdx] = accumCompact; // Accumulated offset:  dense.
  unsigned int rowCount = nRow - denseMax;
  accumCompact += rowCount;
  denseIdx[predIdx] = nPredDense++;
  explicitCount[predIdx] = rowCount;
  return rowCount; // denseMax-many elements implicit.
}


void RankedFrame::modeOffsets() {
  unsigned int denseBase = nonCompact * nRow;
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    unsigned int offSafe = safeOffset[predIdx];
    rrStart[predIdx] = denseRank[predIdx] != noRank ? denseBase + offSafe :
      offSafe * nRow;
  }
}


void RankedFrame::decompress(const unsigned int feRow[], const unsigned int feRank[], const unsigned int feRLE[], unsigned int rleLength) {
  unsigned int rleIdx = 0;
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    unsigned int outIdx = rrStart[predIdx];
    unsigned int row, rank;
    unsigned int runLength = runSlot(feRLE, feRow, feRank, rleIdx, row, rank);
    for (unsigned int rowTot = runLength; rowTot <= nRow; rowTot += runLength) {
      if (rank != denseRank[predIdx]) { // Non-dense runs expanded.
	for (unsigned int i = 0; i < runLength; i++) {
	  rrNode[outIdx++].init(row + i, rank);
	}
      }
      if (++rleIdx == rleLength)
	break;
      runLength = runSlot(feRLE, feRow, feRank, rleIdx, row, rank);
    }
    // Post-condition:  outIdx - rrStart[predIdx] == explicitCount[predIdx]
  }
}


/**
   @brief Destructor.
 */
RankedFrame::~RankedFrame() {
}
