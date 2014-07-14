
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <R.h>

#include "train.h"
#include "node.h"
#include "dataord.h"
#include "predictor.h"
#include "response.h"

#include <iostream>
using namespace std;

bool *DataOrd::inBag = 0; // Whether row in this tree is an in-bag sample.
int *DataOrd::sCountRow = 0; // # Samples per row. 0 <=> OOB.
int *DataOrd::sIdxRow = 0; // Index of row in sample vector.

// Only client is quantile regression.
//int *DataOrd::sample2Count = 0; // # Samples at index in sample vector; >= 0.

Dord *DataOrd::dOrd = 0;

// Assumes Predictor factory has been called.
//
void DataOrd::Factory() {
  inBag = new bool[Predictor::nRow];
  sCountRow = new int[Predictor::nRow];
  sIdxRow = new int[Predictor::nRow];

  // 'dOrd' is reused at each tree.
  //
  dOrd = new Dord[Predictor::nRow * Predictor::nPred];

  // Rewritten across trees:
  //sample2Count = new int[Train::nSamp]; // Actually only need max('bagCount') many.

  // Both methods are implemented by Predictor class as an iterator suitable for exporting
  // has not yet been developed.
  //
  // The construction of 'rank2Row[]' can be blocked in predictor chunks, should memory
  // become a limiting resource.  If 'dOrd' is to be blocked as well, however, then its
  // level-based consumers must also be blocked across trees.
  //
  int *rank2Row = new int[Predictor::nRow * Predictor::nPred];
  Predictor::UniqueRank(rank2Row);
  Predictor::SetSortAndTies(rank2Row, dOrd);
  delete [] rank2Row;
}


//
void DataOrd::DeFactory() {
  delete [] inBag;
  delete [] sCountRow;
  delete [] sIdxRow;
  //  delete [] sample2Count;
  delete [] dOrd;

  inBag = 0;
  sCountRow = 0;
  sIdxRow = 0;
  //  sample2Count = 0;
  dOrd = 0;
}

// Private
void DataOrd::CountRows(const int rvRow[]) {
  for (int row= 0; row < Predictor::nRow; row++)
    sCountRow[row] = 0;

  // Counts occurrences of the rank associated with each target 'row' of the
  // sampling vector.
  //
  for (int i = 0; i < Train::nSamp; i++) {
    int row = rvRow[i]; // Train::rowSamples[i];
    sCountRow[row]++;
  }
}

// Once per tree, inverts the randomly-sampled vector of rows.
// 'rvRow' is the tree-defining ordering of sampled rows.
// 'smpCount' enumerates each row's occurence count.
//
// The number of unique rows is the size of the bag.  With compression,
// however, the resulting number of samples is smaller than the bag count.
//
// Returns the sum of sampled response values for intiializition of topmost
// accumulator.
//
void DataOrd::SampleRows(const int rvRow[], Sample sample[], int sample2Rank[], int &bagCount) {
  CountRows(rvRow);
  // Enables lookup by row, for PredByRank(), or index, for LevelMap.
  //
  int idx = 0;
  for (int row = 0; row < Predictor::nRow; row++) {
    int sCount = sCountRow[row];
    if (sCount > 0) {
      double sum = sCount * Response::response->y[row];
      sample[idx].val = sum;
      sample[idx].rowRun = sCount;
      sIdxRow[row] = idx;

      // Only client for these two is quantile regression, but cheap to compute.
      sample2Rank[idx] = ResponseReg::row2Rank[row];
      idx++;
      inBag[row] = true;
    }
    else {
      inBag[row] = false;
      sIdxRow[row] = -1;
    }
    bagCount = idx;
  }
}

// Same as for regression case, but allocates and sets 'ctg' value, as well.
// Full row count is used to avoid the need to rewalk.
//
void DataOrd::SampleRows(const int rvRow[], const int yCtg[], SampleCtg sampleCtg[], int &bagCount) {
  CountRows(rvRow);

  int idx = 0;
  for (int row = 0; row < Predictor::nRow; row++) {
    int sCount = sCountRow[row];
    if (sCount > 0) {
      double sum = sCount * Response::response->y[row];
      sampleCtg[idx].val = sum;
      sampleCtg[idx].rowRun = sCount;
      sampleCtg[idx].ctg = yCtg[row];
      sIdxRow[row] = idx;
      idx++;
      inBag[row] = true;
    }
    else {
      inBag[row] = false;
      sIdxRow[row] = -1;
    }
    bagCount = idx;
  }
}

// For each predictor derives rank associated with sampled row and random vector index.
// Writes predTree[] for subsequent use by Level() calls.
// 
//
void DataOrd::PredByRank(const int predIdx, const Sample sample[], PredOrd predTree[]) {
  Dord *dCol = dOrd + predIdx * Predictor::nRow;
  // 'rank' values must be recorded in nondecreasing rank order.
  //
  int ptIdx = 0;
  for (int rk = 0; rk < Predictor::nRow; rk++) {
    Dord dc = dCol[rk];
    int row = dc.row;
    int sCount = sCountRow[row]; // Should be predictor-invariant.
    if (sCount > 0) {
      PredOrd tOrd;
      int sampleIdx = sIdxRow[row];
      tOrd.yVal = sample[sampleIdx].val;
      tOrd.rowRun = sample[sampleIdx].rowRun;
      tOrd.rank = dc.rank;
      tOrd.sampleIdx = sampleIdx;
      predTree[ptIdx++] = tOrd;
    }
  }
}

void DataOrd::PredByRank(const int predIdx, const SampleCtg sampleCtg[], PredOrdCtg predTreeCtg[]) {
  Dord *dCol = dOrd + predIdx * Predictor::nRow;
  // 'rank' values must be recorded in nondecreasing rank order.
  //
  int ptIdx = 0;
  for (int rk = 0; rk < Predictor::nRow; rk++) {
    Dord dc = dCol[rk];
    int row = dc.row;
    int sCount = sCountRow[row]; // Should be predictor-invariant.
    if (sCount > 0) {
      PredOrdCtg tOrdCtg;
      int sampleIdx = sIdxRow[row];
      tOrdCtg.yVal = sampleCtg[sampleIdx].val;
      tOrdCtg.rowRun = sampleCtg[sampleIdx].rowRun;
      tOrdCtg.rank = dc.rank;
      tOrdCtg.sampleIdx = sampleIdx;
      tOrdCtg.ctg = sampleCtg[sampleIdx].ctg;
      predTreeCtg[ptIdx++] = tOrdCtg;
    }
  }
}
  //  if (ptIdx != bagCount)
  //cout << "ptIdx:  " << ptIdx << " != " << bagCount << endl;
  // Postconds:  ptIdx == bagCount; sum(sCount) = nSamp

