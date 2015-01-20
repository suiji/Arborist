// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "sample.h"
#include "samplepred.h"
#include "predictor.h"
#include "splitpred.h"
#include "pretree.h"

#include <iostream>
using namespace std;

// Simulation-invariant values.
//
int Sample::nRow = -1;
int Sample::nPred = -1;
int Sample::nSamp = -1;
PredOrd *Sample::predOrd = 0;

// Tree-invariant values.
//
bool *Sample::inBag = 0; // Whether row in this tree is an in-bag sample.
int *Sample::sCountRow = 0; // # Samples per row. 0 <=> OOB.
int *Sample::sIdxRow = 0; // Index of row in sample vector.

int *SampleReg::sample2Rank = 0; // Only client is quantile regression.
SampleReg *SampleReg::sampleReg = 0;

SampleCtg *SampleCtg::sampleCtg = 0;

// Assumes Predictor factory has been called.
//
void Sample::Factory(int _nRow, int _nSamp, int _nPred) {
  nRow = _nRow;
  nPred = _nPred;
  nSamp = _nSamp;

  // 'dOrd' is invariant across the training phase.
  //
 predOrd = new PredOrd[nRow * nPred];

  // Both methods are implemented by Predictor class as an iterator suitable for exporting
  // has not yet been developed.
  //
  // The construction of 'rank2Row[]' can be blocked in predictor chunks, should memory
  // become a limiting resource.  If 'dOrd' is to be blocked as well, however, then its
  // level-based consumers must also be blocked across trees.
  //
  int *rank2Row = new int[nRow * nPred];
  Predictor::UniqueRank(rank2Row);
  Predictor::SetSortAndTies(rank2Row, predOrd);

  // Can instead be retained for scoring by rank.
  delete [] rank2Row; 
}


//
void Sample::DeFactory() {
  delete [] predOrd;
  predOrd = 0;
}

// Private
void Sample::CountRows(const int rvRow[]) {
  TreeInit();

  sCountRow = new int[nRow];
  for (int row= 0; row < nRow; row++)
    sCountRow[row] = 0;

  // Counts occurrences of the rank associated with each target 'row' of the
  // sampling vector.
  //
  for (int i = 0; i < nSamp; i++) {
    int row = rvRow[i];
    sCountRow[row]++;
  }
}

// Allocations for tree-based data structures.
//
void Sample::TreeInit() {
  sIdxRow = new int[nRow];
  inBag = new bool[nRow];

  SamplePred::TreeInit(nPred, nSamp);
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
int SampleReg::SampleRows(const int rvRow[], const double y[], const int row2Rank[]) {
  CountRows(rvRow);

  sampleReg = new SampleReg[nSamp]; // Lives until TreeClear()
  sample2Rank = new int[nSamp]; // " " 

  // Enables lookup by row, for Stage(), or index, for LevelMap.
  //
  int bagCount = 0;
  int idx = 0;
  for (int row = 0; row < nRow; row++) {
    int sCount = sCountRow[row];
    if (sCount > 0) {
      double sum = sCount * y[row];
      sampleReg[idx].sum = sum;
      sampleReg[idx].rowRun = sCount;
      sIdxRow[row] = idx;

      // Only client for these two is quantile regression, but cheap to compute.
      sample2Rank[idx] = row2Rank[row];
      idx++;
      inBag[row] = true;
    }
    else {
      inBag[row] = false;
      sIdxRow[row] = -1;
    }
    bagCount = idx;
  }

  return bagCount;
}

// Same as for regression case, but allocates and sets 'ctg' value, as well.
// Full row count is used to avoid the need to rewalk.
//
int SampleCtg::SampleRows(const int rvRow[], const int yCtg[], const double y[]) {
  sampleCtg = new SampleCtg[nSamp]; // Lives until TreeClear()

  CountRows(rvRow);

  int maxSCount = 1;
  int idx = 0;
  int bagCount = 0;
  for (int row = 0; row < nRow; row++) {
    int sCount = sCountRow[row];
    if (sCount > 0) {
      if (sCount > maxSCount)
	maxSCount = sCount;
      double sum = sCount * y[row];
      sampleCtg[idx].sum = sum;
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
  SamplePred::SetCtgShift(maxSCount);

  return bagCount;
}

void SampleReg::Stage() {
  int predIdx;
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = 0; predIdx < nPred; predIdx++) {
	Stage(predIdx);
      }
    }
}


// For each predictor derives rank associated with sampled row and random vector index.
// Writes predTree[] for subsequent use by Level() calls.
// 
void SampleReg::Stage(int predIdx) {
  SamplePred *samplePred = SamplePred::BufferOff(predIdx, 0);
  PredOrd *dCol = predOrd + predIdx * nRow;

  // 'rk' values must be recorded in nondecreasing rank order.
  //
  int ptIdx = 0;
  for (int rk = 0; rk < nRow; rk++) {
    PredOrd dc = dCol[rk];
    int row = dc.row;
    int sCount = sCountRow[row]; // Should be predictor-invariant.
    if (sCount > 0) {
      int sIdx = sIdxRow[row];
      SampleReg sReg = sampleReg[sIdx];
      SamplePred::SetReg(samplePred, ptIdx++, sIdx, sReg.sum, dc.rank, sReg.rowRun);
    }
  }
}

void SampleCtg::Stage() {
  int predIdx;
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = 0; predIdx < nPred; predIdx++) {
	Stage(predIdx);
      }
    }
}

void SampleCtg::Stage(int predIdx) {
  SamplePred *samplePred = SamplePred::BufferOff(predIdx, 0);
  PredOrd *dCol = predOrd + predIdx * nRow;
  // 'rk' values must be recorded in nondecreasing rank order.
  //
  int ptIdx = 0;
  for (int rk = 0; rk < nRow; rk++) {
    PredOrd dc = dCol[rk];
    int row = dc.row;
    int sCount = sCountRow[row]; // Should be predictor-invariant.
    if (sCount > 0) {
      int sIdx = sIdxRow[row];
      SampleCtg sCtg = sampleCtg[sIdx];
      SamplePred::SetCtg(samplePred, ptIdx++, sIdx, sCtg.sum, dc.rank, sCtg.rowRun, sCtg.ctg);
    }
  }
}
  //  if (ptIdx != bagCount)
  //cout << "ptIdx:  " << ptIdx << " != " << bagCount << endl;
  // Postconds:  ptIdx == bagCount; sum(sCount) = nSamp


// Walks the sample set, accumulating value sums for the associated leaves.  Score
// is the sample mean.  These values could also be computed by passing sums down the
// pre-tree and pulling them from terminal nodes.
//
// 'sampleReg[]' deleted here.
//
void SampleReg::Scores(int bagCount, int treeHeight, double score[]) {
  int *sCount = new int[treeHeight];
  for (int pt = 0; pt < treeHeight; pt++) {
    score[pt] = 0.0;
    sCount[pt]= 0;
  }

  for (int i = 0; i < bagCount; i++) {
    int leafIdx = PreTree::Sample2Leaf(i);
    score[leafIdx] += sampleReg[i].sum;
    sCount[leafIdx] += sampleReg[i].rowRun;
  }

  for (int pt = 0; pt < treeHeight; pt++) {
    if (sCount[pt] > 0)
      score[pt] /= sCount[pt];
  }

  delete [] sCount;
  delete [] sampleReg;
  sampleReg = 0;
}

// Scores are extracted once per tree, after all leaves have been marked.
// 'sampleCtg[]' deleted here.
//
void SampleCtg::Scores(int bagCount, int ctgWidth, int treeHeight, double score[]) {
  double *leafWS = new double[ctgWidth * treeHeight];

  for (int i = 0; i < ctgWidth * treeHeight; i++)
    leafWS[i] = 0.0;

  // Irregular access.  Needs the ability to map sample indices to the factors and
  // weights with which they are associated.
  //
  for (int i = 0; i < bagCount; i++) {
    int leafIdx = PreTree::Sample2Leaf(i);
    int ctg = sampleCtg[i].ctg;
    // ASSERTION:
    if (ctg < 0 || ctg >= ctgWidth)
      cout << "Bad response category:  " << ctg << endl;
    double responseWeight = sampleCtg[i].sum;
    leafWS[leafIdx * ctgWidth + ctg] += responseWeight;
  }

  // Factor weights have been jittered, making ties highly unlikely.  Even in the
  // event of a tie, although the first in the run is chosen, the jittering itself
  // is nondeterministic.
  //
  // Every leaf should obtain a non-negative factor-valued score.
  //
  for (int leafIdx = 0; leafIdx < treeHeight; leafIdx++) {
    double *ctgBase = leafWS + leafIdx * ctgWidth;
    double maxWeight = 0.0;
    int argMaxWeight = -1;
    for (int ctg = 0; ctg < ctgWidth; ctg++) {
      double thisWeight = ctgBase[ctg];
      //cout << "Leaf " << leafIdx << " factor index: " << fac << ", weight:  " << thisWeight << endl;
      if (thisWeight > maxWeight) {
	maxWeight = thisWeight;
	argMaxWeight = ctg;
      }
    }
    score[leafIdx] = argMaxWeight; // For now, upcasts score to double, for compatability with DecTree.
    //    cout << leafIdx << ":  " << maxWeightIdx << endl;
  }
  // ASSERTION:
  //  Can count nonterminals and verify #nonterminals == treeHeight - leafCount

  delete [] leafWS;
  delete [] sampleCtg;
  sampleCtg = 0;
}

void Sample::TreeClear() {
  SamplePred::TreeClear();

  delete [] sCountRow;
  delete [] sIdxRow;
  delete [] inBag;
  sCountRow = 0;
  sIdxRow = 0;
  inBag = 0;
}

void SampleReg::TreeClear() {
  delete [] sample2Rank;
  Sample::TreeClear();
}

void SampleCtg::TreeClear() {
  delete [] sampleCtg;
  Sample::TreeClear();
}

// Copies leaf information into leafOff[] and ranks[].
//
void SampleReg::DispatchQuantiles(int treeSize, int bagCount, int leafPos[], int leafExtent[], int rank[], int rankCount[]) {
  // Must be wide enough to access all decision-tree offsets.
  int *seen = new int[treeSize];
  for (int i = 0; i < treeSize; i++) {
    seen[i] = 0;
    leafExtent[i] = 0;
    leafPos[i] = -1;
  }
  for (int sIdx = 0; sIdx < bagCount; sIdx++) {
    int leafIdx = PreTree::Sample2Leaf(sIdx);
    leafExtent[leafIdx]++;
  }

  int totCt = 0;
  for (int i = 0; i < treeSize; i++) {
    if (leafExtent[i] > 0) {
      leafPos[i] = totCt;
      totCt += leafExtent[i];
    }
  }

  for (int sIdx = 0; sIdx < bagCount; sIdx++) {
    // ASSERTION:
    //    if (rk > Predictor::nRow)
    //  cout << "Invalid rank:  " << rk << " / " << Predictor::nRow << endl;
    int leafIdx = PreTree::Sample2Leaf(sIdx);
    int rkOff = leafPos[leafIdx] + seen[leafIdx];
    rank[rkOff] = sample2Rank[sIdx];
    rankCount[rkOff] = sampleReg[sIdx].rowRun;
    seen[leafIdx]++;
  }

  delete [] seen;
}
