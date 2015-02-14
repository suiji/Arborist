// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sample.cc

   @brief Methods for sampling from the response to begin training an individual tree.

   @author Mark Seligman
 */

#include "sample.h"
#include "samplepred.h"
#include "predictor.h"
#include "splitpred.h"
#include "pretree.h"

//#include <iostream>
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

/**
 @brief Lights off initilizations needed for sampling.

 @param _nRow is the number of response/observation rows.

 @param _nSamp is the number of samples.

 @param _nPred is the number of predictors.

 @return void.
*/
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


/**
   @brief Finalizer.

   @return void.
*/
void Sample::DeFactory() {
  delete [] predOrd;
  predOrd = 0;
}

/**
   @brief Counts instances of each row index in sample.

   @param rvRow is the vector of sampled row indices.

   @return void.
*/
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

/**
   @brief Per-tree initializations and allocations.

   @return void.
*/
void Sample::TreeInit() {
  sIdxRow = new int[nRow];
  inBag = new bool[nRow];

  SamplePred::TreeInit(nPred, nSamp);
}

/**
   @brief Inverts the randomly-sampled vector of rows.

   @param rvRow is the tree-defining ordering of sampled rows.

   @param y is the response vector.

   @param row2Rank is rank of each sampled row.

   @return count of in-bag samples.
*/
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

/**
   @brief Inverts the randomly-sampled vector of rows.

   @param rvRow is the tree-defining ordering of sampled rows.

   @param yCtg is the response vector.

   @param y is the proxy response vector.

   @return count of in-bag samples.
*/
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

/**
   @brief Records ranked regression sample information per predictor.

   @return void.
 */
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


/**
   @brief Stages the regression sample for a given predictor.

   @param predIdx is the predictor index.

   @return void.
*/
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

/**
   @brief Records ranked categorical sample information per predictor.

   @return void.
 */
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

/**
   @brief Stages the categorical sample for a given predictor.

   @param predIdx is the predictor index.

   @return void.
*/
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

/**
   @brief Derives scores for regression tree.

   @param bagCount is the in-bag sample count.

   @param treeHeight is the number of nodes in the pretree.

   @param score outputs the computed scores.

   @return void, with output parameter vector.
*/
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

/**
   @brief Derives scores for categorical tree.

   @param bagCount is the in-bag sample count.

   @param ctgWidth is the response cardinality.

   @param treeHeight is the number of nodes in the pretree.

   @param score outputs the computed scores.

   @return void, with output parameter vector.
*/

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
    //if (ctg < 0 || ctg >= ctgWidth)
    //cout << "Bad response category:  " << ctg << endl;
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
      if (thisWeight > maxWeight) {
	maxWeight = thisWeight;
	argMaxWeight = ctg;
      }
    }
    score[leafIdx] = argMaxWeight; // For now, upcasts score to double, for compatability with DecTree.
  }
  // ASSERTION:
  //  Can count nonterminals and verify #nonterminals == treeHeight - leafCount

  delete [] leafWS;
  delete [] sampleCtg;
  sampleCtg = 0;
}

/**
   @brief Clears per-tree information.

   @return void.
 */
void Sample::TreeClear() {
  SamplePred::TreeClear();

  delete [] sCountRow;
  delete [] sIdxRow;
  delete [] inBag;
  sCountRow = 0;
  sIdxRow = 0;
  inBag = 0;
}

/**
   @brief Clears regression-specific information and calls base clear method.

   @return void.
 */
void SampleReg::TreeClear() {
  delete [] sample2Rank;
  Sample::TreeClear();
}

/**
   @brief Clears categorical-specific information and calls base clear method.

   @return void.
 */
void SampleCtg::TreeClear() {
  delete [] sampleCtg;
  Sample::TreeClear();
}

/**
   @brief Derives and copies quantile leaf information.

   @param treeSize is the height of the pretree.

   @param bagCount is the size of the in-bag sample set.

   @param qLeafPos outputs quantile leaf offsets; vector length treeSize.

   @param qLeafExtent outputs quantile leaf sizes; vector length treeSize.

   @param rank outputs quantile leaf ranks; vector length bagCount.

   @param rankCount outputs rank multiplicities; vector length bagCount.

   @return void, with output parameter vectors.
 */
void SampleReg::TreeQuantiles(int treeSize, int bagCount, int qLeafPos[], int qLeafExtent[], int qRank[], int qRankCount[]) {
  // Must be wide enough to access all tree offsets.
  int *seen = new int[treeSize];
  for (int i = 0; i < treeSize; i++) {
    seen[i] = 0;
    qLeafExtent[i] = 0;
  }
  for (int sIdx = 0; sIdx < bagCount; sIdx++) {
    int leafIdx = PreTree::Sample2Leaf(sIdx);
    qLeafExtent[leafIdx]++;
  }

  int totCt = 0;
  for (int i = 0; i < treeSize; i++) {
    int leafExtent = qLeafExtent[i];
    qLeafPos[i] = leafExtent > 0 ? totCt : -1;
    totCt += leafExtent;
  }
  // By this point qLeafExtent[i] > 0 iff the node at tree offset 'i' is a leaf.
  // Similarly, qLeafPos[i] >= 0 iff this is a leaf.

  for (int sIdx = 0; sIdx < bagCount; sIdx++) {
    // ASSERTION:
    //    if (rk > Predictor::nRow)
    //  cout << "Invalid rank:  " << rk << " / " << Predictor::nRow << endl;
    int leafIdx = PreTree::Sample2Leaf(sIdx);
    int rkOff = qLeafPos[leafIdx] + seen[leafIdx];
    qRank[rkOff] = sample2Rank[sIdx];
    qRankCount[rkOff] = sampleReg[sIdx].rowRun;
    seen[leafIdx]++;
  }

  delete [] seen;
}
