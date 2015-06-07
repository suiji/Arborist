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
#include "callback.h"
#include "samplepred.h"
#include "predictor.h"
#include "splitpred.h"

//#include <iostream>
using namespace std;

// Simulation-invariant values.
//
unsigned int Sample::nRow = 0;
int Sample::nPred = -1;
int Sample::nSamp = -1;

int SampleCtg::ctgWidth = 0;

/**
 @brief Lights off initializations needed for sampling.

 @param _nRow is the number of response/observation rows.

 @param _nSamp is the number of samples.

 @param _nPred is the number of predictors.

 @return void.
*/
void Sample::Immutables(unsigned int _nRow, int _nPred, int _nSamp) {
  nRow = _nRow;
  nPred = _nPred;
  nSamp = _nSamp;
}


/**
   @return void.
 */
void SampleReg::Immutables() {
  SplitPred::ImmutablesReg(nRow, nSamp);
}


/**
   @return void.
 */
void SampleCtg::Immutables(int _ctgWidth) {
  ctgWidth = _ctgWidth;
  SplitPred::ImmutablesCtg(nRow, nSamp, ctgWidth);
}

/**
   @brief Finalizer.

   @return void.
*/
void Sample::DeImmutables() {
  nRow = 0;
  nSamp = nPred = -1;
  SamplePred::DeImmutables();

  // Only one of these two class has been set, but no harm in resetting both:
  SPReg::DeImmutables();
  SPCtg::DeImmutables();
}

/**
   @brief Counts instances of each row index in sampled set.

   @return vector of sample counts, by row.  0 <=> OOB.
*/
int *Sample::CountRows() {
  int *sCountRow = new int[nRow]; // Needed until Stage() finishes.
  for (unsigned int row= 0; row < nRow; row++)
    sCountRow[row] = 0;

  // Counts occurrences of the rank associated with each target 'row' of the
  // sampling vector.
  //
  int *rvRow = new int[nRow];
  CallBack::SampleRows(rvRow);
  for (int i = 0; i < nSamp; i++) {
    int row = rvRow[i];
    sCountRow[row]++;
  }
  delete [] rvRow;

  return sCountRow;
}


/**
   @brief Constructor.
 */
SampleReg::SampleReg() {
  sample2Rank = new int[nSamp]; // Lives until scoring.
  sampleReg = new SampleNode[nSamp]; //  " "
}

/**
   @brief Inverts the randomly-sampled vector of rows.

   @param y is the response vector.

   @param row2Rank is rank of each sampled row.

   @param inBag is a bit vector indicating whether a row is in-bag in this tree.

   @param samplePred is the SamplePred object associated with this tree.

   @param bagSum is the sum of in-bag sample values.  Used for initializing index tree root.

   @return count of in-bag samples.
*/
// The number of unique rows is the size of the bag.  With compression,
// however, the resulting number of samples is smaller than the bag count.
//
// Returns the sum of sampled response values for intiializition of topmost
// accumulator.
//
int SampleReg::Stage(const double y[], const int row2Rank[], const PredOrd *predOrd, unsigned int inBag[], SamplePred *samplePred, SplitPred *&splitPred, double &bagSum) {  
  int *sCountRow = CountRows();
  int *sIdxRow = new int[nRow]; // Index of row in sample vector.
  const unsigned int slotBits = 8 * sizeof(unsigned int);
  
  bagSum = 0.0;
  int slot = 0;
  int idx = 0;
  for (unsigned int base = 0; base < nRow; base += slotBits, slot++) {
    // Enables lookup by row, for Stage(), or index, for LevelMap.
    //
    unsigned int bits = 0;
    unsigned int mask = 1;
    unsigned int supRow = nRow < base + slotBits ? nRow : base + slotBits;
    for (unsigned int row = base; row < supRow; row++, mask <<= 1) {
      int sCount = sCountRow[row];
      if (sCount > 0) {
        double val = sCount * y[row];
        bagSum += val;
        sampleReg[idx].sum = val;
        sampleReg[idx].rowRun = sCount;
        // Only client is quantile regression, but cheap to compute here:
        sample2Rank[idx] = row2Rank[row];
	
        sIdxRow[row] = idx++;
	bits |= mask;
      }
    }
    inBag[slot] = bits;
  }
  bagCount = idx;
  samplePred->StageReg(predOrd, sampleReg, sCountRow, sIdxRow);
  splitPred = SplitPred::FactoryReg(samplePred);

  delete [] sCountRow;
  delete [] sIdxRow;

  return bagCount;
}


/**
   @brief Constructor.

   @param ctgWidth is the response cardinality.

 */
SampleCtg::SampleCtg() {
  sampleCtg = new SampleNodeCtg[nSamp]; // Lives until scoring.
}


/**
   @brief Samples the response, sets in-bag bits and stages.

   @param yCtg is the response vector.

   @param y is the proxy response vector.

   @param inBag is a bit vector indicating in-bag rows.

   @param samplePred is the staged SamplePred object.

   @param bagSum is the sum of in-bag response values, used for splitting the root index node.

   @return count of in-bag samples.
*/
// Same as for regression case, but allocates and sets 'ctg' value, as well.
// Full row count is used to avoid the need to rewalk.
//
int SampleCtg::Stage(const int yCtg[], const double y[], const PredOrd *predOrd, unsigned int inBag[], SamplePred *samplePred, SplitPred *&splitPred, double &bagSum) {
  int *sCountRow = CountRows();
  int *sIdxRow = new int[nRow];
  const unsigned int slotBits = 8 * sizeof(unsigned int);

  int idx = 0;
  bagSum = 0.0;
  bagCount = 0;
  int slot = 0;
  for (unsigned int base = 0; base < nRow; base += slotBits, slot++) {
    unsigned int bits = 0;
    unsigned int mask = 1;
    unsigned int supRow = nRow < base + slotBits ? nRow : base + slotBits;
    for (unsigned int row = base; row < supRow; row++, mask <<= 1) {
      int sCount = sCountRow[row];
      if (sCount > 0) {
        double val = sCount * y[row];
	bagSum += val;
        sampleCtg[idx].sum = val;
        sampleCtg[idx].rowRun = sCount;
        sampleCtg[idx].ctg = yCtg[row];
        sIdxRow[row] = idx++;
        bits |= mask;
      }
    }
    inBag[slot] = bits;
  }
  bagCount = idx;
  samplePred->StageCtg(predOrd, sampleCtg, sCountRow, sIdxRow);
  splitPred = SplitPred::FactoryCtg(samplePred, sampleCtg);

  delete [] sCountRow;
  delete [] sIdxRow;

  return bagCount;
}


/**
   @brief Derives scores for regression tree.

   @param frontierMap maps sample id to pretree terminal id.

   @param treeHeight is the number of nodes in the pretree.

   @param score outputs the computed scores.

   @return void, with output parameter vector.
*/
// Walks the sample set, accumulating value sums for the associated leaves.  Score
// is the sample mean.  These values could also be computed by passing sums down the
// pre-tree and pulling them from terminal nodes.
//
void SampleReg::Scores(const int frontierMap[], int treeHeight, double score[]) {
  int *sCount = new int[treeHeight];
  for (int pt = 0; pt < treeHeight; pt++) {
    score[pt] = 0.0;
    sCount[pt]= 0;
  }

  for (int i = 0; i < bagCount; i++) {
    int leafIdx = frontierMap[i];
    score[leafIdx] += sampleReg[i].sum;
    sCount[leafIdx] += sampleReg[i].rowRun;
  }

  for (int pt = 0; pt < treeHeight; pt++) {
    if (sCount[pt] > 0)
      score[pt] /= sCount[pt];
  }

  delete [] sCount;
}


/**
   @brief Derives scores for categorical tree.

   @param sampleMap maps sample id to pretree terminal id.

   @param treeHeight is the number of nodes in the pretree.

   @param score outputs the computed scores.

   @return void, with output parameter vector.
*/

void SampleCtg::Scores(const int frontierMap[], int treeHeight, double score[]) {
  double *leafWS = new double[ctgWidth * treeHeight];

  for (int i = 0; i < ctgWidth * treeHeight; i++)
    leafWS[i] = 0.0;

  // Irregular access.  Needs the ability to map sample indices to the factors and
  // weights with which they are associated.
  //
  for (int i = 0; i < bagCount; i++) {
    int leafIdx = frontierMap[i];
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
    // For now, upcasts score to double, for compatability with DecTree.
    score[leafIdx] = argMaxWeight;
  }

  // ASSERTION:
  //  Can count nonterminals and verify #nonterminals == treeHeight - leafCount

  delete [] leafWS;
}


SampleCtg::~SampleCtg() {
  delete [] sampleCtg;
}


/**
   @brief Clears per-tree information.

   @return void.
 */
SampleReg::~SampleReg() {
  delete [] sampleReg;
  delete [] sample2Rank;
}


/**
   @brief Derives and copies quantile leaf information.

   @param frontierMap[] maps sample index to its pre-tree terminal id.

   @param treeSize is the height of the pretree.

   @param qLeafPos outputs quantile leaf offsets; vector length treeSize.

   @param qLeafExtent outputs quantile leaf sizes; vector length treeSize.

   @param rank outputs quantile leaf ranks; vector length bagCount.

   @param rankCount outputs rank multiplicities; vector length bagCount.

   @return void, with output parameter vectors.
 */
void SampleReg::Quantiles(const int frontierMap[], int treeSize, int qLeafPos[], int qLeafExtent[], int qRank[], int qRankCount[]) {
  // Must be wide enough to access all tree offsets.
  int *seen = new int[treeSize];
  for (int i = 0; i < treeSize; i++) {
    seen[i] = 0;
    qLeafExtent[i] = 0;
  }
  for (int sIdx = 0; sIdx < bagCount; sIdx++) {
    int leafIdx = frontierMap[sIdx];
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
    int leafIdx = frontierMap[sIdx];
    int rkOff = qLeafPos[leafIdx] + seen[leafIdx];
    qRank[rkOff] = sample2Rank[sIdx];
    qRankCount[rkOff] = sampleReg[sIdx].rowRun;
    seen[leafIdx]++;
  }

  delete [] seen;
}


/**
   @brief Stub:  should be unreachable.
 */
void SampleCtg::Quantiles(const int frontierMap[], int treeSize, int qLeafPos[], int qLeafExtent[], int qRank[], int qRankCount[]) {
  // ASSERTION:
  // Should never get here.
}
