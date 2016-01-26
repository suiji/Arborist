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
#include "bv.h"
#include "callback.h"
#include "rowrank.h"
#include "samplepred.h"
#include "splitpred.h"

//#include <iostream>
//using namespace std;

// Simulation-invariant values.
//
unsigned int Sample::nRow = 0;
unsigned int Sample::nPred = 0;
int Sample::nSamp = -1;

unsigned int SampleCtg::ctgWidth = 0;
double SampleCtg::forestScale = 0.0;

/**
 @brief Lights off initializations needed for sampling.

 @param _nRow is the number of response/observation rows.

 @param _nSamp is the number of samples.

 @return void.
*/
void Sample::Immutables(unsigned int _nRow, unsigned int _nPred, int _nSamp, double _feSampleWeight[], bool _withRepl, unsigned int _ctgWidth, int _nTree) {
  nRow = _nRow;
  nPred = _nPred;
  nSamp = _nSamp;
  CallBack::SampleInit(nRow, _feSampleWeight, _withRepl);
  if (_ctgWidth > 0)
    SampleCtg::Immutables(_ctgWidth, _nTree);
}


/**
   @return void.
 */
void SampleCtg::Immutables(unsigned int _ctgWidth, int _nTree) {
  ctgWidth = _ctgWidth;
  forestScale = 1.0 / (nRow * _nTree);
}


/**
 */
void SampleCtg::DeImmutables() {
  ctgWidth = 0;
  forestScale = 0.0;
}


/**
   @brief Finalizer.

   @return void.
*/
void Sample::DeImmutables() {
  nRow = 0;
  nPred = 0;
  nSamp = -1;
  SampleCtg::DeImmutables();
}


Sample::Sample() {
  inBag = new unsigned int [BV::LengthAlign(nRow)];
  sampleNode = new SampleNode[nSamp]; // Lives until scoring.
}


Sample::~Sample() {
  delete [] inBag;
  delete [] sampleNode;
  delete samplePred;
  delete splitPred;
}


/**
   @brief Samples and enumerates instances of each row index.
   'bagCount'.

   @param sCountRow[] holds the row counts:  0 <=> OOB.

   @param sIdxRow[] row index into sample vector:  -1 <=> OOB.

   @return bagCount, plus output vectors.
*/
int Sample::CountRows(int sCountRow[], int sIdxRow[]) {
  for (unsigned int row = 0; row < nRow; row++) {
    sCountRow[row] = 0;
    sIdxRow[row] = -1;
  }

  // Counts occurrences of the rank associated with each target 'row' of the
  // sampling vector.
  //
  int *rvRow = new int[nSamp];
  CallBack::SampleRows(nSamp, rvRow);
  for (int i = 0; i < nSamp; i++) {
    int row = rvRow[i];
    sCountRow[row]++;
  }
  delete [] rvRow;

  int idx = 0;
  for (unsigned int row = 0; row < nRow; row++) {
    if (sCountRow[row] > 0)
      sIdxRow[row] = idx++;
  }

  return idx;
}


/**
   @brief Static entry for classification.
 */
SampleCtg *SampleCtg::Factory(const double y[], const RowRank *rowRank,  const unsigned int yCtg[]) {
  SampleCtg *sampleCtg = new SampleCtg();
  sampleCtg->Stage(yCtg, y, rowRank);

  return sampleCtg;
}


/**
   @brief Static entry for regression response.

 */
SampleReg *SampleReg::Factory(const double y[], const RowRank *rowRank, const unsigned int row2Rank[]) {
  SampleReg *sampleReg = new SampleReg();
  sampleReg->Stage(y, row2Rank, rowRank);

  return sampleReg;
}


/**
   @brief Constructor.
 */
SampleReg::SampleReg() : Sample() {
  sample2Rank = new unsigned int[nSamp]; // Lives until scoring.
}



/**
   @brief Inverts the randomly-sampled vector of rows.

   @param y is the response vector.

   @param row2Rank is rank of each sampled row.

   @param samplePred is the SamplePred object associated with this tree.

   @param bagSum is the sum of in-bag sample values.  Used for initializing index tree root.

   @return count of in-bag samples.
*/
void SampleReg::Stage(const double y[], const unsigned int row2Rank[], const RowRank *rowRank) {
  int *sIdxRow = Sample::PreStage(y);

  // Only client is quantile regression.
  for (unsigned int row = 0; row < nRow; row++) {
    int sIdx = sIdxRow[row];
    if (sIdx >= 0) {
      sample2Rank[sIdx] = row2Rank[row];
    }
  }
  samplePred = SamplePred::Factory(rowRank, sampleNode, sIdxRow, nRow, nPred, bagCount);
  delete [] sIdxRow;

  splitPred = SplitPred::FactoryReg(samplePred);
}


/**
   @brief Constructor.
 */
SampleCtg::SampleCtg() : Sample() {
}


/**
   @brief Samples the response, sets in-bag bits and stages.

   @param yCtg is the response vector.

   @param y is the proxy response vector.

   @return void, with output vector parameter.
*/
// Same as for regression case, but allocates and sets 'ctg' value, as well.
// Full row count is used to avoid the need to rewalk.
//
void SampleCtg::Stage(const unsigned int yCtg[], const double y[], const RowRank *rowRank) {
  int *sIdxRow = Sample::PreStage(y, yCtg);
  samplePred = SamplePred::Factory(rowRank, sampleNode, sIdxRow, nRow, nPred, bagCount);
  delete [] sIdxRow;

  splitPred = SplitPred::FactoryCtg(samplePred, sampleNode);
}


/**
   @brief Sets the stage, so to speak, for a newly-sampled response set.

   @param y is the proxy / response:  classification / summary.

   @param yCtg is true response / zero:  classification / regression.

   @return vector of compressed indices into sample data structures.
 */
int *Sample::PreStage(const double y[], const unsigned int yCtg[]) {
  int *sIdxRow = new int[nRow];
  int *sCountRow = new int[nRow];
  bagCount = CountRows(sCountRow, sIdxRow);
  unsigned int slotBits = BV::SlotBits();

  bagSum = 0.0;
  int slot = 0;
  for (unsigned int base = 0; base < nRow; base += slotBits, slot++) {
    unsigned int bits = 0;
    unsigned int mask = 1;
    unsigned int supRow = nRow < base + slotBits ? nRow : base + slotBits;
    for (unsigned int row = base; row < supRow; row++, mask <<= 1) {
      int sIdx = sIdxRow[row];
      if (sIdx >= 0) {
	int sCount = sCountRow[row];
        double val = sCount * y[row];
	sampleNode[sIdx].Set(val, sCount, yCtg == 0 ? 0 : yCtg[row]);
	bagSum += val;
        bits |= mask;
      }
    }
    inBag[slot] = bits;
  }
  delete [] sCountRow;

  return sIdxRow;
}


/**
   @brief Derives and copies regression leaf information.

   @param nonTerm is zero iff forest index is at leaf.

   @param leafExtent gives leaf width at forest index.

   @param rank outputs leaf ranks; vector length bagCount.

   @param sCount outputs sample counts; vector length bagCount.

   @return bag count, with output parameter vectors.
 */
void SampleReg::Leaves(const int frontierMap[], int treeHeight, int leafExtent[], double score[], const int nonTerm[], unsigned int rank[], unsigned int sCount[]) {
  Scores(frontierMap, treeHeight, score);
  LeafExtent(frontierMap, leafExtent);

  int *leafPos = LeafPos(nonTerm, leafExtent, treeHeight);
  int *seen = new int[treeHeight];
  for (int i = 0; i < treeHeight; i++) {
    seen[i] = 0;
  }
  for (int sIdx = 0; sIdx < bagCount; sIdx++) {
    int leafIdx = frontierMap[sIdx];
    int rkOff = leafPos[leafIdx] + seen[leafIdx]++;
    sCount[rkOff] = sampleNode[sIdx].SCount();
    rank[rkOff] = sample2Rank[sIdx];
  }

  delete [] seen;
  delete [] leafPos;
}


/**
   @brief Derives scores for regression tree:  intialize, accumulate, divide.

   @param frontierMap maps sample id to pretree terminal id.

   @param treeHeight is the number of nodes in the pretree.

   @param score outputs the computed scores.

   @return void, with output parameter vector.
*/
void SampleReg::Scores(const int frontierMap[], int treeHeight, double score[]) {
  int *sCount = new int[treeHeight];
  for (int ptIdx = 0; ptIdx < treeHeight; ptIdx++) {
    sCount[ptIdx] = 0;
  }

  // score[] is 0.0 for leaves:  only nonterminals have been overwritten.
  //
  for (int i = 0; i < bagCount; i++) {
    int leafIdx = frontierMap[i];
    FltVal _sum;
    unsigned int _sCount;
    (void) Ref(i, _sum, _sCount);
    score[leafIdx] += _sum;
    sCount[leafIdx] += _sCount;
  }

  for (int ptIdx = 0; ptIdx < treeHeight; ptIdx++) {
    if (sCount[ptIdx] > 0)
      score[ptIdx] /= sCount[ptIdx];
  }

  delete [] sCount;
}


/**
   @brief Sets node counts on each leaf.

   @param frontierMap maps samples to tree indices.

   @param leafExtent outputs the node counts by node index.

   @return void with output reference vector.
 */
void Sample::LeafExtent(const int frontierMap[], int leafExtent[]) {
  for (int i = 0; i < bagCount; i++) {
    int leafIdx = frontierMap[i];
    leafExtent[leafIdx]++;
  }
}


/**
   @brief Defines starting positions for ranks associated with a given leaf.

   @param treeHeight is the height of the current tree.

   @param nonTerm is zero iff leaf reference.

   @param leafExtent enumerates leaf widths.

   @return vector of leaf sample offsets, by tree index.
 */
int *SampleReg::LeafPos(const int nonTerm[], const int leafExtent[], int treeHeight) {
  int totCt = 0;
  int *leafPos = new int[treeHeight];
  for (int i = 0; i < treeHeight; i++) {
    if (nonTerm[i] == 0) {
      leafPos[i] = totCt;
      totCt += leafExtent[i];
    }
    else
      leafPos[i] = -1;
  }
  // ASSERTION:  totCt == bagCount
  // By this point leafPos[i] >= 0 iff this 'i' references is a leaf.

  return leafPos;
}


void SampleCtg::Leaves(const int frontierMap[], int treeHeight, int leafExtent[], double score[], const int nonTerm[], double *leafWeight) {
  LeafExtent(frontierMap, leafExtent);
  LeafWeight(frontierMap, nonTerm, treeHeight, leafWeight);
  Scores(leafWeight, treeHeight, nonTerm, score);
}


/**
   @brief Derives scores for categorical tree.

   @param leafWeight holds per-leaf category weights.

   @param treeHeight is the number of nodes in the pretree.

   @param nonTerm is nonzero iff the indexed node is nonterminal.

   @param score outputs the computed scores.

   @return void, with output reference parameter.
*/
void SampleCtg::Scores(double *leafWeight, int treeHeight, const int nonTerm[], double score[]) {

  // Category weights are jittered, making ties highly unlikely.
  //
  for (int idx = 0; idx < treeHeight; idx++) {
    if (nonTerm[idx] != 0)
      continue;
    double *leafBase = leafWeight + idx * ctgWidth;
    double maxWeight = 0.0;
    unsigned int argMax = 0; // Zero will be default score/category.
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      double thisWeight = leafBase[ctg];
      if (thisWeight > maxWeight) {
	maxWeight = thisWeight;
	argMax = ctg;
      }
    }
    
    // Jitters category value by row/tree-scaled sum.
    score[idx] = argMax + maxWeight * forestScale;
  }

  // ASSERTION:
  //  Can count nonterminals and verify #nonterminals == treeHeight - leafCount
}


/**
   @brief Accumulates sums of samples associated with each leaf.

   @param frontierMap associates samples with leaf indices.
   
   @leafWeight output the leaf weights, by category.

   @return void, with reference output vector.
 */
void SampleCtg::LeafWeight(const int frontierMap[], const int nonTerm[], int treeHeight, double *leafWeight) {
  double *leafSum = new double[treeHeight];
  for (int i = 0; i < treeHeight; i++)
    leafSum[i] = 0.0;
  
  for (int i = 0; i < bagCount; i++) {
    int leafIdx = frontierMap[i]; // Irregular access.
    FltVal sum;
    unsigned int dummy;
    int ctg = Ref(i, sum, dummy);
    leafSum[leafIdx] += sum;
    leafWeight[leafIdx * ctgWidth + ctg] += sum;
  }

  // Normalizes weights for probabilities.
  for (int i = 0; i < treeHeight; i++) {
    if (nonTerm[i] == 0) {
      double recipSum = 1.0 / leafSum[i];
      for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
	leafWeight[i * ctgWidth + ctg] *= recipSum;
      }
    }
  }
  delete [] leafSum;
}


SampleCtg::~SampleCtg() {
}


/**
   @brief Clears per-tree information.

   @return void.
 */
SampleReg::~SampleReg() {
  delete [] sample2Rank;
}
