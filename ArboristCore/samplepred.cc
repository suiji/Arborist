// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplepred.cc

   @brief Methods to maintain predictor-wise orderings of sampled response indices.

   @author Mark Seligman
 */

#include "samplepred.h"
#include "sample.h"
#include "rowrank.h"

//#include <iostream>
using namespace std;

unsigned int SPNode::runShift = 0;

/**
   @brief Computes a packing width sufficient to hold all (zero-based) response
   category values.

   @param ctgWidth is the response cardinality.

   @return void.
 */
void SPNode::Immutables(unsigned int ctgWidth) {
  unsigned int bits = 1;
  runShift = 0;
  // Ctg values are zero-based, so the first power of 2 greater than or
  // equal to 'ctgWidth' has sufficient bits to hold all response values.
  while (bits < ctgWidth) {
    bits <<= 1;
    runShift++;
  }
}


/*
**/
void SPNode::DeImmutables() {
  runShift = 0;
}


/**
   @brief Base class constructor.
 */
SamplePred::SamplePred(unsigned int _nRow, unsigned int _nPred, unsigned int _bagCount) : nRow(_nRow), bagCount(_bagCount), nPred(_nPred), bufferSize(_nPred * _bagCount), pitchSP(_bagCount * sizeof(SamplePred)), pitchSIdx(_bagCount * sizeof(unsigned int)) {
  sampleIdx = new unsigned int[2* bufferSize];
  nodeVec = new SPNode[2 * bufferSize];
}


/**
  @brief Base class destructor.
 */
SamplePred::~SamplePred() {
  delete [] nodeVec;
  delete [] sampleIdx;
}


/**
   @brief Static entry for sample staging.

   @return SamplePred object for tree.
 */
SamplePred *SamplePred::Factory(const RowRank *rowRank, const SampleNode sampleNode[], const int sIdxRow[], unsigned int _nRow, unsigned int _nPred, unsigned int _bagCount) {
  SamplePred *samplePred = new SamplePred(_nRow, _nPred, _bagCount);
  samplePred->Stage(rowRank, sampleNode, sIdxRow);

  return samplePred;
}


/**
   @brief Loops through the predictors to stage.

   @return void.
 */
void SamplePred::Stage(const RowRank *rowRank, const SampleNode sampleNode[], const int sIdxRow[]) {  
  unsigned int predIdx;

#pragma omp parallel default(shared) private(predIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (predIdx = 0; predIdx < nPred; predIdx++) {
      Stage(rowRank, sampleNode, sIdxRow, predIdx);
    }
  }
}


/**
   @brief Stages SamplePred objects in non-decreasing predictor order.

   @param predIdx is the predictor index.

   @return void.
*/
void SamplePred::Stage(const RowRank *rowRank, const SampleNode sampleNode[], const int sIdxRow[], int predIdx) {
  unsigned int *smpIdx;
  SPNode *spn = Buffers(predIdx, 0, smpIdx);

  // TODO:  For sparse predictors, stage to DenseRank.
  for (unsigned int idx = 0; idx < nRow; idx++) {
    unsigned int rank;
    unsigned int row = rowRank->Lookup(predIdx, idx, rank);
    int sIdx = sIdxRow[row];
    if (sIdx >= 0) {
      *smpIdx++ = sIdx;
      spn++->Init(&sampleNode[sIdx], rank);
    }
  }
}


/**
   @brief Initializes immutable field values with category packing.

   @param sample holds sampled values.

   @param sIdx is the sample index.

   @param sn contains the sampled values.

   @param _rank is the predictor rank at the sampled row.

   @return void.
 */
void SPNode::Init(const SampleNode *sampleNode, unsigned int _rank) {
  unsigned int ctg = sampleNode->Ref(ySum, sCount);
  sCount = (sCount << runShift) | ctg; // Packed representation.
  rank = _rank;
}


/**
   @brief Fills in the high and low ranks defining a numerical split.

   @param predIdx is the splitting predictor.

   @param level is the current level.

   @param spPos is the index position of the split.

   @param rkLow outputs the low rank.

   @param rkHigh outputs the high rank.

   @return void, with output reference parameters.
 */
void SamplePred::SplitRanks(int predIdx, unsigned int level, int spPos, unsigned int &rkLow, unsigned int &rkHigh) {
  SPNode *spn = SplitBuffer(predIdx, level);
  rkLow = spn[spPos].Rank();
  rkHigh = spn[spPos + 1].Rank();
}


/**
   @brief Maps a block of predictor-associated sample indices to a specified pretree node index.

   @param predIdx is the splitting predictor.

   @param level is the current level.

   @param start is the block starting index.

   @param end is the block ending index.

   @param ptId is the pretree node index to which to map the block.

   @return sum of response values associated with each replayed index.
*/
double SamplePred::Replay(unsigned int sample2PT[], int predIdx, unsigned int level, int start, int end, unsigned int ptId) {
  unsigned int *sIdx;
  SPNode *spn = Buffers(predIdx, level, sIdx);

  double sum = 0.0;
  for (int idx = start; idx <= end; idx++) {
    sum += spn[idx].YSum();
    unsigned int sampleIdx = sIdx[idx];
    sample2PT[sampleIdx] = ptId;
  }

  return sum;
}
