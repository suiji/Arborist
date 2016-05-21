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
SamplePred::SamplePred(unsigned int _nPred, unsigned int _bagCount) : bagCount(_bagCount), nPred(_nPred), bufferSize(_nPred * _bagCount), pitchSP(_bagCount * sizeof(SamplePred)), pitchSIdx(_bagCount * sizeof(unsigned int)) {
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
SamplePred *SamplePred::Factory(unsigned int _nPred, unsigned int _bagCount) {
  SamplePred *samplePred = new SamplePred(_nPred, _bagCount);

  return samplePred;
}


/**
   @brief Initializes column pertaining to a single predictor.

   @param stagePack is a vector of rank/index pairs.

   @param predIdx is the predictor index at which to initialize.

   @return void.
 */
void SamplePred::Stage(const std::vector<StagePack> &stagePack, unsigned int predIdx) {
  unsigned int *smpIdx;
  SPNode *spn = Buffers(predIdx, 0, smpIdx);

  // TODO:  For sparse predictors, stage to DenseRank.

  for (unsigned int idx = 0; idx < stagePack.size(); idx++) {
    unsigned int sIdx = spn++->Init(stagePack[idx]);
    *smpIdx++ = sIdx;
  }
}


/**
   @brief Initializes immutable field values with category packing.

   @param stagePack holds packed staging values.

   @return upacked sample index.
 */
unsigned int SPNode::Init(const StagePack &stagePack) {
  unsigned int sIdx, ctg;
  stagePack.Ref(sIdx, rank, sCount, ctg, ySum);
  sCount = (sCount << runShift) | ctg; // Packed representation.
  
  return sIdx;
}


/**
   @brief Fills in the high and low ranks defining a numerical split.

   @param predIdx is the splitting predictor.

   @param sourceBit (0/1) indicates which buffer holds the current values.

   @param spPos is the index position of the split.

   @param rkLow outputs the low rank.

   @param rkHigh outputs the high rank.

   @return void, with output reference parameters.
 */
void SamplePred::SplitRanks(unsigned int predIdx, unsigned int sourceBit, int spPos, unsigned int &rkLow, unsigned int &rkHigh) {
  SPNode *spn = SplitBuffer(predIdx, sourceBit);
  rkLow = spn[spPos].Rank();
  rkHigh = spn[spPos + 1].Rank();
}


/**
   @brief Maps a block of predictor-associated sample indices to a specified pretree node index.

   @param predIdx is the splitting predictor.

   @param sourceBit (0/1) indicates which buffer holds the current values.

   @param start is the block starting index.

   @param end is the block ending index.

   @param ptId is the pretree node index to which to map the block.

   @return sum of response values associated with each replayed index.
*/
double SamplePred::Replay(unsigned int sample2PT[], unsigned int predIdx, unsigned int sourceBit, int start, int end, unsigned int ptId) {
  unsigned int *sIdx;
  SPNode *spn = Buffers(predIdx, sourceBit, sIdx);

  double sum = 0.0;
  for (int idx = start; idx <= end; idx++) {
    sum += spn[idx].YSum();
    unsigned int sampleIdx = sIdx[idx];
    sample2PT[sampleIdx] = ptId;
  }

  return sum;
}
