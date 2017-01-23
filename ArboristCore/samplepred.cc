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
#include <numeric>

//#include <iostream>
//using namespace std;

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
SamplePred::SamplePred(unsigned int _nPred, unsigned int _bagCount, unsigned int _bufferSize) : bagCount(_bagCount), nPred(_nPred), bufferSize(_bufferSize), pitchSP(_bagCount * sizeof(SamplePred)), pitchSIdx(_bagCount * sizeof(unsigned int)), rel2Sample(std::vector<unsigned int>(bagCount)) {
  indexBase = new unsigned int[2* bufferSize];
  nodeVec = new SPNode[2 * bufferSize];

  std::iota(rel2Sample.begin(), rel2Sample.end(), 0);
  stageOffset.reserve(nPred);
  stageExtent.reserve(nPred);
}


/**
  @brief Base class destructor.
 */
SamplePred::~SamplePred() {
  delete [] nodeVec;
  delete [] indexBase;
}


/**
   @brief Static entry for sample staging.

   @return SamplePred object for tree.
 */
SamplePred *SamplePred::Factory(unsigned int _nPred, unsigned int _bagCount, unsigned int _bufferSize) {
  SamplePred *samplePred = new SamplePred(_nPred, _bagCount, _bufferSize);

  return samplePred;
}


/**
   @brief Initializes column pertaining to a single predictor.

   @param stagePack is a vector of rank/index pairs.

   @param predIdx is the predictor index at which to initialize.

   @return void.
 */
void SamplePred::Stage(const std::vector<StagePack> &stagePack, unsigned int predIdx, unsigned int safeOffset, unsigned int extent) {
  stageOffset[predIdx] = safeOffset;
  stageExtent[predIdx] = extent;

  unsigned int *smpIdx;
  SPNode *spn = Buffers(predIdx, 0, smpIdx);
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
   @brief Maps a block of sample indices from a splitting pair to the pretree node in whose sample set the indices now, as a result of splitting, reside.

   @param predIdx is the splitting predictor.

   @param sourceBit (0/1) indicates which buffer holds the current values.

   @param start is the block starting index.

   @param end is the block ending index.

   @param ptId is the pretree node index to which to map the block.

   @param sample2PT outputs the preTree node to which a sample belongs.

   @return sum of response values associated with each replayed index.
*/
double SamplePred::Replay(unsigned int predIdx, unsigned int sourceBit, unsigned int start, unsigned int end, unsigned int ptId, std::vector<unsigned int> &sample2PT) {
  unsigned int *relIdx;
  SPNode *spn = Buffers(predIdx, sourceBit, relIdx);

  double sum = 0.0;
  for (unsigned int idx = start; idx <= end; idx++) {
    sum += spn[idx].YSum();
    unsigned int sampleIdx = rel2Sample[relIdx[idx]];
    sample2PT[sampleIdx] = ptId;
  }

  return sum;
}
