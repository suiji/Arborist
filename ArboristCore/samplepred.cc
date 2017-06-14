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
#include "path.h"
#include "bv.h"

#include <numeric>

//#include <iostream>
//using namespace std;

unsigned int SPNode::ctgShift = 0;

/**
   @brief Computes a packing width sufficient to hold all (zero-based) response
   category values.

   @param ctgWidth is the response cardinality.

   @return void.
 */
void SPNode::Immutables(unsigned int ctgWidth) {
  unsigned int bits = 1;
  ctgShift = 0;
  // Ctg values are zero-based, so the first power of 2 greater than or
  // equal to 'ctgWidth' has sufficient bits to hold all response values.
  while (bits < ctgWidth) {
    bits <<= 1;
    ctgShift++;
  }
}


/*
**/
void SPNode::DeImmutables() {
  ctgShift = 0;
}


/**
   @brief Base class constructor.
 */
SamplePred::SamplePred(unsigned int _nPred, unsigned int _bagCount, unsigned int _bufferSize) : bagCount(_bagCount), nPred(_nPred), bufferSize(_bufferSize), pitchSP(_bagCount * sizeof(SamplePred)), pitchSIdx(_bagCount * sizeof(unsigned int)), pathIdx(_bufferSize) {
  indexBase = new unsigned int[2* bufferSize];
  nodeVec = new SPNode[2 * bufferSize];
  
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

   @return true iff entire staged set has single rank.  This might be
   a property of the training data or may arise from bagging. 
 */
bool SamplePred::Stage(const std::vector<StagePack> &stagePack, unsigned int predIdx, unsigned int safeOffset, unsigned int extent) {
  stageOffset[predIdx] = safeOffset;
  stageExtent[predIdx] = extent;

  const unsigned int bufIdx = 0;
  unsigned int *smpIdx;
  SPNode *spn = Buffers(predIdx, bufIdx, smpIdx);
  for (unsigned int idx = 0; idx < stagePack.size(); idx++) {
    unsigned int sIdx = spn++->Init(stagePack[idx]);
    *smpIdx++ = sIdx;
  }

  // Singleton iff either:
  //   Dense and all indices implicit.
  //   Not dense and all ranks equal.
  return bagCount == stagePack.size() ? SingleRank(predIdx, bufIdx, 0, bagCount) : (stagePack.size() == 0 ? true : false);
}


/**
   @brief Initializes immutable field values with category packing.

   @param stagePack holds packed staging values.

   @return upacked sample index.
 */
unsigned int SPNode::Init(const StagePack &stagePack) {
  unsigned int sIdx, ctg;
  stagePack.Ref(sIdx, rank, sCount, ctg, ySum);
  sCount = (sCount << ctgShift) | ctg; // Packed representation.
  
  return sIdx;
}


/**
   @brief Walks a block of adjacent SamplePred records associated with
   the explicit component of a split.

   @param predIdx is the argmax predictor for the split.

   @param sourceBit is a dual-buffer toggle.

   @param start is the starting SamplePred index for the split.

   @param extent is the number of SamplePred indices subsumed by the split.

   @param replayExpl sets bits corresponding to explicit indices defined
   by the split.  Indices are either node- or subtree-relative, depending
   on Bottom's current indexing mode.

   @return sum of responses within the block.
 */
double SamplePred::BlockReplay(unsigned int predIdx, unsigned int sourceBit, unsigned int start, unsigned int extent, BV *replayExpl) {
  unsigned int *idx;
  SPNode *spn = Buffers(predIdx, sourceBit, idx);

  double sum = 0.0;
  for (unsigned int spIdx = start; spIdx < start + extent; spIdx++) {
    sum += spn[spIdx].YSum();
    replayExpl->SetBit(idx[spIdx]);
  }

  return sum;
}


/**
   @brief Pass-through to Path method.  Looks up reaching cell in appropriate
   buffer.

   @return void.
 */
void SamplePred::Prepath(const IdxPath *idxPath, const unsigned int reachBase[], unsigned int predIdx, unsigned int bufIdx, unsigned int startIdx, unsigned int extent, unsigned int pathMask, bool idxUpdate, unsigned int pathCount[]) {
  idxPath->Prepath(reachBase, idxUpdate, startIdx, extent, pathMask, BufferIndex(predIdx, bufIdx), &pathIdx[StageOffset(predIdx)], pathCount);
}


/**
   @brief Restages and tabultates rank counts.

   @return void.
 */
void SamplePred::RestageRank(unsigned int predIdx, unsigned int bufIdx, unsigned int startIdx, unsigned int extent, unsigned int reachOffset[], unsigned int rankPrev[], unsigned int rankCount[]) {
  SPNode *source, *targ;
  unsigned int *idxSource, *idxTarg;
  Buffers(predIdx, bufIdx, source, idxSource, targ, idxTarg);

  PathT *pathBlock = &pathIdx[StageOffset(predIdx)];
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int path = pathBlock[idx];
    if (path != NodePath::noPath) {
      SPNode spNode = source[idx];
      unsigned int rank = spNode.Rank();
      rankCount[path] += (rank == rankPrev[path] ? 0 : 1);
      rankPrev[path] = rank;
      unsigned int destIdx = reachOffset[path]++;
      targ[destIdx] = spNode;
      idxTarg[destIdx] = idxSource[idx];
    }
  }
}

