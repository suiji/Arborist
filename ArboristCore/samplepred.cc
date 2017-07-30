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
  @brief Initializes immutable field values with category packing.

   @param stagePack holds packed staging values.

  @return void.
*/
void SPNode::Init(const SampleNode &sampleNode, unsigned int _rank) {
  rank = _rank;
  unsigned int ctg = sampleNode.Ref(ySum, sCount);
  sCount = (sCount << ctgShift) | ctg; // Packed representation.
}



/**
   @brief Base class constructor.
 */
SamplePred::SamplePred(unsigned int _nPred, unsigned int _bagCount, unsigned int _bufferSize) : bagCount(_bagCount), nPred(_nPred), bufferSize(_bufferSize), pitchSP(bagCount * sizeof(SamplePred)), pitchSIdx(bagCount * sizeof(unsigned int)), pathIdx(bufferSize) {
  indexBase = new unsigned int[2* bufferSize];
  nodeVec = new SPNode[2 * bufferSize];

  // Coprocessor variants:
  destRestage = new unsigned int[bufferSize];
  destSplit = new unsigned int[bufferSize];
  
  stageOffset.reserve(nPred);
  stageExtent.reserve(nPred);
}


/**
  @brief Base class destructor.
 */
SamplePred::~SamplePred() {
  delete [] nodeVec;
  delete [] indexBase;

  delete [] destRestage;
  delete [] destSplit;
}


/**
   @brief Sets staging boundaries for a given predictor.

   @return voidl
 */
SPNode *SamplePred::StageBounds(unsigned int predIdx, unsigned int safeOffset, unsigned int extent, unsigned int *&smpIdx) {
  stageOffset[predIdx] = safeOffset;
  stageExtent[predIdx] = extent;

  return  Buffers(predIdx, 0, smpIdx);

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
double SamplePred::BlockReplay(unsigned int predIdx, unsigned int sourceBit, unsigned int start, unsigned int extent, BV *replayExpl, std::vector<SumCount> &ctgExpl) {
  unsigned int *idx;
  SPNode *spn = Buffers(predIdx, sourceBit, idx);

  double sumExpl = 0.0;
  if (!ctgExpl.empty()) {
    for (unsigned int spIdx = start; spIdx < start + extent; spIdx++) {
      FltVal ySum;
      unsigned int yCtg;
      unsigned sCount = spn[spIdx].CtgFields(ySum, yCtg);
      ctgExpl[yCtg].Accum(ySum, sCount);
      sumExpl += ySum;
      replayExpl->SetBit(idx[spIdx]);
    }
  }
  else {
    for (unsigned int spIdx = start; spIdx < start + extent; spIdx++) {
      sumExpl += spn[spIdx].YSum();
      replayExpl->SetBit(idx[spIdx]);
    }
  }

  return sumExpl;
}


/**
   @brief Pass-through to Path method.  Looks up reaching cell in appropriate
   buffer.

   @return void.
 */
void SamplePred::Prepath(const IdxPath *idxPath, const unsigned int reachBase[], unsigned int predIdx, unsigned int bufIdx, unsigned int startIdx, unsigned int extent, unsigned int pathMask, bool idxUpdate, unsigned int pathCount[]) {
  Prepath(idxPath, reachBase, idxUpdate, startIdx, extent, pathMask, BufferIndex(predIdx, bufIdx), &pathIdx[StageOffset(predIdx)], pathCount);
}

/**
   @brief Localizes copies of the paths to each index position.  Also
   localizes index positions themselves, if in a node-relative regime.

   @param reachBase is non-null iff index offsets enter as node relative.

   @param idxUpdate is true iff the index is to be updated.

   @param startIdx is the beginning index of the cell.

   @param extent is the count of indices in the cell.

   @param pathMask mask the relevant bits of the path value.

   @param idxVec inputs the index offsets, relative either to the
   current subtree or the containing node and may output an updated
   value.

   @param prePath outputs the (masked) path reaching the current index.

   @param pathCount enumerates the number of times a path is hit.  Only
   client is currently dense packing.

   @return void.
 */
void SamplePred::Prepath(const IdxPath *idxPath, const unsigned int *reachBase, bool idxUpdate, unsigned int startIdx, unsigned int extent, unsigned int pathMask, unsigned int idxVec[], PathT prepath[], unsigned int pathCount[]) const {
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int path = idxPath->IdxUpdate(idxVec[idx], pathMask, reachBase, idxUpdate);
    prepath[idx] = path;
    if (path != NodePath::noPath) {
      pathCount[path]++;
    }
  }
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


// Coprocessor variant.
void SamplePred::IndexRestage(const IdxPath *idxPath, const unsigned int reachBase[], unsigned int predIdx, unsigned int bufIdx, unsigned int idxStart, unsigned int extent, unsigned int pathMask, bool idxUpdate, unsigned int reachOffset[], unsigned int splitOffset[]) {
  unsigned int *idxSource, *idxTarg;
  IndexBuffers(predIdx, bufIdx, idxSource, idxTarg);

  for (unsigned int idx = idxStart; idx < idxStart + extent; idx++) {
    unsigned int sIdx = idxSource[idx];
    unsigned int path = idxPath->IdxUpdate(sIdx, pathMask, reachBase, idxUpdate);
    if (path != NodePath::noPath) {
      unsigned int targOff = reachOffset[path]++;
      idxTarg[targOff] = sIdx; // Semi-regular:  split-level target store.
      destRestage[idx] = targOff;
      //      destSplit[idx] = splitOffset[path]++; // Speculative.
    }
    else {
      destRestage[idx] = bagCount;
      //destSplit[idx] = bagCount;
    }
  }
}

