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
#include "splitcand.h"
#include "sample.h"
#include "rowrank.h"
#include "path.h"
#include "bv.h"
#include "level.h"

#include <numeric>


/**
   @brief Base class constructor.
 */
SamplePred::SamplePred(unsigned int _nPred,
                       unsigned int _bagCount,
                       unsigned int _bufferSize) :
  nPred(_nPred),
  bufferSize(_bufferSize),
  pitchSP(bagCount * sizeof(SamplePred)),
  pitchSIdx(_bagCount * sizeof(unsigned int)),
  pathIdx(bufferSize),
  stageOffset(nPred),
  stageExtent(nPred),
  bagCount(_bagCount) {
  indexBase = new unsigned int[2* bufferSize];
  nodeVec = new SampleRank[2 * bufferSize];

  // Coprocessor variants:
  destRestage = new unsigned int[bufferSize];
  destSplit = new unsigned int[bufferSize];
  
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

   @return 
 */
SampleRank *SamplePred::StageBounds(unsigned int predIdx,
                                    unsigned int safeOffset,
                                    unsigned int extent,
                                    unsigned int*& smpIdx) {
  stageOffset[predIdx] = safeOffset;
  stageExtent[predIdx] = extent;

  return buffers(predIdx, 0, smpIdx);

}

// TODO:  Merge
void SamplePred::Stage(const class RRNode *rrNode,
                       unsigned int rrTot,
                       const vector<class SampleNux>& sampleNode,
                       const vector<unsigned int>& row2Sample,
                       vector<StageCount>& stageCount) {}


unsigned int SamplePred::Stage(const vector<SampleNux>&
                               sampleNode,
                               const RRNode *rrPred,
                               const vector<unsigned int>& row2Sample,
                               unsigned int explMax,
                               unsigned int predIdx,
                               unsigned int safeOffset,
                               unsigned int extent,
                               bool& isSingleton) {
  unsigned int *smpIdx;
  SampleRank *spn = StageBounds(predIdx, safeOffset, extent, smpIdx);
  unsigned int expl = 0;
  for (unsigned int idx = 0; idx < explMax; idx++) {
    Stage(sampleNode, rrPred[idx], row2Sample, spn, smpIdx, expl);
  }

  isSingleton = singleton(expl, predIdx);
  return expl;
}


/**
   @brief Fills in sampled response summary and rank information associated
   with an RRNode reference.

   @param rrNode summarizes an element of the compressed design matrix.

   @param spn is the cell to initialize.

   @param smpIdx is the associated sample index.

   @param expl accumulates the current explicitly staged offset.

   @return void.
 */
void SamplePred::Stage(const vector<SampleNux> &sampleNode,
		       const RRNode &rrNode,
		       const vector<unsigned int> &row2Sample,
		       SampleRank *spn,
		       unsigned int *smpIdx,
		       unsigned int &expl) {
  unsigned int row, rank;
  rrNode.Ref(row, rank);

  unsigned int sIdx = row2Sample[row];
  if (sIdx < bagCount) {
    spn[expl].join(rank, sampleNode[sIdx]);
    smpIdx[expl] = sIdx;
    expl++;
  }
}

double SamplePred::blockReplay(const SplitCand& argMax,
                               BV* replayExpl,
                               vector<SumCount>& ctgExpl) {
  return blockReplay(argMax, argMax.getExplicitBranchStart(), argMax.getExplicitBranchExtent(), replayExpl, ctgExpl);
}


double SamplePred::blockReplay(const SplitCand& argMax,
                               unsigned int blockStart,
                               unsigned int blockExtent,
                               BV* replayExpl,
                               vector<SumCount> &ctgExpl) {
  unsigned int* idx;
  SampleRank* spn = buffers(argMax.getPredIdx(), argMax.getBufIdx(), idx);

  double sumExpl = 0.0;
  if (!ctgExpl.empty()) {
    for (unsigned int spIdx = blockStart; spIdx < blockStart + blockExtent; spIdx++) {
      FltVal ySum;
      unsigned int yCtg;
      unsigned sCount = spn[spIdx].ctgFields(ySum, yCtg);
      ctgExpl[yCtg].Accum(ySum, sCount);
      sumExpl += ySum;
      replayExpl->setBit(idx[spIdx]);
    }
  }
  else {
    for (unsigned int spIdx = blockStart; spIdx < blockStart + blockExtent; spIdx++) {
      sumExpl += spn[spIdx].getYSum();
      replayExpl->setBit(idx[spIdx]);
    }
  }

  return sumExpl;
}


/**
   @brief Pass-through to Path method.  Looks up reaching cell in appropriate
   buffer.

   @return void.
 */
void SamplePred::prepath(const IdxPath *idxPath,
                         const unsigned int reachBase[],
                         unsigned int predIdx,
                         unsigned int bufIdx,
                         unsigned int startIdx,
                         unsigned int extent,
                         unsigned int pathMask,
                         bool idxUpdate,
                         unsigned int pathCount[]) {
  prepath(idxPath, reachBase, idxUpdate, startIdx, extent, pathMask, bufferIndex(predIdx, bufIdx), &pathIdx[getStageOffset(predIdx)], pathCount);
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
void SamplePred::prepath(const IdxPath *idxPath,
                         const unsigned int *reachBase,
                         bool idxUpdate,
                         unsigned int startIdx,
                         unsigned int extent,
                         unsigned int pathMask,
                         unsigned int idxVec[],
                         PathT prepath[],
                         unsigned int pathCount[]) const {
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    PathT path = idxPath->update(idxVec[idx], pathMask, reachBase, idxUpdate);
    prepath[idx] = path;
    if (NodePath::isActive(path)) {
      pathCount[path]++;
    }
  }
}


/**
   @brief Virtual pass-through to appropriate restaging method.

   @return void.
 */
void SamplePred::Restage(Level *levelBack,
                         Level *levelFront,
                         const SPPair &mrra,
                         unsigned int bufIdx) {
  levelBack->rankRestage(this, mrra, levelFront, bufIdx);
}


/**
   @brief Restages and tabultates rank counts.

   @return void.
 */
void SamplePred::rankRestage(unsigned int predIdx,
                             unsigned int bufIdx,
                             unsigned int startIdx,
                             unsigned int extent,
                             unsigned int reachOffset[],
                             unsigned int rankPrev[],
                             unsigned int rankCount[]) {
  SampleRank *source, *targ;
  unsigned int *idxSource, *idxTarg;
  buffers(predIdx, bufIdx, source, idxSource, targ, idxTarg);

  PathT *pathBlock = &pathIdx[getStageOffset(predIdx)];
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int path = pathBlock[idx];
    if (NodePath::isActive(path)) {
      SampleRank spNode = source[idx];
      unsigned int rank = spNode.getRank();
      rankCount[path] += (rank == rankPrev[path] ? 0 : 1);
      rankPrev[path] = rank;
      unsigned int destIdx = reachOffset[path]++;
      targ[destIdx] = spNode;
      idxTarg[destIdx] = idxSource[idx];
    }
  }
}


void SamplePred::indexRestage(const IdxPath *idxPath,
                              const unsigned int reachBase[],
                              unsigned int predIdx,
                              unsigned int bufIdx,
                              unsigned int idxStart,
                              unsigned int extent,
                              unsigned int pathMask,
                              bool idxUpdate,
                              unsigned int reachOffset[],
                              unsigned int splitOffset[]) {
  unsigned int *idxSource, *idxTarg;
  indexBuffers(predIdx, bufIdx, idxSource, idxTarg);

  for (unsigned int idx = idxStart; idx < idxStart + extent; idx++) {
    unsigned int sIdx = idxSource[idx];
    PathT path = idxPath->update(sIdx, pathMask, reachBase, idxUpdate);
    if (NodePath::isActive(path)) {
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

