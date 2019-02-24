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
#include "ompthread.h"

#include <numeric>


/**
   @brief Base class constructor.
 */
SamplePred::SamplePred(unsigned int _nPred,
                       unsigned int _bagCount,
                       unsigned int _bufferSize) :
  nPred(_nPred),
  bagCount(_bagCount),
  bufferSize(_bufferSize),
  pathIdx(bufferSize),
  stageOffset(nPred),
  stageExtent(nPred) {
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
void SamplePred::setStageBounds(const RowRank* rowRank,
                                unsigned int predIdx) {
  unsigned int extent;
  unsigned int safeOffset = rowRank->getSafeOffset(predIdx, bagCount, extent);
  stageOffset[predIdx] = safeOffset;
  stageExtent[predIdx] = extent;
}



/**
   @brief Loops through the predictors to stage.

   @return void.
 */
vector<StageCount> SamplePred::stage(const RowRank* rowRank,
                                     const vector<SampleNux>  &sampleNode,
                                     const Sample* sample) {
  vector<StageCount> stageCount(rowRank->getNPred());

  OMPBound predIdx;
  OMPBound predTop = nPred;
#pragma omp parallel default(shared) private(predIdx) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (predIdx = 0; predIdx < predTop; predIdx++) {
      stage(rowRank, sampleNode, sample, predIdx, stageCount[predIdx]);
    }
  }

  return move(stageCount);
}


/**
   @brief Stages SamplePred objects in non-decreasing predictor order.

   @param predIdx is the predictor index.

   @return void.
*/
void SamplePred::stage(const RowRank* rowRank,
                       const vector<SampleNux>& sampleNode,
                       const Sample* sample,
                       unsigned int predIdx,
                       StageCount& stageCount) {
  setStageBounds(rowRank, predIdx);
  unsigned int *smpIdx;
  SampleRank *spn = buffers(predIdx, 0, smpIdx);
  const RRNode* rrPred = rowRank->predStart(predIdx);
  unsigned int expl = 0;
  for (unsigned int idx = 0; idx < rowRank->getExplicitCount(predIdx); idx++) {
    stage(sampleNode, rrPred[idx], sample, expl, spn, smpIdx);
  }

  stageCount.singleton = singleton(expl, predIdx);
  stageCount.expl = expl;
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
void SamplePred::stage(const vector<SampleNux> &sampleNode,
		       const RRNode &rrNode,
                       const Sample* sample,
                       unsigned int &expl,
		       SampleRank spn[],
		       unsigned int smpIdx[]) const {
  unsigned int sIdx;
  if (sample->sampledRow(rrNode.getRow(), sIdx)) {
    spn[expl].join(rrNode.getRank(), sampleNode[sIdx]);
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
  if (!ctgExpl.empty()) {
    return replayCtg(spn, blockStart, blockExtent, idx, replayExpl, ctgExpl);
  }
  else {
    return replayNum(spn, blockStart, blockExtent, idx, replayExpl);
  }
}

double SamplePred::replayCtg(const SampleRank spn[], unsigned int start, unsigned int extent, const unsigned int idx[], BV* replayExpl, vector<SumCount> &ctgExpl) {
  double sumExpl = 0.0;
  for (unsigned int spIdx = start; spIdx < start + extent; spIdx++) {
    FltVal ySum;
    unsigned int yCtg;
    unsigned sCount = spn[spIdx].ctgFields(ySum, yCtg);
    ctgExpl[yCtg].accum(ySum, sCount);
    sumExpl += ySum;
    replayExpl->setBit(idx[spIdx]);
  }

  return sumExpl;
}


double SamplePred::replayNum(const SampleRank spn[], unsigned int start, unsigned int extent, const unsigned idx[], BV* replayExpl) {
  double sumExpl = 0.0;
  for (unsigned int spIdx = start; spIdx < start + extent; spIdx++) {
    sumExpl += spn[spIdx].getYSum();
    replayExpl->setBit(idx[spIdx]);
  }

  return sumExpl;
}


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
void SamplePred::restage(Level *levelBack,
                         Level *levelFront,
                         const SPPair &mrra,
                         unsigned int bufIdx) {
  levelBack->rankRestage(this, mrra, levelFront, bufIdx);
}


/**
   @brief Restages and tabulates rank counts.

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

