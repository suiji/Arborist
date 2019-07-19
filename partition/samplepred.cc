// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplepred.cc

   @brief Observation matrix, partitioned by tree node.

   @author Mark Seligman
 */

#include "samplepred.h"
#include "splitnux.h"
#include "sample.h"
#include "summaryframe.h"
#include "path.h"
#include "bv.h"
#include "level.h"
#include "ompthread.h"

#include <numeric>


/**
   @brief Base class constructor.
 */
SamplePred::SamplePred(const SummaryFrame* frame,
                       IndexType bagCount_) :
  nPred(frame->getNPred()),
  bagCount(bagCount_),
  bufferSize(frame->safeSize(bagCount)),
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



vector<StageCount> SamplePred::stage(const RankedFrame* rankedFrame,
                                     const vector<SampleNux>  &sampleNode,
                                     const Sample* sample) {
  vector<StageCount> stageCount(rankedFrame->getNPred());

  OMPBound predIdx;
  OMPBound predTop = nPred;
#pragma omp parallel default(shared) private(predIdx) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (predIdx = 0; predIdx < predTop; predIdx++) {
      stage(rankedFrame, sampleNode, sample, predIdx, stageCount[predIdx]);
    }
  }

  return stageCount;
}


void SamplePred::stage(const RankedFrame* rankedFrame,
                       const vector<SampleNux>& sampleNode,
                       const Sample* sample,
                       unsigned int predIdx,
                       StageCount& stageCount) {
  setStageBounds(rankedFrame, predIdx);
  unsigned int* smpIdx;
  SampleRank* spn = buffers(predIdx, 0, smpIdx);
  const RowRank* rrPred = rankedFrame->predStart(predIdx);
  unsigned int expl = 0;
  for (unsigned int idx = 0; idx < rankedFrame->getExplicitCount(predIdx); idx++) {
    stage(sampleNode, rrPred[idx], sample, expl, spn, smpIdx);
  }

  stageCount.singleton = singleton(expl, predIdx);
  stageCount.expl = expl;
}


void SamplePred::setStageBounds(const RankedFrame* rankedFrame,
                                unsigned int predIdx) {
  unsigned int extent;
  stageOffset[predIdx] = rankedFrame->getSafeOffset(predIdx, bagCount, extent);
  stageExtent[predIdx] = extent;
}


void SamplePred::stage(const vector<SampleNux> &sampleNode,
		       const RowRank &rowRank,
                       const Sample* sample,
                       unsigned int &expl,
		       SampleRank spn[],
		       unsigned int smpIdx[]) const {
  IndexType sIdx;
  if (sample->sampledRow(rowRank.getRow(), sIdx)) {
    spn[expl].join(rowRank.getRank(), sampleNode[sIdx]);
    smpIdx[expl] = sIdx;
    expl++;
  }
}


double SamplePred::blockReplay(const SplitNux& argMax,
                               const IndexRange& range,
                               BV* replayExpl,
                               vector<SumCount> &ctgExpl) {
  IndexType* idx;
  SampleRank* spn = buffers(argMax.getPredIdx(), argMax.getBufIdx(), idx);
  double sumExpl = 0.0;
  for (IndexType spIdx = range.getStart(); spIdx < range.getEnd(); spIdx++) {
    sumExpl += spn[spIdx].accum(ctgExpl);
    replayExpl->setBit(idx[spIdx]);
  }

  return sumExpl;
}


void SamplePred::prepath(const IdxPath *idxPath,
                         const unsigned int reachBase[],
                         unsigned int predIdx,
                         unsigned int bufIdx,
                         const IndexRange& idxRange,
                         unsigned int pathMask,
                         bool idxUpdate,
                         unsigned int pathCount[]) {
  prepath(idxPath, reachBase, idxUpdate, idxRange, pathMask, bufferIndex(predIdx, bufIdx), &pathIdx[getStageOffset(predIdx)], pathCount);
}

void SamplePred::prepath(const IdxPath *idxPath,
                         const unsigned int *reachBase,
                         bool idxUpdate,
                         const IndexRange& idxRange,
                         unsigned int pathMask,
                         unsigned int idxVec[],
                         PathT prepath[],
                         unsigned int pathCount[]) const {
  for (IndexType idx = idxRange.getStart(); idx < idxRange.getEnd(); idx++) {
    PathT path = idxPath->update(idxVec[idx], pathMask, reachBase, idxUpdate);
    prepath[idx] = path;
    if (NodePath::isActive(path)) {
      pathCount[path]++;
    }
  }
}


void SamplePred::restage(Level *levelBack,
                         Level *levelFront,
                         const SplitCoord &mrra,
                         unsigned int bufIdx) {
  levelBack->rankRestage(this, mrra, levelFront, bufIdx);
}


void SamplePred::rankRestage(unsigned int predIdx,
                             unsigned int bufIdx,
                             const IndexRange& idxRange,
                             unsigned int reachOffset[],
                             unsigned int rankPrev[],
                             unsigned int rankCount[]) {
  SampleRank *source, *targ;
  unsigned int *idxSource, *idxTarg;
  buffers(predIdx, bufIdx, source, idxSource, targ, idxTarg);

  PathT *pathBlock = &pathIdx[getStageOffset(predIdx)];
  for (IndexType idx = idxRange.idxLow; idx < idxRange.getEnd(); idx++) {
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
                              const IndexRange& idxRange,
                              unsigned int pathMask,
                              bool idxUpdate,
                              unsigned int reachOffset[],
                              unsigned int splitOffset[]) {
  unsigned int *idxSource, *idxTarg;
  indexBuffers(predIdx, bufIdx, idxSource, idxTarg);

  for (IndexType idx = idxRange.idxLow; idx < idxRange.getEnd(); idx++) {
    IndexType sIdx = idxSource[idx];
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

