// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file obspart.cc

   @brief Methods to repartition observation frame by tree node.

   @author Mark Seligman
 */

#include "obspart.h"
#include "sample.h"
#include "summaryframe.h"
#include "splitfrontier.h"
#include "frontier.h"
#include "path.h"
#include "ompthread.h"

#include <numeric>


/**
   @brief Base class constructor.
 */
ObsPart::ObsPart(const SummaryFrame* frame,
                       IndexT bagCount_) :
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
ObsPart::~ObsPart() {
  delete [] nodeVec;
  delete [] indexBase;

  delete [] destRestage;
  delete [] destSplit;
}



vector<StageCount> ObsPart::stage(const RankedFrame* rankedFrame,
                                     const vector<SampleNux>  &sampleNode,
                                     const Sample* sample) {
  vector<StageCount> stageCount(rankedFrame->getNPred());

  OMPBound predTop = nPred;
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound predIdx = 0; predIdx < predTop; predIdx++) {
      stage(rankedFrame, sampleNode, sample, predIdx, stageCount[predIdx]);
    }
  }

  return stageCount;
}


void ObsPart::stage(const RankedFrame* rankedFrame,
                       const vector<SampleNux>& sampleNode,
                       const Sample* sample,
                       PredictorT predIdx,
                       StageCount& stageCount) {
  setStageBounds(rankedFrame, predIdx);
  IndexT* sIdx;
  SampleRank* spn = buffers(predIdx, 0, sIdx);
  const RowRank* rrPred = rankedFrame->predStart(predIdx);
  unsigned int expl = 0;
  for (IndexT idx = 0; idx < rankedFrame->getExplicitCount(predIdx); idx++) {
    stage(sampleNode, rrPred[idx], sample, expl, spn, sIdx);
  }

  stageCount.singleton = singleton(expl, predIdx);
  stageCount.expl = expl;
}


void ObsPart::setStageBounds(const RankedFrame* rankedFrame,
                                PredictorT predIdx) {
  unsigned int extent;
  stageOffset[predIdx] = rankedFrame->getSafeOffset(predIdx, bagCount, extent);
  stageExtent[predIdx] = extent;
}


void ObsPart::stage(const vector<SampleNux> &sampleNode,
		       const RowRank &rowRank,
                       const Sample* sample,
                       unsigned int &expl,
		       SampleRank spn[],
		       unsigned int smpIdx[]) const {
  IndexT sIdx;
  if (sample->sampledRow(rowRank.getRow(), sIdx)) {
    spn[expl].join(rowRank.getRank(), sampleNode[sIdx]);
    smpIdx[expl] = sIdx;
    expl++;
  }
}


double ObsPart::blockReplay(const SplitFrontier* splitFrontier,
                            const IndexSet* iSet,
                            const IndexRange& range,
                            bool leftExpl,
                            Replay* replay,
                            vector<SumCount>& ctgCrit) const {
  IndexT* sIdx;
  SampleRank* spn = buffers(splitFrontier->getDefCoord(iSet), sIdx);
  double sumExpl = 0.0;
  for (IndexT opIdx = range.getStart(); opIdx < range.getEnd(); opIdx++) {
    sumExpl += spn[opIdx].accum(ctgCrit);
    IndexT bitIdx = sIdx[opIdx];
    replay->set(bitIdx, leftExpl);
  }

  return sumExpl;
}


IndexT* ObsPart::indexBuffer(const SplitFrontier* splitFrontier,
                                const IndexSet* iSet) {
  return indexBuffer(splitFrontier->getDefCoord(iSet));
}


void ObsPart::prepath(const IdxPath *idxPath,
		      const unsigned int reachBase[],
		      const DefCoord& mrra,
		      const IndexRange& idxRange,
		      unsigned int pathMask,
		      bool idxUpdate,
		      unsigned int pathCount[]) {
  prepath(idxPath, reachBase, idxUpdate, idxRange, pathMask, bufferIndex(mrra), &pathIdx[getStageOffset(mrra.splitCoord.predIdx)], pathCount);
}

void ObsPart::prepath(const IdxPath *idxPath,
                         const unsigned int *reachBase,
                         bool idxUpdate,
                         const IndexRange& idxRange,
                         unsigned int pathMask,
                         unsigned int idxVec[],
                         PathT prepath[],
                         unsigned int pathCount[]) const {
  for (IndexT idx = idxRange.getStart(); idx < idxRange.getEnd(); idx++) {
    PathT path = idxPath->update(idxVec[idx], pathMask, reachBase, idxUpdate);
    prepath[idx] = path;
    if (NodePath::isActive(path)) {
      pathCount[path]++;
    }
  }
}


void ObsPart::rankRestage(const DefCoord& mrra,
                          const IndexRange& idxRange,
                          unsigned int reachOffset[],
                          unsigned int rankPrev[],
                          unsigned int rankCount[]) {
  SampleRank *source, *targ;
  IndexT *idxSource, *idxTarg;
  buffers(mrra, source, idxSource, targ, idxTarg);

  PathT *pathBlock = &pathIdx[getStageOffset(mrra.splitCoord.predIdx)];
  for (IndexT idx = idxRange.idxLow; idx < idxRange.getEnd(); idx++) {
    unsigned int path = pathBlock[idx];
    if (NodePath::isActive(path)) {
      SampleRank spNode = source[idx];
      IndexT rank = spNode.getRank();
      rankCount[path] += (rank == rankPrev[path] ? 0 : 1);
      rankPrev[path] = rank;
      IndexT destIdx = reachOffset[path]++;
      targ[destIdx] = spNode;
      idxTarg[destIdx] = idxSource[idx];
    }
  }
}


void ObsPart::indexRestage(const IdxPath *idxPath,
                           const unsigned int reachBase[],
                           const DefCoord& mrra,
                           const IndexRange& idxRange,
                           unsigned int pathMask,
                           bool idxUpdate,
                           unsigned int reachOffset[],
                           unsigned int splitOffset[]) {
  unsigned int *idxSource, *idxTarg;
  indexBuffers(mrra, idxSource, idxTarg);

  for (IndexT idx = idxRange.idxLow; idx < idxRange.getEnd(); idx++) {
    IndexT sIdx = idxSource[idx];
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

