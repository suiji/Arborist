// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file layout.cc

   @brief Methods for laying out observations.

   @author Mark Seligman
 */


#include "stagecount.h"
#include "layout.h"
#include "valrank.h"
#include "partition.h"
#include "sampleobs.h"
#include "ompthread.h"
#include "trainframe.h"

#include <algorithm>
#include <numeric>


Layout::Layout(const TrainFrame* trainFrame_,
	       double autoCompress) :
  trainFrame(trainFrame_),
  nRow(trainFrame->getNRow()),
  nPred(trainFrame->getNPred()),
  noRank(trainFrame->cardinality.empty() ? nRow : max(nRow, static_cast<IndexT>(*max_element(trainFrame->cardinality.begin(), trainFrame->cardinality.end())))),
  nPredDense(0),
  nonCompact(0),
  lengthCompact(0),
  denseThresh(autoCompress * nRow),
  implExpl(denseBlock(trainFrame)) {
}


vector<ImplExpl> Layout::denseBlock(const TrainFrame* trainFrame) {
  vector<ImplExpl> implExpl(nPred);

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
      implExpl[predIdx] = setDense(trainFrame, predIdx);
    }
  }

  return implExpl;
}


ImplExpl Layout::setDense(const TrainFrame* trainFrame, PredictorT predIdx) {
  IndexT denseMax = 0; // Running maximum of run counts.
  PredictorT argMax = noRank;
  PredictorT rankPrev = noRank; // Forces write on first iteration.
  IndexT runCount = 0; // Dummy value:  written before read.
  for (auto rle : trainFrame->getRLE(predIdx)) {
    IndexT rank = rle.val;
    IndexT extent = rle.extent;

    if (rank == rankPrev) {
      runCount += extent;
    }
    else {
      runCount = extent;
      rankPrev = rank;
    }

    if (runCount > denseMax) {
      denseMax = runCount;
      argMax = rank;
    }
  }
  // Post condition:  rowTot == nRow.
  return denseMax <= denseThresh ? ImplExpl(noRank, nRow) : ImplExpl(argMax, nRow - denseMax);
}


void Layout::accumOffsets() {
  for (auto & ie : implExpl) {
    if (ie.rankImpl == noRank) {
      ie.safeOffset = nonCompact++;
      ie.denseIdx = nPred;
    }
    else {
      ie.safeOffset = lengthCompact;
      ie.denseIdx = nPredDense++;
      lengthCompact += ie.countExpl;
    }
  }
}


vector<StageCount> Layout::stage(const SampleObs* sample,
				 ObsPart* obsPart) const {
  vector<StageCount> stageCount(nPred);

  OMPBound predTop = nPred;
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound predIdx = 0; predIdx < predTop; predIdx++) {
      stageCount[predIdx] = stage(sample, obsPart, predIdx);
    }
  }

  return stageCount;
}


StageCount Layout::stage(const SampleObs* sample,
			 ObsPart* obsPart,
			 PredictorT predIdx) const {
  obsPart->setStageRange(predIdx, getSafeRange(predIdx, sample->getBagCount()));
  IndexT rankDense = implExpl[predIdx].rankImpl;
  IndexT* idxStart;
  ObsCell* srStart = obsPart->buffers(predIdx, 0, idxStart);
  ObsCell* spn = srStart;
  IndexT* sIdx = idxStart;
  for (auto rle : trainFrame->getRLE(predIdx)) {
    IndexT rank = rle.val;
    if (rank != rankDense) {
      for (IndexT row = rle.row; row < rle.row + rle.extent; row++) {
	sample->joinRank(row, sIdx, spn, rank);
      }
    }
  }
  IndexT idxExplicit = spn - srStart;
  return StageCount(sample->getBagCount() - idxExplicit, obsPart->countRanks(predIdx, 0, noRank, idxExplicit));
  // Post-condition:  rrOut.size() == explicitCount[predIdx]
}


IndexRange Layout::getSafeRange(PredictorT predIdx,
				IndexT bagCount) const {
  if (implExpl[predIdx].rankImpl == noRank) {
    return IndexRange(implExpl[predIdx].safeOffset * bagCount, bagCount);
  }
  else {
    return IndexRange(nonCompact * bagCount + implExpl[predIdx].safeOffset, implExpl[predIdx].countExpl);
    }
}


