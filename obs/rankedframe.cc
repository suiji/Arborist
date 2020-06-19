// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rankedframe.cc

   @brief Methods for presorting and accessing predictors by rank.

   @author Mark Seligman
 */

#include "rankedframe.h"
#include "valrank.h"
#include "obspart.h"
#include "sample.h"
#include "ompthread.h"

#include <algorithm>
#include <numeric>

// Observations are blocked according to type.  Blocks written in separate
// calls from front-end interface.

RankedFrame::RankedFrame(const RLEFrame* rleFrame_,
                         double autoCompress,
			 PredictorT predPermute_) :
  rleFrame(rleFrame_),
  nRow(rleFrame->getNRow()),
  nPred(rleFrame->getNPred()),
  noRank(rleFrame->cardinality.empty() ? nRow : max(nRow, *max_element(rleFrame->cardinality.begin(), rleFrame->cardinality.end()))),
  predPermute(predPermute_),
  nPredDense(0),
  denseIdx(vector<unsigned int>(nPred)),
  nonCompact(0),
  lengthCompact(0),
  denseRank(vector<IndexT>(nPred)),
  rrPred(vector<vector<RowRank>>(nPred)),
  safeOffset(vector<IndexT>(nPred)),
  denseThresh(autoCompress * nRow) {
  denseBlock();
}


void RankedFrame::denseBlock() {

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
      countExplicit(predIdx);
    }
  }

  //  Loop-carried dependencies.
  for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
    accumOffsets(predIdx);
  }
}


void RankedFrame::countExplicit(PredictorT predIdx) {
  IndexT denseMax = 0; // Running maximum of run counts.
  PredictorT argMax = noRank;
  PredictorT rankPrev = noRank; // Forces write on first iteration.
  IndexT runCount = 0; // Dummy value:  written before read.
  for (size_t rleIdx = rleFrame->idxStart(predIdx); rleIdx != rleFrame->idxEnd(predIdx); rleIdx++) {
    IndexT rank = rleFrame->getVal(rleIdx);
    IndexT extent = rleFrame->getExtent(rleIdx);

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

  denseRank[predIdx] = denseMax <= denseThresh ? noRank : argMax;
  rrExplicit(predIdx);
}


void RankedFrame::rrExplicit(PredictorT predIdx) {
  vector<RowRank>& rrOut = rrPred[predIdx];
  IndexT rankDense = denseRank[predIdx];
  for (size_t rleIdx = rleFrame->idxStart(predIdx); rleIdx != rleFrame->idxEnd(predIdx); rleIdx++) {
    IndexT rank = rleFrame->getVal(rleIdx);
    if (rank != rankDense) { // Non-dense runs expanded.
      IndexT row = rleFrame->getRow(rleIdx);
      for (IndexT i = 0; i < rleFrame->getExtent(rleIdx); i++) {
	rrOut.emplace_back(row + i, rank);
      }
    }
  }
  // Post-condition:  rrOut.size() == explicitCount[predIdx]
}


void RankedFrame::accumOffsets(PredictorT predIdx) {
  if (denseRank[predIdx] == noRank) {
    safeOffset[predIdx] = nonCompact++; // Index:  non-dense storage.
    denseIdx[predIdx] = nPred;
  }
  else {  // Sufficiently long run found:
    safeOffset[predIdx] = lengthCompact; // Accumulated offset:  dense.
    lengthCompact += rrPred[predIdx].size();
    denseIdx[predIdx] = nPredDense++;
  }
}


RankedFrame::~RankedFrame() {
}


vector<IndexT> RankedFrame::stage(const Sample* sample,
				      ObsPart* obsPart) const {
  vector<IndexT> stageCount(nPred);

  OMPBound predTop = nPred;
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound predIdx = 0; predIdx < predTop; predIdx++) {
      obsPart->stageRange[predIdx] = getSafeRange(predIdx, sample->getBagCount());
      stageCount[predIdx] = stage(sample, predIdx, obsPart);
    }
  }

  return stageCount;
}


IndexT RankedFrame::stage(const Sample* sample,
			      PredictorT predIdx,
			      ObsPart* obsPart) const {
  IndexT* idxStart;
  SampleRank* spn = obsPart->buffers(predIdx, 0, idxStart);
  IndexT* sIdx = idxStart;
  for (auto rr : rrPred[predIdx]) {
    const SampleNux* sNux;
    if (sample->sampledRow(rr.row, sIdx, sNux)) {
      spn++->join(rr.rank, sNux);
    }
  }
  return sIdx - idxStart;
}


