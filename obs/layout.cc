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

#include "layout.h"
#include "valrank.h"
#include "obspart.h"
#include "sample.h"
#include "ompthread.h"
#include "bheap.h"
#include "callback.h"

#include <algorithm>
#include <numeric>


Layout::Layout(const RLEFrame* rleFrame_,
	       double autoCompress,
	       PredictorT predPermute_) :
  rleFrame(rleFrame_),
  nRow(rleFrame->getNRow()),
  nPred(rleFrame->getNPred()),
  noRank(rleFrame->cardinality.empty() ? nRow : max(nRow, *max_element(rleFrame->cardinality.begin(), rleFrame->cardinality.end()))),
  predPermute(predPermute_),
  nPredDense(0),
  nonCompact(0),
  lengthCompact(0),
  denseThresh(autoCompress * nRow),
  implExpl(denseBlock(rleFrame)) {
  if (predPermute < nPred) {
    framePermute = rleFrame->permute(predPermute, shuffleRows());
  }
}


vector<size_t> Layout::shuffleRows() const {
  auto vUnif = CallBack::rUnif(nRow);
  vector<BHPair> heap(nRow);
  for (IndexT row = 0; row < nRow; row++) {
    BHeap::insert(&heap[0], row, vUnif[row]);
  }

  IndexT i = 0;
  vector<size_t> shuffle(nRow);
  for (IndexT heapSize = nRow; heapSize > 0; heapSize--) {
    shuffle[i++] = BHeap::slotPop(&heap[0], heapSize - 1);
  }

  return shuffle;
}


vector<ImplExpl> Layout::denseBlock(const RLEFrame* rleFrame) {
  vector<ImplExpl> implExpl(nPred);

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
      implExpl[predIdx] = setDense(rleFrame, predIdx);
    }
  }

  return implExpl;
}


ImplExpl Layout::setDense(const RLEFrame* rleFrame, PredictorT predIdx) {
  IndexT denseMax = 0; // Running maximum of run counts.
  PredictorT argMax = noRank;
  PredictorT rankPrev = noRank; // Forces write on first iteration.
  IndexT runCount = 0; // Dummy value:  written before read.
  for (auto rle : rleFrame->getRLE(predIdx)) {
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


Layout::~Layout() {
}


vector<IndexT> Layout::stage(const Sample* sample,
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


IndexT Layout::stage(const Sample* sample,
		     PredictorT predIdx,
		     ObsPart* obsPart) const {
  IndexT rankDense = implExpl[predIdx].rankImpl;
  IndexT* idxStart;
  SampleRank* spn = obsPart->buffers(predIdx, 0, idxStart);
  IndexT* sIdx = idxStart;
  for (auto rle : getStageFrame(predIdx)) {
    IndexT rank = rle.val;
    if (rank != rankDense) {
      IndexT row = rle.row;
      for (IndexT i = 0; i < rle.extent; i++) {
	const SampleNux* sNux;
	if (sample->sampledRow(row + i, sIdx, sNux)) {
	  spn++->join(rank, sNux);
	}
      }
    }
  }

  return sIdx - idxStart;
  // Post-condition:  rrOut.size() == explicitCount[predIdx]
}


const vector<RLEVal<unsigned int>>& Layout::getStageFrame(PredictorT predIdx) const {
  if (predIdx == predPermute) {
    return framePermute;
  }
  else {
    return rleFrame->getRLE(predIdx);
  }
}
