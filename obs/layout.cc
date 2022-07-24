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
  row2Rank(vector<vector<IndexT>>(nPred)),
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
  row2Rank[predIdx] = vector<IndexT>(nRow);
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
    for (IndexT idx = 0; idx != extent; idx++) {
      row2Rank[predIdx][rle.row + idx] = rank;
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


IndexRange Layout::getSafeRange(PredictorT predIdx,
				IndexT bagCount) const {
  if (implExpl[predIdx].rankImpl == noRank) {
    return IndexRange(implExpl[predIdx].safeOffset * bagCount, bagCount);
  }
  else {
    return IndexRange(nonCompact * bagCount + implExpl[predIdx].safeOffset, implExpl[predIdx].countExpl);
    }
}


const vector<RLEVal<unsigned int>>& Layout::getRLE(PredictorT predIdx) const {
  return trainFrame->getRLE(predIdx);
}
