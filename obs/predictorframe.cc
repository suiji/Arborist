// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictorframe.cc

   @brief Methods for laying out predictor observations.

   @author Mark Seligman
 */


#include "predictorframe.h"
#include "coproc.h"
#include "valrank.h"
#include "ompthread.h"
#include "splitnux.h"

PredictorFrame::PredictorFrame(unique_ptr<RLEFrame> rleFrame_,
			       double autoCompress,
			       bool enableCoproc,
			       vector<string>& diag) :
  rleFrame(std::move(rleFrame_)),
  nObs(rleFrame->nObs),
  coproc(Coproc::Factory(enableCoproc, diag)),
  nPredNum(rleFrame->getNPredNum()),
  factorTop(cardinalities()),
  factorExtent(extents()),
  nPredFac(rleFrame->getNPredFac()),
  nPred(nPredFac + nPredNum),
  feIndex(mapPredictors(rleFrame->factorTop)),
  noRank(rleFrame->noRank),
  denseThresh(autoCompress * nObs),
  row2Rank(vector<vector<IndexT>>(nPred)),
  nonCompact(0),
  lengthCompact(0) {
  implExpl = denseBlock();
  obsPredictorFrame();
}


vector<Layout> PredictorFrame::denseBlock() {
  vector<Layout> implExpl(nPred);

#pragma omp parallel default(shared) num_threads(OmpThread::getNThread())
  {
#pragma omp for schedule(dynamic, 1)
    for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
      implExpl[predIdx] = surveyRanks(predIdx);
    }
  }

  return implExpl;
}


Layout PredictorFrame::surveyRanks(PredictorT predIdx) {
  IndexT rankMissing = rleFrame->findRankMissing(feIndex[predIdx]);
  
  row2Rank[predIdx] = vector<IndexT>(nObs);
  IndexT denseMax = 0; // Running maximum of run counts.
  PredictorT argMax = noRank;
  PredictorT rankPrev = noRank; // Forces write on first iteration.
  IndexT obsCount = 0; // Dummy initialization:  written before read.
  for (auto rle : getRLE(predIdx)) {
    IndexT rank = rle.val;
    IndexT extent = rle.extent;
    if (rank == rankPrev) {
      obsCount += extent;
    }
    else {
      obsCount = extent;
      rankPrev = rank;
    }

    // Tracks non-missing rank with highest # observations.
    if (rank != rankMissing && obsCount > denseMax) {
      denseMax = obsCount;
      argMax = rank;
    }

    // Piggybacks assignment of rank vector.
    for (IndexT idx = 0; idx != extent; idx++) {
      row2Rank[predIdx][rle.row + idx] = rank;
    }
  }

  // Post condition:  rowTot == nObs.
  return denseMax <= denseThresh ? Layout(noRank, nObs, rankMissing) : Layout(argMax, nObs - denseMax, rankMissing);
}


void PredictorFrame::obsPredictorFrame() {
  IndexT nPredDense = 0;
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


IndexRange PredictorFrame::getSafeRange(PredictorT predIdx,
				IndexT sampleCount) const {
  if (implExpl[predIdx].rankImpl == noRank) {
    return IndexRange(implExpl[predIdx].safeOffset * sampleCount, sampleCount);
  }
  else {
    return IndexRange(nonCompact * sampleCount + implExpl[predIdx].safeOffset, implExpl[predIdx].countExpl);
    }
}


vector<PredictorT> PredictorFrame::cardinalities() const {
  vector<PredictorT> cardPred;
  for (auto card : rleFrame->factorTop) {
    cardPred.push_back(card);
  }
  return cardPred;
}


vector<PredictorT> PredictorFrame::extents() const {
  vector<PredictorT> extentPred;
  for (auto facRanked : rleFrame->facRanked) {
    extentPred.push_back(facRanked.size());
  }
  return extentPred;
}


vector<PredictorT> PredictorFrame::mapPredictors(const vector<unsigned int>& factorTop_) const {
  vector<PredictorT> core2FE(nPred);
  PredictorT predIdx = 0;
  PredictorT facIdx = nPredNum;
  PredictorT numIdx = 0;
  for (auto card : factorTop_) {
    if (card > 0) {
      core2FE[facIdx++] = predIdx++;
    }
    else {
      core2FE[numIdx++] = predIdx++;
    }
  }
  return core2FE;
}


bool PredictorFrame::isFactor(const SplitNux& nux) const {
  return isFactor(nux.getPredIdx());
}


PredictorT PredictorFrame::getFactorExtent(const class SplitNux& nux) const {
  return rleFrame->getFactorTop(feIndex[nux.getPredIdx()]);
}


