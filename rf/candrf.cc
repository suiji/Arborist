// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file candrf.cc

   @brief Methods building the list of splitting candidates.

   @author Mark Seligman
 */

#include "runset.h"
#include "candrf.h"
#include "splitfrontier.h"
#include "defmap.h"
#include "callback.h"


PredictorT CandRF::predFixed = 0;
vector<double> CandRF::predProb;


void
CandRF::init(PredictorT feFixed,
	     const vector<double>& feProb) {
  predFixed = feFixed;
  for (auto prob : feProb) {
    predProb.push_back(prob);
  }
}


void
CandRF::deInit() {
  predFixed = 0;
  predProb.clear();
}


vector<DefCoord>
CandRF::precandidates(SplitFrontier* splitFrontier,
		      const DefMap* bottom) const {
// TODO:  Preempt overflow by walking wide subtrees depth-nodeIdx.
  IndexT splitCount = splitFrontier->getNSplit();
  PredictorT nPred = splitFrontier->getNPred();
  IndexT cellCount = splitCount * nPred;
  
  auto ruPred = CallBack::rUnif(cellCount);
  vector<BHPair> heap(predFixed == 0 ? 0 : cellCount);

  vector<DefCoord> preCand;
  for (IndexT splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    IndexT splitOff = splitIdx * nPred;
    if (splitFrontier->isUnsplitable(splitIdx)) { // Node cannot split.
      continue;
    }
    else if (predFixed == 0) { // Probability of predictor splitable.
      candidateProb(splitFrontier, bottom, splitIdx, &ruPred[splitOff], preCand);
    }
    else { // Fixed number of predictors splitable.
      candidateFixed(splitFrontier, bottom, splitIdx, &ruPred[splitOff], &heap[splitOff], preCand);
    }
  }

  return preCand;
}


void
CandRF::candidateProb(SplitFrontier* splitFrontier,
		      const DefMap* bottom,
		      IndexT splitIdx,
		      const double ruPred[],
		      vector<DefCoord>& preCand) const {
  for (PredictorT predIdx = 0; predIdx < splitFrontier->getNPred(); predIdx++) {
    if (ruPred[predIdx] < predProb[predIdx]) {
      (void) bottom->preschedule(splitFrontier, SplitCoord(splitIdx, predIdx), preCand);
    }
  }
}


void
CandRF::candidateFixed(SplitFrontier* splitFrontier,
		       const DefMap* bottom,
		       IndexT splitIdx,
		       const double ruPred[],
		       BHPair heap[],
		       vector<DefCoord>& preCand) const {
  // Inserts negative, weighted probability value:  choose from lowest.
  PredictorT nPred = splitFrontier->getNPred();
  for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
    BHeap::insert(heap, predIdx, -ruPred[predIdx] * predProb[predIdx]);
  }

  // Pops 'predFixed' items in order of increasing value.
  PredictorT schedCount = 0;
  for (PredictorT heapSize = nPred; heapSize > 0; heapSize--) {
    SplitCoord splitCoord(splitIdx, BHeap::slotPop(heap, heapSize - 1));
    schedCount += bottom->preschedule(splitFrontier, splitCoord, preCand);
    if (schedCount == predFixed)
      break;
  }
}
