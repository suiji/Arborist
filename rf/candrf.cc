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

#include "bheap.h"
#include "candrf.h"
#include "sfcart.h"
#include "defmap.h"
#include "callback.h"


PredictorT CandRF::predFixed = 0;
vector<double> CandRF::predProb;


void CandRF::init(PredictorT feFixed,
	     const vector<double>& feProb) {
  predFixed = feFixed;
  for (auto prob : feProb) {
    predProb.push_back(prob);
  }
}


void CandRF::deInit() {
  predFixed = 0;
  predProb.clear();
}


vector<PreCand> CandRF::precandidates(SplitFrontier* splitFrontier,
				      DefMap* defMap) {
// TODO:  Preempt overflow by walking wide subtrees depth-nodeIdx.
  IndexT splitCount = splitFrontier->getNSplit();
  PredictorT nPred = splitFrontier->getNPred();
  IndexT cellCount = splitCount * nPred;
  
  auto ruPred = CallBack::rUnif(cellCount);
  vector<BHPair> heap(predFixed == 0 ? 0 : cellCount);

  vector<PreCand> preCand;
  for (IndexT splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    IndexT splitOff = splitIdx * nPred;
    if (splitFrontier->isUnsplitable(splitIdx)) { // Node cannot split.
      continue;
    }
    else if (predFixed == 0) { // Probability of predictor splitable.
      candidateProb(nPred, defMap, splitIdx, &ruPred[splitOff], preCand);
    }
    else { // Fixed number of predictors splitable.
      candidateFixed(nPred, defMap, splitIdx, &ruPred[splitOff], &heap[splitOff], preCand);
    }
  }

  return preCand;
}


void CandRF::candidateProb(PredictorT nPred,
			   DefMap* defMap,
			   IndexT splitIdx,
			   const double ruPred[],
			   vector<PreCand>& preCand) {
  for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
    if (ruPred[predIdx] < predProb[predIdx]) {
      (void) defMap->preschedule(SplitCoord(splitIdx, predIdx), preCand);
    }
  }
}


void CandRF::candidateFixed(PredictorT nPred,
			    DefMap* defMap,
			    IndexT splitIdx,
			    const double ruPred[],
			    BHPair heap[],
			    vector<PreCand>& preCand) {
  // Inserts negative, weighted probability value:  choose from lowest.
  for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
    BHeap::insert(heap, predIdx, -ruPred[predIdx] * predProb[predIdx]);
  }

  // Pops 'predFixed' items in order of increasing value.
  PredictorT schedCount = 0;
  for (PredictorT heapSize = nPred; heapSize > 0; heapSize--) {
    SplitCoord splitCoord(splitIdx, BHeap::slotPop(heap, heapSize - 1));
    schedCount += defMap->preschedule(splitCoord, preCand);
    if (schedCount == predFixed)
      break;
  }
}
