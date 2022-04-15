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

#include "candrf.h"
#include "defmap.h"
#include "prng.h"


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


void CandRF::precandidates(DefMap* defMap) {
// TODO:  Preempt overflow by walking wide subtrees depth-nodeIdx.
  if (predFixed == 0) {
    candidateProb(defMap);
  }
  else {
    candidateFixed(defMap);
  }
}


void CandRF::candidateProb(DefMap* defMap) {
  IndexT splitCount = defMap->getNSplit();
  PredictorT nPred = defMap->getNPred();

  vector<double> ruPred = PRNG::rUnif(splitCount * nPred);
  for (IndexT splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    if (defMap->isUnsplitable(splitIdx)) { // Node cannot split.
      continue;
    }
    IndexT ruOff = splitIdx * nPred;
    for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
      if (ruPred[ruOff] < predProb[predIdx]) {
	(void) defMap->preschedule(SplitCoord(splitIdx, predIdx), ruPred[ruOff]);
      }
      ruOff++;
    }
  }
}

void CandRF::candidateFixed(DefMap* defMap) {
  IndexT splitCount = defMap->getNSplit();
  PredictorT nPred = defMap->getNPred();
  vector<double> ruPred = PRNG::rUnif(splitCount * nPred);

  for (IndexT splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    if (defMap->isUnsplitable(splitIdx)) { // Node cannot split.
      continue;
    }
    vector<PredictorT> predRand(nPred);
    iota(predRand.begin(), predRand.end(), 0);
    IndexT ruOff = splitIdx * nPred;
    PredictorT schedCount = 0;
    for (PredictorT predTop = nPred; predTop != 0; predTop--) {
      PredictorT idxRand = predTop * ruPred[ruOff];
      PredictorT predIdx = exchange(predRand[idxRand], predRand[predTop-1]);
      if (defMap->preschedule(SplitCoord(splitIdx, predIdx), ruPred[ruOff])) {
	if (++schedCount == predFixed) {
	  break;
	}
      }
      ruOff++;
    }
  }
}
