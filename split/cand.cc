// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file cand.cc

   @brief Methods building the list of splitting candidates.

   @author Mark Seligman
 */


#include "interlevel.h"
#include "cand.h"
#include "prng.h"
#include "frontier.h"
#include "splitfrontier.h"


Cand::Cand(const InterLevel* interLevel) :
  nSplit(interLevel->getNSplit()),
  nPred(interLevel->getNPred()),
  preCand(vector<vector<PreCand>>(nSplit)) {
}

  
void Cand::precandidates(const Frontier* frontier,
			 InterLevel* interLevel) {
  candidateCartesian(frontier, interLevel);
}


void Cand::candidateCartesian(const Frontier* frontier,
			      InterLevel* interLevel) {
  IndexT idx = 0;
  vector<double> dRand = PRNG::rUnif<double>(nPred * nSplit);
  for (IndexT splitIdx = 0; splitIdx < nSplit; splitIdx++) {
    if (!frontier->isUnsplitable(splitIdx)) { // Node can split.
      for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
	SplitCoord coord(splitIdx, predIdx);
	if (interLevel->preschedule(coord)) {
	  preCand[splitIdx].emplace_back(coord, getRandLow(dRand[idx++]));
	}
      }
    }
  }
}


void Cand::candidateBernoulli(const Frontier* frontier,
			      InterLevel* interLevel,
			      const vector<double>& predProb) {
  vector<double> ruPred = PRNG::rUnif<double>(nSplit * nPred);
  for (IndexT splitIdx = 0; splitIdx < nSplit; splitIdx++) {
    if (frontier->isUnsplitable(splitIdx)) { // Node cannot split.
      continue;
    }
    IndexT ruOff = splitIdx * nPred;
    for (PredictorT predIdx = 0; predIdx != nPred; predIdx++) {
      if (ruPred[ruOff] < predProb[predIdx]) {
	SplitCoord coord(splitIdx, predIdx);
	if (interLevel->preschedule(coord)) {
	  preCand[splitIdx].emplace_back(coord, getRandLow(ruPred[ruOff]));
	}
      }
      ruOff++;
    }
  }
}


void Cand::candidateFixed(const Frontier* frontier,
			  InterLevel* interLevel,
			  PredictorT predFixed) {
  vector<double> ruPred = PRNG::rUnif<double>(nSplit * nPred);

  for (IndexT splitIdx = 0; splitIdx < nSplit; splitIdx++) {
    if (frontier->isUnsplitable(splitIdx)) { // Node cannot split.
      continue;
    }
    vector<PredictorT> predRand(nPred);
    iota(predRand.begin(), predRand.end(), 0);
    IndexT ruOff = splitIdx * nPred;
    PredictorT schedCount = 0;
    for (PredictorT predTop = nPred; predTop != 0; predTop--) {
      PredictorT idxRand = predTop * ruPred[ruOff];
      PredictorT predIdx = exchange(predRand[idxRand], predRand[predTop-1]);
      SplitCoord coord(splitIdx, predIdx);
      if (interLevel->preschedule(coord)) {
	preCand[splitIdx].emplace_back(coord, getRandLow(ruPred[ruOff]));
	if (++schedCount == predFixed) {
	  break;
	}
      }
      ruOff++;
    }
  }
}


vector<SplitNux> Cand::stagedSimple(const InterLevel* interLevel,
				    SplitFrontier* sf) const {
  vector<SplitNux> postCand;
  for (IndexT nodeIdx = 0; nodeIdx < nSplit; nodeIdx++) {
    for (PreCand pc : preCand[nodeIdx]) {
      StagedCell* cell;
      if (interLevel->isStaged(pc.coord, cell)) {
	postCand.emplace_back(cell, pc.randVal, sf);
      } // Otherwise delisted.
    }
  }
  sf->accumPreset();

  return postCand;
}


vector<vector<SplitNux>> Cand::stagedCompound(const InterLevel* interLevel,
					      SplitFrontier* sf) const {
  vector<vector<SplitNux>> postCand(nSplit);
  for (IndexT nodeIdx = 0; nodeIdx < nSplit; nodeIdx++) {
    for (PreCand pc : preCand[nodeIdx]) {
      StagedCell* cell;
      if (interLevel->isStaged(pc.coord, cell)) {
	postCand[nodeIdx].emplace_back(cell, pc.randVal, sf);
      } // Otherwise delisted.
    }
  }
  sf->accumPreset();

  return postCand;
}
