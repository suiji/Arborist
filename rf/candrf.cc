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
#include "interlevel.h"
#include "frontier.h"


PredictorT CandRF::predFixed = 0;
vector<double> CandRF::predProb;


CandRF::CandRF(InterLevel* interLevel) :
  Cand(interLevel) {
}


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


void CandRF::precandidates(const Frontier* frontier,
			   InterLevel* interLevel) {
  if (predFixed == 0) {
    candidateBernoulli(frontier, interLevel, predProb);
  }
  else {
    candidateFixed(frontier, interLevel, predFixed);
  }
}
