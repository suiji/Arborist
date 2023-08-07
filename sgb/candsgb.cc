// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file candsgb.cc

   @brief Methods building the list of splitting candidates.

   @author Mark Seligman
 */

#include "candsgb.h"
#include "interlevel.h"
#include "frontier.h"


PredictorT CandSGB::predFixed = 0;
vector<double> CandSGB::predProb;


CandSGB::CandSGB(InterLevel* interLevel) :
  Cand(interLevel) {
}


void CandSGB::init(PredictorT feFixed,
	     const vector<double>& feProb) {
  predFixed = feFixed;
  for (auto prob : feProb) {
    predProb.push_back(prob);
  }
}


void CandSGB::deInit() {
  predFixed = 0;
  predProb.clear();
}


void CandSGB::precandidates(const Frontier* frontier,
			   InterLevel* interLevel) {
  if (predFixed == 0) {
    candidateBernoulli(frontier, interLevel, predProb);
  }
  else {
    candidateFixed(frontier, interLevel, predFixed);
  }
}
