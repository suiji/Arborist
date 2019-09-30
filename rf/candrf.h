// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef RF_CANDRF_H
#define RF_CANDRF_H

/**
   @file candrf.h

   @brief Manages RF-specific splitting candidate selection.

   @author Mark Seligman
 */

#include "cand.h"
#include "typeparam.h"

#include <vector>


class CandRF : public Cand {
  // Predictor sampling paraemters.
  static PredictorT predFixed;
  static vector<double> predProb;


  void
  candidateProb(class SplitFrontier* splitFroniter,
		const class Bottom* bottom,
		IndexT splitIdx,
		const double ruPred[],
		vector<class SplitNux>& preCand) const;

  void
  candidateFixed(class SplitFrontier* splitFrontier,
		 const class Bottom* bottom,
		 IndexT splitIdx,
		 const double ruPred[],
		 class BHPair heap[],
		 vector<class SplitNux>& preCand) const;

 public:
  
  static void
  init(PredictorT feFixed,
       const vector<double>& feProb);

  static void
  deInit();

  vector<class SplitNux>
  precandidates(class SplitFrontier* splitFrontier,
		const class Bottom* bottom) const;
};

#endif
