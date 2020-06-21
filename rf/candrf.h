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


struct CandRF : public Cand {
  static void
  init(PredictorT feFixed,
       const vector<double>& feProb);

  static void
  deInit();

  static vector<PreCand> precandidates(class SplitFrontier* splitFrontier,
				       class DefMap* bottom);


private:
  // Predictor sampling paraemters.
  static PredictorT predFixed;
  static vector<double> predProb;


  static void candidateProb(PredictorT nPred,
			    class DefMap* bottom,
			    IndexT splitIdx,
			    const double ruPred[],
			    vector<PreCand>& preCand);

  static void candidateFixed(PredictorT nPred,
			     class DefMap* bottom,
			     IndexT splitIdx,
			     const double ruPred[],
			     struct BHPair heap[],
			     vector<PreCand>& preCand);
};

#endif
