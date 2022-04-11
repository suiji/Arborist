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
#include <numeric>
#include <algorithm>

struct CandRF : public Cand {
  static void
  init(PredictorT feFixed,
       const vector<double>& feProb);

  static void
  deInit();

  static void precandidates(class DefMap* defMap);


private:
  // Predictor sampling paraemters.
  static PredictorT predFixed;
  static vector<double> predProb;


  static void candidateProb(class DefMap* defMap);


  static void candidateFixed(class DefMap* defMap);
};

#endif
