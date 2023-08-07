// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictsb.cc

   @brief Algorithm-specific prediction scoring.

   @author Mark Seligman
 */

#include "predict.h"
#include "predictscorer.h"


unsigned int PredictReg::scoreObs(size_t obsIdx) {
  (*yTarg)[obsIdx] = scorer->predictSum(obsIdx);
  return nEst;
}


void PredictCtg::scoreObs(size_t obsIdx) {
  (*yTarg)[obsIdx] = scorer->predictProb(obsIdx, ctgProb.get(), &census[ctgIdx(obsIdx)]);
}
