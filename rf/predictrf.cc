// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictrf.cc

   @brief Algorithm-specific prediction scoring.

   @author Mark Seligman
 */

#include "predict.h"
#include "predictscorer.h"
#include "quant.h"


unsigned int PredictReg::scoreObs(size_t row) {
  (*yTarg)[row] = scorer->predictMean(row);
  if (!quant->isEmpty()) {
    quant->predictRow(this, row);
  }
  return nEst;
}


void PredictCtg::scoreObs(size_t row) {
  (*yTarg)[row] = scorer->predictPlurality(row, &census[ctgIdx(row)]);
  if (!ctgProb->isEmpty())
    ctgProb->predictRow(this, row, &census[ctgIdx(row)]);
}
