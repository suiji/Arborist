// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rlecresc.cc

   @brief Methods for representing data frame using run-length encoding.

   @author Mark Seligman
 */

#include "rlecresc.h"


RLECresc::RLECresc(size_t nRow_,
		   unsigned int nPred) :
  nRow(nRow_),
  predForm(vector<PredictorForm>(nPred)),
  typedIdx(vector<unsigned int>(nPred)),
  nFactor(0),
  nNumeric(0) {
  fill(predForm.begin(), predForm.end(), PredictorForm::numeric); // Default initialization.
}


void RLECresc::dump(vector<size_t>& valOut,
		    vector<size_t>& extentOut,
		    vector<size_t>& rowOut) const {
  size_t i = 0;
  for (auto rlEnc : rle) {
    valOut[i] = rlEnc.val;
    extentOut[i] = rlEnc.extent;
    rowOut[i] = rlEnc.row;
    i++;
  }
}
