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
#include "ompthread.h"


RLECresc::RLECresc(size_t nRow_,
		   unsigned int nPred) :
  nRow(nRow_),
  predForm(vector<PredictorForm>(nPred)),
  typedIdx(vector<unsigned int>(nPred)),
  rle(vector<vector<RLEVal<unsigned int>>>(nPred)),
  nFactor(0),
  nNumeric(0) {
  fill(predForm.begin(), predForm.end(), PredictorForm::numeric); // Default initialization.
}


void RLECresc::encodeFrame(const vector<void*>& colBase) {
  valFac = vector<vector<unsigned int>>(nFactor);
  valNum = vector<vector<double>>(nNumeric);

  OMPBound nPred = colBase.size();
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound predIdx = 0; predIdx < nPred; predIdx++) {
    bool isFactor;
    unsigned int typedIdx = getTypedIdx(predIdx, isFactor);
    if (isFactor) { // Only factors and numerics present.
      encodeColumn<unsigned int>(static_cast<unsigned int*>(colBase[predIdx]), valFac[typedIdx], predIdx);
    }
    else {
      encodeColumn<double>(static_cast<double*>(colBase[predIdx]), valNum[typedIdx], predIdx);
    }
  }
  }
}

void RLECresc::encodeFrameNum(const vector<double>&  feVal,
			      const vector<size_t>&  feRowStart,
			      const vector<size_t>&  feRunLength) {
  valFac = vector<vector<unsigned int>>(0);
  valNum = encodeSparse<double>(predForm.size(), feVal, feRowStart, feRunLength);
}


void RLECresc::encodeFrameNum(const double*  feVal) {
  OMPBound nPred = predForm.size();
  valFac = vector<vector<unsigned int>>(0);
  valNum = vector<vector<double>>(nPred);
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound predIdx = 0; predIdx < nPred; predIdx++) {
      encodeColumn(&feVal[predIdx * nRow], valNum[predIdx], predIdx);
    }
  }
}


void RLECresc::dump(vector<size_t>& valOut,
		    vector<size_t>& extentOut,
		    vector<size_t>& rowOut) const {
  size_t i = 0;
  for (auto rlePred : rle) {
    for (auto rlEnc : rlePred) {
      valOut[i] = rlEnc.val;
      extentOut[i] = rlEnc.extent;
      rowOut[i] = rlEnc.row;
      i++;
    }
  }
}
