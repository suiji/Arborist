// This file is part of Deframe.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file localframe.cc

   @brief Methods decompressing sections of an RLEFrame.

   @author Mark Seligman
 */

#include "rleframe.h"
#include "predictframe.h"

// Inclusion only:
#include "quant.h"


PredictFrame::PredictFrame(const RLEFrame* frame) :
  nPredNum(frame == nullptr ? 0 : frame->getNPredNum()),
  nPredFac(frame == nullptr ? 0 : frame->getNPredFac()),
  idxTr(vector<size_t>(nPredNum + nPredFac)) {
}


void PredictFrame::transpose(const RLEFrame* frame,
			     size_t obsStart,
			     size_t extent) {
  baseObs = obsStart;
  num.clear();
  fac.clear();
  frame->transpose(idxTr, obsStart, extent, num, fac);
}

