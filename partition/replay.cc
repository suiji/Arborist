// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file replay.cc

   @brief Mapping samples to L/R splits.

   @author Mark Seligman
 */


#include "replay.h"

Replay::Replay(IndexT bagCount) :
  expl(make_unique<BV>(bagCount)),
  left(make_unique<BV>(bagCount)) {
}


void Replay::reset() {
  expl->clear();
  left->saturate();
}


void Replay::set(IndexT idx, bool leftExpl) {
  expl->setBit(idx);
  if (!leftExpl) { // Preset to full.
    left->setBit(idx, false);
  }
}
