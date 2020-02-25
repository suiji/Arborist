// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file branchsense.cc

   @brief Mapping samples to true/false branch partition.

   @author Mark Seligman
 */


#include "branchsense.h"

BranchSense::BranchSense(IndexT bagCount) :
  expl(make_unique<BV>(bagCount)),
  explTrue(make_unique<BV>(bagCount)) {
}


void BranchSense::frontierReset() {
  expl->clear();
  explTrue->saturate();
}


void BranchSense::set(IndexT idx, bool explTrueExpl) {
  expl->setBit(idx);
  if (!explTrueExpl) { // ExplTrue has been preset to full.
    explTrue->setBit(idx, false);
  }
}
