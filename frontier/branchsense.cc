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
#include "critencoding.h"


BranchSense::BranchSense(IndexT bagCount) :
  expl(make_unique<BV>(bagCount)),
  explTrue(make_unique<BV>(bagCount)) {
  explTrue->saturate();
}


void BranchSense::set(IndexT idx, bool trueEncoding) {
  expl->setBit(idx);
  if (!trueEncoding) {
    explTrue->setBit(idx, false);
  }
}


void BranchSense::unset(IndexT idx, bool trueEncoding) {
  expl->setBit(idx, false);
  if (!trueEncoding) {
    explTrue->setBit(idx, true);
  }
}
