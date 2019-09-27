// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplenux.cc

   @brief Methods for internal representation of sampled response and
   ranks.

   All but the static (de)initializer methods are inlined in the header.

   @author Mark Seligman
 */

#include "samplenux.h"
#include "sumcount.h"

unsigned int SampleNux::nCtg = 0;
unsigned int SampleNux::ctgShift = 0;


void SampleNux::immutables(unsigned int ctgWidth) {
  nCtg = ctgWidth;
  unsigned int bits = 1;
  ctgShift = 0;
  // Ctg values are zero-based, so the first power of 2 greater than or
  // equal to 'ctgWidth' has sufficient bits to hold all response values.
  while (bits < nCtg) {
    bits <<= 1;
    ctgShift++;
  }
}


void SampleNux::deImmutables() {
  ctgShift = 0;
}


FltVal SampleRank::accum(vector<SumCount>& ctgExpl) const {
  if (!ctgExpl.empty()) {
    ctgExpl[getCtg()] += SumCount(ySum, getSCount());
  }
  return ySum;
}
