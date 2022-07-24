// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file obs.cc

   @brief Methods to set and unset observation cell packing parameters.

   @author Mark Seligman
 */

#include "obs.h"
#include "splitnux.h"


unsigned int Obs::ctgMask = 0;
unsigned int Obs::multLow = 0;
unsigned int Obs::multMask = 0;
unsigned int Obs::numMask = 0;

  /**
     @brief Sets internal packing parameters.
   */
void Obs::setShifts(unsigned int ctgBits,
		    unsigned int multBits) {
  multLow = ctgLow + ctgBits;
  multMask = (1ul << multBits) - 1;
  ctgMask = (1ul << ctgBits) - 1;
  numMask = ~((1ul << (ctgBits + multBits + 1)) - 1);
}


void Obs::deImmutables() {
  multLow = multMask = ctgMask = numMask = 0;
}
