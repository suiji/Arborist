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
#include "obs.h"


unsigned int SampleNux::ctgBits = 0;
unsigned int SampleNux::ctgMask = 0;

unsigned int SampleNux::multMask = 0;

unsigned int SampleNux::rightBits = 0;
unsigned int SampleNux::rightMask = 0;


void SampleNux::setShifts(PredictorT nCtg,
			  IndexT maxSCount) {
  unsigned int bits = 1;
  ctgBits = 0;
  // Ctg values are zero-based, so the first power of 2 greater than or
  // equal to 'ctgWidth' has sufficient bits to hold all response values.
  while (bits < nCtg) {
    bits <<= 1;
    ctgBits++;
  }
  ctgMask = (1ul << ctgBits) - 1;
  
  unsigned int multBits = 1;
  bits = 1;
  while (bits < maxSCount) {
    bits <<= 1;
    multBits++;
  }
  multMask = (1ul << multBits) - 1;

  rightBits = ctgBits + multBits;
  rightMask = (1ul << rightBits) - 1;

  Obs::setShifts(ctgBits, multBits);
}


void SampleNux::deImmutables() {
  ctgBits = 0;
  ctgMask = 0;
  multMask = 0;
  rightBits = 0;
  rightMask = 0;

  Obs::deImmutables();
}
