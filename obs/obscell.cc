// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file obscell.cc

   @brief Methods to set and unset observation cell packing parameters.

   @author Mark Seligman
 */

#include "obscell.h"

IndexT ObsCell::maxSCount = 0;
unsigned int ObsCell::ctgBits = 0;
unsigned int ObsCell::ctgMask = 0;
unsigned int ObsCell::multBits = 0;
unsigned int ObsCell::multMask = 0;

double ObsCell::scale = 1.0;
double ObsCell::recipScale = 1.0;

  /**
     @brief Sets internal packing parameters.
   */
void ObsCell::setShifts(IndexT maxSCount_,
			unsigned int ctgBits_,
		        unsigned int multBits_) {
  maxSCount = maxSCount_;
  ctgBits = ctgBits_;
  multBits = multBits_;
  multMask = (1ul << multBits) - 1;
  ctgMask = (1ul << ctgBits) - 1;
}


void ObsCell::setScale(double yMax) {
  scale = (yMax * maxSCount) / 0.49;
  if (scale < 1.0)
    scale = 1.0;
  recipScale = 1.0 / scale;
}


void ObsCell::deImmutables() {
  maxSCount = ctgBits = multBits = multMask = ctgMask = 0;
  scale = recipScale = 1.0;
}
