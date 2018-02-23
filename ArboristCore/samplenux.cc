// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplenux.cc

   @brief Methods for internal representation of sampled response and
   ranks.

   @author Mark Seligman
 */


#include "samplenux.h"

unsigned int SampleNux::nCtg = 0;
unsigned int SampleNux::ctgShift = 0;


/**
   @brief Computes a packing width sufficient to hold all (zero-based) response
   category values.

   @param ctgWidth is the response cardinality.

   @return void.
 */
void SampleNux::Immutables(unsigned int ctgWidth) {
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


/*
**/
void SampleNux::DeImmutables() {
  ctgShift = 0;
}
