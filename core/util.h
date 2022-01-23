// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file util.h

   @brief General utilities.

   @author Mark Seligman
 */

#ifndef CORE_UTIL_H
#define CORE_UTIL_H

#include "typeparam.h"
#include <vector>

struct Util {
  /**
     @brief Computes bits needed to subsume a value:  brute-force log2.

     @return number of bits subsumed.
   */
  static unsigned int packedWidth(IndexT sz) {
    unsigned int width = 1;
    PackedT shift = 2ull;
    while (shift < sz) {
      shift <<= 1;
      width++;
    }

    return width;
  }
};

#endif
