// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file mrra.h

   @brief Represents most-recently restaged ancestor cell.

   @author Mark Seligman
 */

#ifndef PARTITION_MRRA_H
#define PARTITION_MRRA_H

#include "splitcoord.h"

/**
   @brief Coordinates of the most-recently restaged ancestor (definition).
 */
struct MRRA {
  SplitCoord splitCoord;
  unsigned char bufIdx; // Double-buffer index of definition.
  unsigned char del; // Delta between frontier and level of definition.


  MRRA() = default;

  
  MRRA(const SplitCoord& splitCoord_,
       unsigned int bufIdx_,
       unsigned int del_ = 0) :
  splitCoord(splitCoord_),
    bufIdx(bufIdx_),
    del(del_) {
  }

  
  /**
     @return index of complementary buffer.
   */
  unsigned int compBuffer() const {
    return 1 - bufIdx;
  }
};

#endif
