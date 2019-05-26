// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rle.h

   @brief Run-length encoding for typename parameter.

   @author Mark Seligman
 */

#ifndef FRAMEMAP_RLE_H
#define FRAMEMAP_RLE_H


/**
   @brief Run-length encoding class for parametrized type.
 */

template<typename rec>
struct RLE {
  rec val;
  unsigned int runLength;

  RLE() {
  }

  
  RLE(const rec& val_,
      unsigned int runLength_) : val(val_),
                                 runLength(runLength_) {
  }
  
  void bumpLength(unsigned int bump) {
    runLength += bump;
  }
};

#endif
