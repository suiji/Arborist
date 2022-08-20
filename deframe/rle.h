// This file is part of deframe.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rle.h

   @brief Run-length encoding for typename parameter.

   @author Mark Seligman
 */

#ifndef DEFRAME_RLE_H
#define DEFRAME_RLE_H

#include<cstddef>
using namespace std;

/**
   @brief Run-length encoding class for parametrized type.
 */

template<typename valType>
struct RLE {
  valType val;
  size_t extent;

  RLE() = default;

  
  RLE(const valType& val_,
      size_t extent_) : val(val_),
			extent(extent_) {
  }
  
  void bumpLength(size_t bump) {
    extent += bump;
  }
};


/**
  @brief Sparse representation imposed by front-end.
*/
template<typename valType>
struct RLEVal {
  valType val;
  size_t row;
  size_t extent;

  RLEVal(valType val_,
         size_t row_,
         size_t extent_ = 1) : val(val_),
			   row(row_),
			   extent(extent_) {
  }

  RLEVal(const RLEVal<valType>& rle) :
    val(rle.val),
    row(rle.row),
    extent(rle.extent) {
  }


  /**
     @brief Computes end position.

     @return row index beyond end.
   */
  size_t getRowEnd() const {
    return row + extent;
  }
};


#endif
