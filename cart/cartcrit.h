// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file cartcrit.h

   @brief Class defintion for CART splitting criterion.

   @author Mark Seligman
 */

#ifndef CART_CARTCRIT_H
#define CART_CARTCRIT_H

#include "typeparam.h"

/**
   @brief Untagged union of split encodings; fields keyed by predictor type.

   Numerical splits begin as rank ranges and are later adjusted to double.
   Factor splits are tree-relative offsets.
 */
typedef union {
  double num; // Rank-derived splitting value:  quantile or cut.
  IndexT offset; // Tree-relative bit-vector offset:  factor.

  void setNum(double numVal) {
    num = numVal;
  }

  void setOffset(size_t bitPos) {
    offset = bitPos;
  }
  
} SplitVal;


/**
   @brief Splitting criterion.
 */
struct CartCrit {
  PredictorT predIdx;
  SplitVal val;

  CartCrit(PredictorT predIdx_,
	   double quantRank) :
  predIdx(predIdx_) {
    val.setNum(quantRank);
  }


  CartCrit(PredictorT predIdx_,
	   size_t bitPos) :
  predIdx(predIdx_) {
    val.setOffset(bitPos);
  }

  CartCrit() : predIdx(0) {
    val.setNum(0.0);
  }

  void setNum(double num) {
    val.setNum(num);
  }


  auto getNumVal() const {
    return val.num;
  }

  
  auto getBitOffset() const {
    return val.offset;
  }
};


#endif

