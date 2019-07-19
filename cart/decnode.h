// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file decnode.h

   @brief Class defintion for decision tree node.

   @author Mark Seligman
 */

#ifndef CART_DECNODE_H
#define CART_DECNODE_H

#include "typeparam.h"

/**
   @brief Untagged union of split encodings; fields keyed by predictor type.

   Numerical splits begin as rank ranges and are later adjusted to double.
   Factor splits are tree-relative offsets.
 */
typedef union {
  IndexRange rankRange; // Range of splitting ranks:  numeric, pre-update.
  double num; // Rank-derived splitting value:  numeric, post-update.
  IndexType offset; // Tree-relative bit-vector offset:  factor.
} SplitVal;



/**
   @brief Splitting criterion.
 */
struct SplitCrit {
  unsigned int predIdx;
  SplitVal val;

  SplitCrit(unsigned int predIdx_,
            const IndexRange& rankRange) :
  predIdx(predIdx_) {
    val.rankRange = rankRange;
  }


  SplitCrit(unsigned int predIdx_,
            IndexType bitPos) :
  predIdx(predIdx_) {
    val.offset = bitPos;
  }

  SplitCrit() : predIdx(0) {
    val.num = 0.0;
  }

  void setNum(double num) {
    this->val.num = num;
  }


  auto getNumVal() const {
    return val.num;
  }

  
  auto getBitOffset() const {
    return val.offset;
  }
  

  /**
     @brief Imputes an intermediate rank.

     @param scale is a proportion value in [0.0, 1.0].

     @return fractional rank at the scaled position.
   */
  double imputeRank(double scale) const {
    return val.rankRange.interpolate(scale);
  }
};


/**
   @brief Decision tree node.
*/
struct DecNode {
  IndexType lhDel;  // Delta to LH subnode. Nonzero iff non-terminal.
  SplitCrit criterion;

  /**
     @brief Constructor.  Defaults to terminal.
   */
 DecNode() :
  lhDel(0) {
  }
};


#endif

