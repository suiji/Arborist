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
  unsigned int offset; // Tree-relative bit-vector offset:  factor.
} SplitVal;


/**
   @brief Decision tree node.
*/
class DecNode {
 public:
  unsigned int lhDel;  // Delta to LH subnode. Nonzero iff non-terminal.
  unsigned int predIdx; // Predictor index: nonterminal only.
  SplitVal splitVal; // Split encoding:  terminal only.

  /**
     @brief Constructor.  Defaults to terminal.
   */
 DecNode() :
  lhDel(0) {
  }
};


#endif

