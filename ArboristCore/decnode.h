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

#ifndef ARBORIST_DECNODE_H
#define ARBORIST_DECNODE_H

#include "typeparam.h"

typedef union {
  RankRange rankRange; // Range of splitting ranks:  numeric, pre-update.
  unsigned int offset; // Bit-vector offset:  factor.
  double num; // Rank-derived splitting value, post-update.
} SplitVal;


/**
   @brief Decision tree node.
*/
class DecNode {
 public:
  unsigned int lhDel;  // Delta to LH subnode. Nonzero iff non-terminal.
  unsigned int predIdx; // Nonterminal only.
  SplitVal splitVal;

  /**
     @brief Constructor.  Defaults to terminal.
   */
 DecNode() :
  lhDel(0) {
  }
};


#endif

