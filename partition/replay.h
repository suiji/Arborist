// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file replay.h

   @brief Encodes L/R partitioning of frontier.

   @author Mark Seligman

 */

#ifndef PARTITION_REPLAY_H
#define PARTITION_REPLAY_H

#include "bv.h"
#include "typeparam.h"

#include <memory>

class Replay {
  unique_ptr<BV> expl;  // Whether index be explicitly replayed.
  unique_ptr<BV> left;  // Explicit:  L/R ; else undefined.

public:
  Replay(IndexT bagCount);

  void reset();
  
  /**
     @brief Determines whether sample be assigned to left successor.

     N.B.:  Undefined for non-splitting IndexSet.

     @param sIdx indexes the sample in question.

     @return true iff sample index is assigned to the left successor.
   */
  inline bool senseLeft(IndexT sIdx,
			bool leftImpl) const {
    return expl->testBit(sIdx) ? left->testBit(sIdx) : leftImpl;
  }


  void set(IndexT idx,
	   bool leftExpl);
  
};

#endif
