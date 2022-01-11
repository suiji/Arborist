// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file branchsense.h

   @brief Encodes true/false branch partitioning of frontier.

   @author Mark Seligman

 */

#ifndef PARTITION_BRANCHSENSE_H
#define PARTITION_BRANCHSENSE_H

#include "bv.h"
#include "typeparam.h"

#include <memory>

class BranchSense {
  unique_ptr<BV> expl;  // Whether index be explicitly replayed.
  unique_ptr<BV> explTrue;  // If expl set, whether sense is true or false; else undefined.

public:
  BranchSense(IndexT bagCount);

  /**
     @brief Determines whether sample be assigned to explTrue successor.

     N.B.:  Undefined for non-splitting IndexSet.

     @param sIdx indexes the sample in question.

     @return true iff sample index is assigned to the true-branching successor.
   */
  inline bool senseTrue(IndexT sIdx,
			bool implicitTrue) const {
    return expl->testBit(sIdx) ? explTrue->testBit(sIdx) : implicitTrue;
  }


  /**
     @param sIdx indexes a sample.

     @return true iff sample has been explicity replayed.
   */
  inline bool isExplicit(IndexT sIdx) const {
    return expl->testBit(sIdx);
  }


  void set(IndexT idx,
	   bool trueEncoding);
  

  void unset(IndexT idx,
	     bool trueEncoding);
};

#endif
