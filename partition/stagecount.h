// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file stagecount.h

   @brief Summarizes cell-column statistics following (re)staging.

   @author Mark Seligman
 */

#ifndef PARTITION_STAGECOUNT_H
#define PARTITION_STAGECOUNT_H


#include "typeparam.h"


/**
   @brief Column statistics following (re)staging.
 */
struct StageCount {
  IndexT idxImplicit; // # implicit staged SampleRank indices.
  IndexT rankCount; // # distinct explicit rank/codes.

  StageCount(IndexT idxImplicit_,
	     IndexT rankCount_) :
    idxImplicit(idxImplicit_),
    rankCount(rankCount_) {
  }

  StageCount() = default;

  /**
     @return Total number of explicit and implicit runs.
   */
  IndexT getRunCount() const {
    return rankCount + (idxImplicit == 0 ? 0 : 1);
  }

  bool isSingleton() const {
    return getRunCount() == 1;
  }

  
  /**
     @brief Checks whether the container has been initialized nontrivially.

     Testing only.

     @return true iff run-count is nozero.
   */
  bool isInitialized() const {
    return getRunCount() != 0;
  }
};

#endif
