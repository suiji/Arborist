// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file runnux.h

   @brief Minimal representation of predictor run within a partition.

   @author Mark Seligman
 */

#ifndef SPLIT_RUNNUX_H
#define SPLIT_RUNNUX_H

#include "sumcount.h"

/**
   @brief Accumulates statistics for runs of factors having the same internal code.

   Allocated in bulk by Fortran-style workspace, the RunSet.
 */
struct RunNux {
  double sum; ///< Sum of responses associated with run.
  IndexT sCount; ///< Sample count of factor run.
  IndexRange obsRange; ///< Observation indices.


  /**
     @brief Initialzier for subsequent accumulation.
  */
  inline void init() {
    sCount = 0;
    sum = 0.0;
  }


  inline void startRange(IndexT idxStart) {
    obsRange.idxStart = idxStart;
  }
  

  inline void endRange(IndexT idxEnd) {
    obsRange.idxExtent = idxEnd - obsRange.idxStart + 1;
  }


  /**
     @brief Initializes as residual.
  */
  inline void setResidual(const SumCount& scImplicit,
			  IndexT obsEnd,
			  IndexT extent) {
    this->sCount = scImplicit.sCount;
    this->sum = scImplicit.sum;
    obsRange = IndexRange(obsEnd, extent);
  }

  
  /**
     @brief Range accessor.  N.B.:  Should not be invoked on dense
     run, as 'start' will hold a reserved value.

     @return range of indices subsumed by run.
   */
  inline IndexRange getRange() const {
    return obsRange;
  }


  /**
     @brief Accumulates run contents into caller.
  */
  inline void accum(SumCount& scAccum) const {
    scAccum.sCount += sCount;
    scAccum.sum += sum;
  }
};


#endif
