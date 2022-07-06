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


/**
   @brief Accumulates statistics for runs of factors having the same internal code.

   Allocated in bulk by Fortran-style workspace, the RunSet.
 */
struct RunNux {
private:
  PredictorT code; // RESTORE to public
public:
  static IndexT noStart; // Inattainable starting index.
  //PredictorT code; // EXIT.  Same 0-based value as internal code.
  IndexT sCount; ///< Sample count of factor run:  need not equal length.
  double sum; ///< Sum of responses associated with run.
  IndexRange range;

  RunNux() : sCount(0), sum(0.0), range(IndexRange()) {
    range.idxStart = noStart;
  }


  PredictorT getCode() const {
    return code;
  }

  void setCode(PredictorT code) {
    this->code = code;
  }

  /**
     @brief Initialzier for subsequent accumulation.
  */  
  inline void init() {
    sCount = 0;
    sum = 0;
  }


  inline void startRange(IndexT idxStart) {
    range.idxStart = idxStart;
  }
  

  inline void endRange(IndexT idxEnd) {
    range.idxExtent = idxEnd - range.idxStart + 1;
  }


  /**
     @brief Initializes as residual.
  */
  inline void setResidual(IndexT implicitIdx, // EXIT
			  PredictorT code,
			  IndexT sCount,
			  double sum,
			  IndexT extent) {
    setCode(code);
    this->sCount = sCount;
    this->sum = sum;
    range.idxExtent = extent;
  }

  
  /**
     @brief Range accessor.  N.B.:  Should not be invoked on dense
     run, as 'start' will hold a reserved value.

     @return range of indices subsumed by run.
   */
  inline IndexRange getRange() const {
    return range;
  }


  /**
     @brief Accumulates run contents into caller.
   */
  inline void accum(IndexT& sCount,
                    double& sum) const {
    sCount += this->sCount;
    sum += this->sum;
  }


  /**
     @brief Implicit runs are characterized by a start value of 'noStart'.

     @return Whether this run is dense.
  */
  bool isImplicit() const {
    return range.getStart() == noStart;
  }
};


#endif
