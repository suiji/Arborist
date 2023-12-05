// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file stagedcell.h

   @brief Summarizes cell-column statistics following (re)staging.

   @author Mark Seligman
 */

#ifndef OBS_STAGEDCELL_H
#define OBS_STAGEDCELL_H

#include "splitcoord.h"
#include "typeparam.h"

#include <vector>

/**
   @brief Cell statistics following (re)staging.
 */
struct StagedCell {
  const SplitCoord coord; ///< Associated node/predictor pair.
  const unsigned char bufIdx; ///< 0/1; flips at constructor.
  const unsigned char trackRuns; ///< Whether to order run values.
  unsigned char live; ///< Initialized to true; false is sticky.
  const IndexT valIdx; ///< Base offset of run values, iff tracked.
  IndexT runCount; ///< # runs.
  IndexRange obsRange; ///< Initialized from node; adjusted.
  IndexT obsImplicit; ///<  # implicit observations.
  IndexT preResidual; ///< # obs preceding residual, iff implicit.
  IndexT obsMissing; ///< # obs with missing predictor values.

  /**
     @brief Root constructor.
   */
StagedCell(PredictorT predIdx,
	   IndexT valIdx_,
	   IndexT extent,
	   IndexT runCount_,
	   bool trackRuns_ = false)
  : coord(SplitCoord(0, predIdx)),
    bufIdx(0),
    trackRuns(trackRuns_),
    live(true),
    valIdx(valIdx_),
    runCount(runCount_),
    obsRange(IndexRange(0, extent)),
    preResidual(0) {
    }
  

  /**
     @brief Restaging constructor.
   */
StagedCell(IndexT nodeIdx,
	   const StagedCell& source,
	   IndexT valIdx_,
	   const IndexRange& range_) :
  coord(SplitCoord(nodeIdx, source.getPredIdx())),
    bufIdx(1 - source.bufIdx),
    trackRuns(source.trackRuns),
    live(true),
    valIdx(valIdx_),
    obsRange(range_),
    preResidual(0) {
  }


  bool isLive() const {
    return live;
  }


  IndexT getNodeIdx() const {
    return coord.nodeIdx;
  }
  

  PredictorT getPredIdx() const {
    return coord.predIdx;
  }


  /**
     @return complementary buffer index.
   */
  unsigned int compBuffer() const {
    return 1 - bufIdx;
  }


  /**
     @brief Sets final rank count.

     @param runCount enumerates distinct predictor values.

     A runCount value of zero is short-hand for all singletons.
   */
  void setRunCount(IndexT runCount) {
    this->runCount = (runCount != 0 ? runCount : obsRange.idxExtent) + (obsImplicit == 0 ? 0 : 1);
  }


  void setPreresidual(IndexT preResidual) {
    this->preResidual = preResidual;
  }

  
  /**
     @brief Updates observation statistics; root only.
   */
  void updateCounts(IndexT implicitCount,
		    IndexT obsMissing) {
    obsRange.idxExtent -= implicitCount;
    obsImplicit = implicitCount;
    this->obsMissing = obsMissing;
  }


  /**
     @brief Initializes target cell from per-path statisics.
   */
  void updatePath(IndexT idxStart,
		  IndexT extent,
		  IndexT preResidual,
		  IndexT obsMissing) {
    setRange(idxStart, extent);
    this->preResidual = preResidual;
    this->obsMissing = obsMissing;
  }
  

  void setRange(IndexT idxStart,
		IndexT extent) {
    obsImplicit = obsRange.getExtent() - extent;
    obsRange = IndexRange(idxStart, extent);
  }


  IndexRange getObsRange() const {
    return obsRange;
  }
  

  /**
     @brief Marks extinct.
   */
  void delist() {
    live = false;
  }


  /**
     @return true iff cell contains implicit observations.
   */
  bool implicitObs() const {
    return obsImplicit != 0;
  }


  /**
     @return total number of explicit and implicit ranks.
   */
  IndexT getRunCount() const {
    return runCount;
  }


  /**
     @return true iff at least two distinct values present.
   */
  bool splitable() const {
    return runCount > 1;
  }


  /**
     @return true iff cell has trackable ties.
   */
  bool trackableTies() const {
    if (obsImplicit != 0) // Possible ties, but untrackable.
      return runCount != (obsRange.getExtent() + 1);
    else
      return runCount != obsRange.getExtent();
  }
};

#endif
