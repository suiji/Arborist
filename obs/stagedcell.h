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
  const char bufIdx; ///< Alternates source/target.
  char live; ///< Initialized to true; false is sticky.
  const IndexT rankStart; ///< Index into frontier rank buffer.
  IndexT rankCount; ///< # unique ranks.
  IndexRange obsRange; ///< Initialized from node; adjusted iff implicit.
  IndexT obsImplicit; ///<  # implicit observations.
  IndexT preResidual; ///< # obs preceding residual, iff implicit.
  // IndexT nNA; ///< # undefined observations in cell.


  /**
     @brief Root constructor.
   */
StagedCell(PredictorT predIdx,
	   IndexT extent,
	   IndexT rankStart_)
  : coord(SplitCoord(0, predIdx)),
    bufIdx(0),
    live(true),
    rankStart(rankStart_),
    obsRange(IndexRange(0, extent)),
    preResidual(0) {
    }
  

  /**
     @brief Restaging constructor.
   */
StagedCell(IndexT nodeIdx,
	   const StagedCell& source,
	   const IndexRange& range_,
	   IndexT rankStart_)
  : coord(SplitCoord(nodeIdx, source.getPredIdx())),
    bufIdx(1 - source.bufIdx),
    live(true),
    rankStart(rankStart_),
    obsRange(range_),
    preResidual(0) {
  }


  inline bool isLive() const {
    return live;
  }


  /**
     @return rank at specified offset from rear.
   */
  inline IndexT rankRear(IndexT backIdx) const {
    return rankStart + rankCount - 1 - backIdx;
  }

  

  inline IndexT getNodeIdx() const {
    return coord.nodeIdx;
  }
  

  inline PredictorT getPredIdx() const {
    return coord.predIdx;
  }


  /**
     @return complementary buffer index.
   */
  inline unsigned int compBuffer() const {
    return 1 - bufIdx;
  }


  /**
     @brief Sets final rank count.  Marks extinct if singleton.
   */
  inline void setRankCount(IndexT rankCount) {
    this->rankCount = rankCount;
  }


  inline void setPreresidual(IndexT preResidual) {
    this->preResidual = preResidual;
  }

  
  /**
     @brief Sets range internally:  root only.
   */
  void updateRange(IndexT implicitCount) {
    obsRange.idxExtent -= implicitCount;
    obsImplicit = implicitCount;
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
     @return position of the residual rank.

     Return value is legal, but useless, if no implicit indices.
   */
  IndexT residualPosition() const {
    return rankStart + rankCount - 1;
  }

  

  /**
     @return total number of explicit and implicit ranks.
   */
  inline IndexT getRankCount() const {
    return rankCount;
  }


  bool isSingleton() const {
    return rankCount == 1;
  }


  /**
     @return true iff cell has ties.
   */
  bool hasTies() const {
    if (obsImplicit != 0)
      return rankCount != (obsRange.getExtent() + 1);
    else
      return rankCount != obsRange.getExtent();
  }
};

#endif
