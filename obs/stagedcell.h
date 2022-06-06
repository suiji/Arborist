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
  IndexRange range; ///< Initialized from node; adjusted iff implict.
  IndexT idxImplicit; ///<  # implicit indices.
  IndexT denseCut; ///< Index just left of dense rank, if any.
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
    range(IndexRange(0, extent)) {
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
    range(range_) {
    }


  inline bool isLive() const {
    return live;
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

  
  inline IndexT getRankCount() const {
    return rankCount;
  }
  
  
  /**
     @brief Sets range internally:  root only.
   */
  void updateRange(IndexT implicitCount) {
    range.idxExtent -= implicitCount;
    idxImplicit = implicitCount;
  }


  void setRange(IndexT idxStart,
		IndexT extent) {
    idxImplicit = range.getExtent() - extent;
    range = IndexRange(idxStart, extent);
  }


  IndexRange getRange() const {
    return range;
  }
  

  /**
     @brief Marks extinct.
   */
  void delist() {
    live = false;
  }


  bool isDense() const {
    return idxImplicit != 0;
  }
  
  
  /**
     @return Total number of explicit and implicit runs.
   */
  IndexT getRunCount() const {
    return rankCount + (idxImplicit == 0 ? 0 : 1);
  }

  
  bool isSingleton() const {
    return getRunCount() == 1;
  }
};

#endif
