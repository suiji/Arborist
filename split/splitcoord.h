// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitcoord.h

   @brief Class definition characterizing a split.

   @author Mark Seligman
 */

#ifndef SPLIT_SPLITCOORD_H
#define SPLIT_SPLITCOORD_H

#include "typeparam.h"

/**
   @brief Split/predictor coordinate pair.
 */

// Blunt assignment of inattainable predictor index.
static constexpr PredictorT noPred = sizeof(PredictorT) == sizeof(size_t) ? ~0ull : (1ull << 8*sizeof(PredictorT)) - 1;

struct SplitCoord {
  IndexT nodeIdx;
  PredictorT predIdx;

  SplitCoord(IndexT nodeIdx_,
             PredictorT predIdx_) :
  nodeIdx(nodeIdx_),
    predIdx(predIdx_) {
  }

  SplitCoord() :
  nodeIdx(0),
    predIdx(noPred) {
  }

  /**
     @brief Indicates whether coord has been initialized to an actual predictor.
   */
  inline bool noCoord() const {
    return predIdx == noPred;
  }
  
  /**
     @brief Computes node-major offset using passed stride value.
   */
  inline size_t strideOffset(unsigned int stride) const {
    return nodeIdx * stride + predIdx;
  }

  
  /**
     @brief Scales node index to account for multi-level binary splitting.

     @param del is a specified number of back levels.

     @return node index scaled by level difference.
   */
  inline size_t backScale(unsigned int del) const {
    return nodeIdx << del;
  }
};
#endif
