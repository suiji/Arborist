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
static constexpr PredictorT noPred = PredictorT(~0ull);

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
  bool noCoord() const {
    return predIdx == noPred;
  }
  
  /**
     @brief Computes node-major offset using passed stride value.
   */
  size_t strideOffset(unsigned int stride) const {
    return nodeIdx * stride + predIdx;
  }
};
#endif
