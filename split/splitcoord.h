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
static constexpr PredictorT noPred = 0xffffffff;

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


/**
   @brief Includes the index of the buffer containing the cell's definition.
 */
struct PreCand {
  SplitCoord splitCoord;
  unsigned char bufIdx; // Double-buffer containing definition.
  unsigned char del; // Delta between current level and level of definition.


  PreCand() :
  splitCoord(SplitCoord()),
    bufIdx(0),
    del(0) {
  }

  
  PreCand(const SplitCoord& splitCoord_,
	   unsigned int bufIdx_,
	   unsigned int del_ = 0) :
  splitCoord(splitCoord_),
    bufIdx(bufIdx_),
    del(del_) {
  }

  
  /**
     @return index of complementary buffer.
   */
  unsigned int compBuffer() const {
    return 1 - bufIdx;
  }
};

#endif
