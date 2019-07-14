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

#ifndef CORE_SPLITCOORD_H
#define CORE_SPLITCOORD_H

#include "typeparam.h"

/**
   @brief Split/predictor coordinate pair.
 */
//typedef pair<IndexType, unsigned int> SPPair;

struct SplitCoord {
  IndexType nodeIdx;
  unsigned int predIdx;

  SplitCoord(IndexType nodeIdx_,
             unsigned int predIdx_) :
  nodeIdx(nodeIdx_),
    predIdx(predIdx_) {
  }

  SplitCoord() :
  nodeIdx(0),
    predIdx(0) {
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
