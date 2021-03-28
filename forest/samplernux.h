// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplernux.h

   @brief Forest-wide packed representation of sampled observations.

   @author Mark Seligman
 */

#ifndef FOREST_SAMPLERNUX_H
#define FOREST_SAMPLERNUX_H

#include "typeparam.h"

class SamplerNux {
  IndexT sCount; // # times bagged:  == 0 iff marker.
  IndexT leafIdx; // Leaf index within tree, iff non-marker
  IndexT delRow; // Difference in adjacent row numbers, iff non-marker.

public:
  SamplerNux() :
    sCount(0) {
  }
  
  SamplerNux(IndexT delRow_,
	     IndexT leafIdx_,
	     IndexT sCount_) :
    sCount(sCount_),
    leafIdx(leafIdx_),
    delRow(delRow_) {
  }

  inline auto getDelRow() const {
    return delRow;
  }
  

  inline auto getLeafIdx() const {
    return leafIdx;
  }

  
  inline auto getSCount() const {
    return sCount;
  }
};


#endif
