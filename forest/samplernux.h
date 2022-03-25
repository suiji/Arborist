// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplernuxh

   @brief Compact summary of observation sampling.

   @author Mark Seligman
 */

#ifndef FOREST_SAMPLERNUX_H
#define FOREST_SAMPLERNUX_H

#include "util.h"
#include "typeparam.h"

class SamplerNux {
  // As with RankCount, unweighted sampling typically incurs very
  // small sample counts and row deltas.
  PackedT packed;

public:
  static PackedT delMask;
  static unsigned int rightBits;


  static void setMasks(IndexT nObs) {
    rightBits = Util::packedWidth(nObs);
    delMask = (1ull << rightBits) - 1;
  }
  

  static void unsetMasks() {
    delMask = 0;
    rightBits = 0;
  }


  /**
     @brief Constructor for external packed value.
   */
  SamplerNux(PackedT packed_) :
    packed(packed_) {
  }

  
  SamplerNux(IndexT delRow,
	     IndexT sCount) :
    packed(delRow | (static_cast<PackedT>(sCount) << rightBits)) {
  }

  
  /**
     @return difference in adjacent row numbers.  Always < nObs.
   */
  inline IndexT getDelRow() const {
    return packed & delMask;
  }
  

  /**
     @return sample count
   */  
  inline IndexT getSCount() const {
    return packed >> rightBits;
  }

  
  /**
     @brief Obtains sample count for external packed value.
   */
  static IndexT getSCount(PackedT packed) {
    return packed >> rightBits;
  }


  PackedT getPacked() const {
    return packed;
  }
};

#endif
