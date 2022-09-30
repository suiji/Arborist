/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_CUTSET_H
#define SPLIT_CUTSET_H

/**
   @file cutset.h

   @brief Manages cut accumulutators as distinct workspace.

   @author Mark Seligman

 */

#include "typeparam.h"
#include "accum.h"

#include <vector>


/**
   @brief Minimal information needed to reconstruct cut.
 */
struct CutSig {
  // In CART-like implementations, obsLeft and obsRight are adjacent.
  IndexT obsLeft; ///< sup of left Obs indices.
  IndexT obsRight;  ///< inf of right Obs indices.
  IndexT implicitTrue; ///< # implicit Obs indices associated with true sense.
  double quantRank; ///< Interpolated cut rank.
  bool cutLeft; ///< True iff cut encodes left portion.

  CutSig(const IndexRange& idxRange) :
    obsLeft(idxRange.getStart()),
    obsRight(idxRange.getEnd() - 1),
    cutLeft(true) { // Default.
  }

  CutSig() :
    cutLeft(true) {
  }


  void write(const class InterLevel* interLevel,
	     const class SplitNux& nux,
	     const class CutAccum& accum);
};


class CutSet {
  IndexT nAccum;
  vector<CutSig> cutSig;

public:
  CutSet() = default;


  /**
     @brief Allocates cutSet vector.
   */
  void accumPreset();

  
  CutSig getCut(IndexT sigIdx) const;


  /**
     @brief Same as above, but looks up from nux accum index.
   */
  CutSig getCut(const SplitNux& nux) const;

  
  void setCut(IndexT sigIdx, const CutSig& sig);


  IndexT preIndex() {
    return nAccum++;
  }
  

  void write(const class InterLevel* ofFront,
	     const class SplitNux& nux,
	     const class CutAccum& accum);

  
  /**
     @return true iff cut associated with split has left sense.
   */
  bool leftCut(const class SplitNux& nux) const;


  /**
     @brief Sets the sense of a given cut.
   */
  void setCutSense(IndexT cutIdx,
		   bool sense);


  double getQuantRank(const class SplitNux& nux) const;


  IndexT getIdxRight(const class SplitNux& nux) const;

  
  IndexT getIdxLeft(const class SplitNux& nux) const;

  
  IndexT getImplicitTrue(const class SplitNux& nux) const;
};

#endif
