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


class CutSet {
  vector<class CutAccum> cutAccum;
  vector<struct CutSig> cutSig; // EXIT

public:
  CutSet();


  CutSig getCut(IndexT accumIdx) const;


  /**
     @brief Same as above, but looks up from nux accum index.
   */
  struct CutSig getCut(const SplitNux& nux) const;

  
  void setCut(IndexT accumIdx, const struct CutSig& sig);
  
  
  IndexT addCut(const class SplitFrontier* splitFrontier,
		const class SplitNux* cand);

  
  IndexT getAccumCount() const {
    return cutAccum.size();
  }
  

  void write(const class SplitNux* nux,
	     const class CutAccum* accum);

  
  /**
     @return true iff cut associated with split has left sense.
   */
  bool leftCut(const class SplitNux* nux) const;


  /**
     @brief Sets the sense of a given cut.
   */
  void setCutSense(IndexT cutIdx,
		   bool sense);

  double getQuantRank(const class SplitNux* nux) const;


  IndexT getIdxRight(const class SplitNux* nux) const;

  
  IndexT getIdxLeft(const class SplitNux* nux) const;

  
  IndexT getImplicitTrue(const class SplitNux* nux) const;
};

#endif
