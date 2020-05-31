/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file accum.cc

   @brief Methods implementing a generic split accumulator.

   @author Mark Seligman

 */

#include "accum.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "branchsense.h"


Accum::Accum(const SplitFrontier* splitFrontier,
	     const SplitNux* cand) :
  sampleRank(splitFrontier->getPredBase(cand)),
  sampleIndex(splitFrontier->getBufferIndex(cand)),
  rankDense(splitFrontier->getDenseRank(cand)),
  idxStart(cand->getIdxStart()),
  idxEnd(cand->getIdxEnd()),
  sumCand(cand->getSum()),
  sCountCand(cand->getSCount()),
  sCount(sCountCand),
  sum(sumCand),
  info(cand->getInfo()) {
}


bool Accum::findEdge(const BranchSense* branchSense,
		     bool leftward,
		     IndexT idxTerm,
		     bool sense,
		     IndexT& edge) const {
  // Breaks out and returns true iff matching-sense sample found.
  if (leftward) { // Decrement to start.
    for (edge = idxTerm; edge > idxStart; edge--) {
      if (branchSense->isExplicit(sampleIndex[edge]) == sense) {
	return true;
      }
    }
    if (branchSense->isExplicit(sampleIndex[edge]) == sense) {
      return true;
    }
  }
  else { // Increment to end.
    for (edge = idxTerm; edge <= idxEnd; edge++) {
      if (branchSense->isExplicit(sampleIndex[edge]) == sense) {
	return true;
      }
    }
  }

  return false; // No match.
}


