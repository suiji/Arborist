// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file runset.h

   @brief Definitions for the Run classes, which maintain predictor
   runs, especially factor-valued predictors.

   @author Mark Seligman
 */

#ifndef SPLIT_RUNSET_H
#define SPLIT_RUNSET_H

#include <vector>

#include "splitcoord.h"
#include "sumcount.h"
#include "algparam.h"

/**
   @brief  Runs only:  caches pre-computed workspace starting indices to
   economize on address recomputation during splitting.

   Run objects are allocated per-tree, and live throughout training.
*/
class RunSet {
  const SplitStyle style; // Splitting style, fixed by frontier class.
  IndexT wideRuns; // Sum of run sizes greater than max.
  vector<RunAccumT> runAccum;
  vector<double> rvWide;

public:

  /**
     @brief Constructor.

     @param nRow is the number of training rows:  inattainable offset.
  */
  RunSet(const class SplitFrontier* sf,
	 IndexT nRow);


  /**
     @brief Adds local run count to vector of safe counts.

     @return offset of run just appended.
   */
  IndexT addRun(const class SplitFrontier* splitFrontier,
		const class SplitNux* cand,
		PredictorT runCount);

  
  /**
     @brief Consolidates the safe count vector.

     Classification:  only wide run sets use the heap.
  */
  void setOffsets(const class SplitFrontier* sf);

  
  /**
     @brief Accessor for RunAccum at specified index.
   */
  inline RunAccumT* getAccumulator(PredictorT accumIdx) {
    return &runAccum[accumIdx];
  }


  auto getAccumCount() {
    return runAccum.size();
  }
  
  
  /**
     @brief Gets safe count associated with a given index.
   */
  auto getSafeCount(PredictorT accumIdx) const {
    return runAccum[accumIdx].getSafeCount();
  }


  vector<IndexRange> getRange(const class SplitNux& nux,
			      const struct CritEncoding& enc) const;


  /**
     @retrun SR index range of top run.
   */
  vector<IndexRange> getTopRange(const class SplitNux& nux,
				 const struct CritEncoding& enc) const;


  struct RunDump dumpRun(PredictorT accumIdx) const {
    return runAccum[accumIdx].dump();
  }

  

  /**
     @brief Accumulates sum of implicit LH (true-sense) slots.

     @return sum of implicit extents over LH runs.
   */
  IndexT getImplicitTrue(const class SplitNux* nux) const;
  

  /**
     @brief Sets bits corresponding to true-sense branch.

     Passes through to accumulator method.
   */
  void setTrueBits(const class SplitNux& nux,
		   class BV* splitBits,
		   size_t bitPos) const;

  
  /**
     @brief As above, but all observed bits.
   */
  void setObservedBits(const class SplitNux& nux,
		       class BV* splitBits,
		       size_t bitPos) const;


  PredictorT getRunCount(const class SplitNux* nux) const;


  void resetRunCount(PredictorT accumIdx,
		   PredictorT runCount);
 

  /**
     @brief Passes through to RunAccum method.
   */
  void topSlot(const class SplitNux* nux);


  /**
     @brief Dispatches candidate finalizer.
   */
  void updateAccum(const class SplitNux& cand);
};

#endif
