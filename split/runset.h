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
  vector<RunAccumT> runAccum;
  vector<double> rvWide;
  const SplitStyle style; // Splitting style, fixed by frontier class.


  /**
     @brief Classification:  only wide run sets use the heap.
  */
  void offsetsCtg();
  

public:
  const PredictorT nCtg;  // Response cardinality; zero iff numerical.

  /**
     @brief Constructor.

     @param nCtg_ is the response cardinality.

     @param nRow is the number of training rows:  inattainable offset.
  */
  RunSet(SplitStyle factorStyle,
	 PredictorT nCtg_,
	 IndexT nRow);


  /**
     @brief Consolidates the safe count vector.
   */
  void setOffsets();

  
  /**
     @brief Adds local run count to vector of safe counts.

     @return offset of run just appended.
   */
  IndexT addRun(const class SplitFrontier* splitFrontier,
		const class SplitNux* cand,
		PredictorT runCount);

  
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


  vector<IndexRange> getRange(const class SplitNux* nux,
				 const class CritEncoding& enc) const;

  
  /**
     @retrun SR index range of top run.
   */
  IndexRange getTopRange(const class SplitNux* nux,
			 const class CritEncoding& enc) const;



  struct RunDump dumpRun(PredictorT accumIdx) const {
    return runAccum[accumIdx].dump();
  }

  

  /**
     @brief Accumulates sum of implicit LH (true-sense) slots.

     @return sum of implicit extents over LH runs.
   */
  IndexT getImplicitTrue(const class SplitNux* nux) const;
  

  /**
     @return vector of codes corresponding to true-sense branch.
   */
  vector<PredictorT> getTrueBits(const class SplitNux* nux) const;


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
  void updateAccum(const class SplitNux* cand);
};

#endif
