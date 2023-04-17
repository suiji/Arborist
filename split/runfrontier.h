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

#include "splitcoord.h"
#include "sumcount.h"
#include "algparam.h"
#include "runsig.h"

#include <vector>
#include <memory>
#include <algorithm>


/**
   @brief  Runs only:  caches pre-computed workspace starting indices to
   economize on address recomputation during splitting.
*/
class RunSet {
  PredictorT nAccum; ///> # of accumulators.
  vector<RunSig> runSig;

  // Non-binary categorical only:
  vector<IndexT> runWide; ///> Wide-run accumulator indices, ordered.
  vector<double> rvWide; ///> Random variates for sampling wide runs.

public:

  const SplitStyle style; // Splitting style, fixed by frontier class.

  /**
     @brief Constructor.
  */
  RunSet(const class SplitFrontier* sf);


  /**
     @brief Determines position of random-variate slice.

     Slices have an implicit width of RunAccum::maxWidth.
     
     @param sigIdx is the index of a wide-run accumulator.

     @return base variate position for accumulator.
   */
  const double* rvSlice(IndexT sigIdx) const;


  /**
     @brief Adds local run count to vector of safe counts.

     @return offset of run just appended.
   */
  IndexT preIndex(const class SplitFrontier* sf,
		  const class SplitNux& cand);


  void setSplit(class SplitNux& cand,
		vector<RunNux> runNux,
		const struct SplitRun& splitRun);

  
  /**
     @brief Consolidates the safe count vector.

     Classification:  only wide run sets use the heap.
  */
  void accumPreset(const class SplitFrontier* sf);


  const vector<RunNux>& getRunNux(const class SplitNux& cand) const;

  
  vector<IndexRange> getRunRange(const class SplitNux& nux,
				 const struct CritEncoding& enc) const;


  /**
     @retrun SR index range of top run.
   */
  vector<IndexRange> getTopRange(const class SplitNux& nux,
				 const struct CritEncoding& enc) const;


  /**
     @brief Accumulates sum of implicit LH (true-sense) slots.

     @return sum of implicit extents over LH runs.
   */
  IndexT getImplicitTrue(const class SplitNux& nux) const;
  

  /**
     @brief Sets bits corresponding to true-sense branch.

     Passes through to accumulator method.
   */
  void setTrueBits(const class InterLevel* interLevel,
		   const class SplitNux& nux,
		   class BV* splitBits,
		   size_t bitPos) const;

  
  /**
     @brief As above, but all observed bits.
   */
  void setObservedBits(const class InterLevel* interLevel,
		       const class SplitNux& nux,
		       class BV* splitBits,
		       size_t bitPos) const;


  PredictorT getRunCount(const class SplitNux* nux) const;


  void resetRunSup(PredictorT sigIdx,
		   PredictorT runCount);
 

  /**
     @brief Passes through to RunAccum method.
   */
  void topSlot(const class SplitNux* nux);


  /**
     @brief Updates chosen accumulator for encoding.
   */
  void accumUpdate(const class SplitNux& cand);


  vector<IndexRange> getRange(const class SplitNux& nux,
			      const struct CritEncoding& enc) const;
};

#endif
