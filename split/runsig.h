// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file runsig.h

   @brief Minimal representation of partitioned predictor runs.

   @author Mark Seligman
 */

#ifndef SPLIT_RUNSIG_H
#define SPLIT_RUNSIG_H

#include "sumcount.h"

#include <vector>

enum class SplitStyle;


/**
   @brief Accumulates statistics for runs of factors having the same internal code.

   Allocated in bulk by Fortran-style workspace, the RunSet.
 */
struct RunNux {
  SumCount sumCount; ///< Sum, sample count of associated responses.
  IndexRange obsRange; ///< Observation indices.


  /**
     @brief Initialzier for subsequent accumulation.
  */
  inline void init() {
    sumCount = SumCount();
  }


  inline void startRange(IndexT idxStart) {
    obsRange.idxStart = idxStart;
  }
  

  inline void endRange(IndexT idxEnd) {
    obsRange.idxExtent = idxEnd - obsRange.idxStart + 1;
  }

  
  /**
     @brief Range accessor.  N.B.:  Should not be invoked on dense
     run, as 'start' will hold a reserved value.

     @return range of indices subsumed by run.
   */
  inline IndexRange getRange() const {
    return obsRange;
  }


  /**
     @brief Accumulates run contents into caller.
  */
  inline void accum(SumCount& scAccum) const {
    scAccum += sumCount;
  }

  /**
     @brief Subtracts contents of top run and sets range end.

     @Paramol idxEnd is the high terminal index of the run.
   */
  inline void endRun(SumCount& scExplicit,
		     IndexT idxEnd) {
    scExplicit -= sumCount;
    endRange(idxEnd);
  }


  /**
     @brief Initializes as residual.
  */
  inline void setResidual(const SumCount& scImplicit,
			  IndexT obsEnd,
			  IndexT extent) {
    sumCount = scImplicit;
    obsRange = IndexRange(obsEnd, extent);
  }
};


/**
   @brief Minimal information needed to convey a run-based split.
 */
struct RunSig {
  // Initialized by splitting:
  vector<RunNux> runNux;
  PredictorT splitToken; ///< Cut or bits.

  PredictorT runsSampled; ///< # ctg participating in split.
  PredictorT baseTrue; ///< Base of true-run slots.
  PredictorT runsTrue; ///< Count of true-run slots.
  IndexT implicitTrue; ///< # implicit true-sense indices:  post-encoding.
  IndexT runSup; ///< # active runs, <= runNux size; top splits only.


  RunSig() = default;


  RunSig(vector<RunNux> runNux_,
	 PredictorT splitToken_,
	 PredictorT runsSampled_);


  inline void resetRunSup(PredictorT nRun) {
    this->runSup = nRun;
  }


  /**
     @brief Resets top index and contents, if applicable.
     
     @param runStart is the previous top position.

     @param runIdx is the index from which to copy the top position.
   */
  inline void reset(PredictorT runStart,
		    PredictorT runIdx) {
    if (runIdx != runNux.size()) {
      runNux[runStart] = runNux[runIdx]; // New top value.
      runSup = runStart + 1;
    }
    else { // No new top, run-count restored.
      runSup = runStart;
    }
  }


  /**
     @brief Looks up run parameters by indirection through output vector.
     
     N.B.:  does not apply to residual runs.

     @return index range associated with run.
  */
  IndexRange getBounds(PredictorT slot) const {
    return runNux[slot].getRange();
  }



  /**
    @brief Outputs sample and index counts at a given slot.

    @param slot is the run slot in question.

    @return total SR index count subsumed.
  */
  inline IndexT getExtent(PredictorT slot) const {
    return runNux[slot].obsRange.getExtent();
  }


  /**
     @brief Revises slot or bit contents for criterion.

     @param cand is the associated splitting candidate.
   */
  void updateCriterion(const class SplitNux& cand,
		       SplitStyle style);


  /**
     @brief Obtains number of runs in play.

     @return size of runNux.
   */
  inline auto getRunCount() const {
    return runNux.size();
  }


  inline auto getImplicitTrue() const {
    return implicitTrue;
  }


  /**
     @brief Decodes bit vector of argmax factor.

     @param lhBits encodes sampled LH/RH slot indices as on/off bits, respectively.

     @param invertTest indicates whether to complement true branch bits:  EXIT.
  */
  void leadBits(const class SplitNux& nux);


  /**
     @brief Determines the complement of a bit pattern of fixed size.

     Equivalent to  (~subset << (32 - runNux.size()))) >> (32 - runNux.size()).
     
     @param subset is a collection of runNux.size()-many bits.

     @return bit (ones) complement of subset.
  */
  inline unsigned int slotComplement(unsigned int subset) const {
    return (1 << runNux.size()) - (subset + 1);
  }


  /**
     @return representative observation index within specified slot.
   */
  auto getObs(PredictorT slot) const {
    return runNux[slot].obsRange.idxStart;
  }


  /**
     @brief Emits the left-most codes as true-branch bit positions.

     True codes are enumerated from the left, by convention.  Implicit runs are
     guranteed not to lie on the left.
   */
  void setTrueBits(const class InterLevel* interLevel,
		   const class SplitNux& nux,
		   class BV* splitBits,
		   size_t bitPos) const;


  /**
     @brief Reports the factor codes observed at the node.
   */
  void setObservedBits(const class InterLevel* interLevel,
		       const class SplitNux& nux,
		       class BV* splitBits,
		       size_t bitPos) const;
  
  /**
     @brief Establishes cut position of argmax factor.

     @param nux encodes whether test is to be inverted.
  */
  void leadSlots(const class SplitNux& nux);


  /**
     @brief Appends a single slot to the lh set.

     @param nux encodes implicit slot, if any.
   */
  void topSlot(const SplitNux& nux);


  /**
     @return implicit count associated with a slot.
   */
  IndexT getImplicitExtent(const SplitNux& cand,
			   PredictorT slot) const;


  /**
     @return vector of block ranges associated with encoding.
   */
  vector<IndexRange> getRange(const struct CritEncoding& enc) const;


  vector<IndexRange> getRange(PredictorT slotStart,
			      PredictorT slotEnd) const;
  

  /**
     @return top-most block range associated with encoding.
   */
  vector<IndexRange> getTopRange(const struct CritEncoding& enc) const;
};


#endif
