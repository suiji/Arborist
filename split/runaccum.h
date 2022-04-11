// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file runaccum.h

   @brief Definitions for the Run classes, which maintain predictor
   runs, especially factor-valued predictors.

   @author Mark Seligman
 */

#ifndef SPLIT_RUNACCUM_H
#define SPLIT_RUNACCUM_H

#include <vector>

#include "splitcoord.h"
#include "sumcount.h"
#include "accum.h"
#include "bheap.h"
#include "runnux.h"

enum class SplitStyle;


/**
   RunAccums live only during a single level, from argmax pass one (splitting)
   through argmax pass two.  They accumulate summary information for split/
   predictor pairs anticipated to have two or more distinct runs.  RunAccums
   are not yet built for numerical predictors, which have so far been
   generally assumed to have dispersive values.

   Run lengths for a given predictor decrease, although not necessarily
   monotonically, with splitting.  Hence once a pair becomes a singleton, the
   fact is worth preserving for the duration of training.  Numerical predictors
   are assigned a nonsensical run length of zero, which is changed to a sticky
   value of unity, should a singleton be identified.  Run lengths are
   transmitted between levels duing restaging, which is the only phase to
   maintain a map between split nodes and their descendants.  Similarly, new
   singletons are very easy to identify when updating the partition.

   Other than the "bottom" value of unity, run lengths can generally only be
   known precisely by first walking the predictor codes.  Hence a conservative
   value is used for storage allocation.
*/
class RunAccum : public Accum {
  const PredictorT nCtg; // Non-zero iff classification.
  const PredictorT rcSafe; // Conservative run count.
  vector<RunNux> runZero; // SR block, partitioned by code.
  vector<BHPair<PredictorT>> heapZero; // Sorting workspace.
  vector<PredictorT> idxRank; // Slot rank, according to ad-hoc ordering.
  vector<double> cellSum; // Categorical:  run x ctg checkerboard.
  double* rvZero; // Non-binary wide runs:  random variates for sampling.

  PredictorT implicitSlot; // Which run, if any has no explicit SR range.
  PredictorT obsCount; // Actual number of runs employed by splitting.
  PredictorT baseTrue; // Base of true-run slots.
  PredictorT runsTrue; // Count of true-run slots.
  PredictorT splitToken; // Cut or bits.
  IndexT implicitTrue; // # implicit true-sense indices:  post-encoding.


  /**
     @brief Subtracts contents of top run from accumulators and sets its
     high terminal index.

     @param idxEnd is the high terminal index of the run.
   */
  inline void endRun(RunNux& nux,
		     IndexT idxEnd) {
    sCount -= nux.sCount;
    sum -= nux.sum;
    nux.endRange(idxEnd);
  }
  
  
  /**
     @brief Appends a run for the dense rank using residual values.

     @param sumSlic is the per-category response over the frontier node.
  */
  PredictorT appendImplicit(PredictorT runIdx,
			    const vector<double>& sumSlice = vector<double>(0));


  /**
     @brief Looks up run parameters by indirection through output vector.
     
     N.B.:  should not be called with a dense run.

     @return index range associated with run.
  */
  IndexRange getBounds(PredictorT slot) const {
    return runZero[slot].getRange();
  }

  
  /**
     @brief Accumulates runs for regression.
   */
  PredictorT regRuns();

  
  /**
     @brief Determines split having highest weighted variance.

     Runs initially sorted by mean response.
   */
  void maxVar();
  

  /**
     @brief Accumulates runs for classification.

     @param sumSlice is the per-category response decomposition.

     @return count of runs encountered.
  */
  PredictorT ctgRuns(const class SFCtg* sf,
		     const class SplitNux* cand);

  

  /**
     @brief Gini-based splitting for categorical response and predictor.

     Nodes are now represented compactly as a collection of runs.
     For each node, subsets of these collections are examined, looking for the
     Gini argmax beginning from the pre-bias.

     Iterates over nontrivial subsets, coded by unsigneds as bit patterns.  By
     convention, the final run is incorporated into RHS of the split, if any.
     Excluding the final run, then, the number of candidate LHS subsets is
     '2^(obsCount-1) - 1'.

     @param ctgSum is the per-category sum of responses.
  */
  void ctgGini(const class SFCtg* sf, const SplitNux* cand);

  
  /**
     @brief As above, but specialized for binary response.
   */
  void binaryGini(const class SFCtg* sf, const SplitNux* cand);

  
  /**
     @brief Sorts by random variate ut effect sampling w/o replacement.

     @param runCount is the total number of runs.
   */
  void heapRandom(PredictorT runCount);

  
  /**
     @brief Sorts by probability, binary response.
   */
  void heapBinary();

  
  /**
     @brief Sorts by mean response.
   */
  void heapMean();

public:
  static constexpr unsigned int maxWidth = 10; // Algorithmic threshold.

  
  RunAccum(const class SplitFrontier* splitFrontier,
	   const class SplitNux* cand,
	   SplitStyle style,
	   PredictorT rcSafe_);

  
  /**
     @brief Counts the number of wide level extents.

     @return level extent iff beyond the threshold else zero.
   */
  IndexT countWide() const {
    return rcSafe > maxWidth ? rcSafe : 0;
  }


  /**
     @return extent of implicit slot, if in true branch, else zero.
   */
  IndexT getImplicitCut() const {
    return (implicitSlot >= baseTrue && implicitSlot < baseTrue + runsTrue) ? getExtent(implicitSlot) : 0;
  }
  

  /**
     @brief Overwrites leading slots with sampled subset of runs.
  */
  PredictorT deWide(PredictorT runCount);


  /**
     @brief Reorders the per-category response decomposition to compensate for run reordering.

     @param leadCount is the number of leading runs reordered.
   */
  void ctgReorder(PredictorT leadCount);


  /**
     @breif Static entry for regression splitting.
   */
  static void split(const class SFReg* sf,
		    class SplitNux* cand);

  
  /**
     @brief Private entry for regression splitting.
   */
  void splitReg(class SplitNux* cand);


  /**
     @brief Static entry for classification splitting.
   */
  static void split(const class SFCtg* sf,
		    class SplitNux* cand);
  

  /**
     @brief Private entry for categorical splitting.
   */
  void splitCtg(const class SFCtg* sf,
		class SplitNux* cand);


  /**
     @brief Depopulates the heap associated with a pair and places sorted ranks into rank vector.

     @param pop is the number of elements to pop from the heap.
  */
  void slotReorder(PredictorT pop = 0);


  /**
     @brief Revises slot or bit contents for argmax accumulator.

     @param cand is a successful splitting candidate.
   */
  void update(const SplitNux& cand,
	      SplitStyle style);

  
  /**
     @brief Updates local vector bases with their respective offsets,
     addresses, now known.

     @param[in, out] rvOff accumulates wide level counts.
  */
  void  reWide(vector<double>& rvWide,
	       IndexT& rvOff);


  void initReg(IndexT runLeft,
	       PredictorT runIdx);

  
  double* initCtg(IndexT runLeft,
		  PredictorT runIdx);

  
  /**
     @brief As above, but skips masked SR indices.

     @param maskSense indicates whether to screen set or unset mask.
   */  
  PredictorT regRunsMasked(const class BranchSense* branchSense,
			   IndexT edgeRight,
			   IndexT edgeLeft,
			   bool maskSense);


  /**
     @brief Writes to heap arbitrarily:  sampling w/o replacement.

     @param runCount is the total number of runs.

     @param leadCount is the number of leading elments to reorder.
  */
  void orderRandom(PredictorT runCount,
		   PredictorT leadCount);


  /**
     @brief Writes to heap, weighting by slot mean response.
  */
  void orderMean();


  /**
     @brief Writes to heap, weighting by category-1 probability.
  */
  void orderBinary();


  /**
     @brief Sets splitting token for possible elaboration.
   */
  inline void setToken(PredictorT token) {
    splitToken = token;
  }
  

  /**
     @brief Getter for runCount.

     @return run count.
   */
  inline auto getRunCount() const {
    return obsCount;
  }


  inline auto getImplicitTrue() const {
    return implicitTrue;
  }
  

  inline void resetRunCount(PredictorT runCount) {
    this->obsCount = runCount;
  }
  

  /**
     @brief Safe-count getter.
   */
  inline auto getSafeCount() const {
    return rcSafe;
  }


  /**
     @brief Accumulates contents at position referenced by a given index.

     @param slot is a run index.

     @param sCount[in, out] accumulates sample count using the output position.

     @param sum[in, out] accumulates the sum using the output position.
   */
  inline void sumAccum(PredictorT slot,
		       IndexT& sCount,
		       double& sum) const {
    runZero[slot].accum(sCount, sum);
  }


  /**
     @brief Resets top index and contents, if applicable.
     
     @param runStart is the previous top position.

     @param runIdx is the index from which to copy the top position.
   */
  inline void reset(PredictorT runStart,
		    PredictorT runIdx) {
    if (runIdx != obsCount) {
      runZero[runStart] = runZero[runIdx]; // New top value.
      obsCount = runStart + 1;
    }
    else { // No new top, run-count restored.
      obsCount = runStart;
    }
  }


  /**
     @return checkerboard value at slot for category.
   */
  inline double getCellSum(PredictorT runIdx,
			   PredictorT yCtg) const {
    return cellSum[runIdx * nCtg + yCtg];
  }


  /**
     @brief Subtracts a run's per-category responses from the current run.

     @param sumSlice decomposes the response by category.
   */
  inline void residCtg(const vector<double>& sumSlice,
		       PredictorT runIdx);
  

  /**
     @brief Accumulates the two binary response sums for a run.

     @param slot is a run index.

     @param[in, out] sum0 accumulates the response at code 0.

     @param[in, out] sum1 accumulates the response at code 1.

     @return true iff next run sufficiently different from this.
   */
  inline bool accumBinary(PredictorT slot,
			  double& sum0,
			  double& sum1) {
    sum0 += getCellSum(slot, 0);
    double cell1 = getCellSum(slot, 1);
    sum1 += cell1;

    // Two runs are deemed significantly different if their sample
    // counts differ. If identical, then checks whether the response
    // sums differ by some measure.
    PredictorT slotNext = slot+1;
    return (runZero[slot].sCount != runZero[slotNext].sCount) ||  getCellSum(slotNext, 1) > cell1;
  }


  /**
    @brief Outputs sample and index counts at a given slot.

    @param slot is the run slot in question.

    @return total SR index count subsumed.
  */
  inline IndexT getExtent(PredictorT slot) const {
    return runZero[slot].range.getExtent();
  }


  /**
     @return level code at specified slot.
   */
  auto getCode(PredictorT slot) const {
    return runZero[slot].code;
  }

  
  auto getSum(PredictorT slot) const {
    return runZero[slot].sum;
  }

  
  auto getSCount(PredictorT slot) const {
    return runZero[slot].sCount;
  }

  
  /**
     @brief Decodes bit vector of argmax factor.

     @param lhBits encodes sampled LH/RH slot indices as on/off bits, respectively.

     @param invertTest indicates whether to complement true branch bits:  EXIT.
  */
  void leadBits(bool invertTest);


  /**
     @brief Determines the complement of a bit pattern of fixed size.

     Equivalent to  (~subset << (32 - obsCount))) >> (32 - obsCount).
     
     @param subset is a collection of obsCount-many bits.

     @return bit (ones) complement of subset.
  */
  inline unsigned int slotComplement(unsigned int subset) const {
    return (1 << obsCount) - (subset + 1);
  }


  /**
     @brief Determines Gini of a subset of runs encoded as bits.

     @param sumSlice decomposes the partition node response by category.

     @param subset bit-encodes a collection of runs.

     N.B.:  Gini value should be symmetric w.r.t. fixed-size complements.
     
     N.B.:  trivial subsets, beside being uninformative, may precipitate
     division by zero.

     @return Gini coefficient of subset.
   */
  double subsetGini(const vector<double>& sumSlice,
		    unsigned int subset) const;


  /**
     @brief Emits the left-most codes as true-branch bit positions.

     True codes are enumerated from the left, by convention.  Implicit runs are
     guranteed not to lie on the left.
   */
  void setTrueBits(class BV* splitBits,
		   size_t bitPos) const;


  /**
     @brief Reports the factor codes observed at the node.
   */
  void setObservedBits(class BV* splitBits,
		       size_t bitPos) const;
  
  /**
     @brief Establishes cut position of argmax factor.

     @param invertTest indicates whether to complement true-branch bits:  Exit.
  */
  void leadSlots(bool invertTest);


  /**
     @brief Appends a single slot to the lh set.
   */
  void topSlot();


  /**
     @return implicit count associated with a slot.
   */
  IndexT getImplicitExtent(PredictorT slot) const {
    return runZero[slot].isImplicit() ? getExtent(slot) : 0;
  }

  
  /**
     @return vector of block ranges associated with encoding.
   */
  vector<IndexRange> getRange(const class CritEncoding& enc) const;


  vector<IndexRange> getRange(PredictorT slotStart,
			      PredictorT slotEnd) const;
  

  /**
     @return top-most block range associated with encoding.
   */
  vector<IndexRange> getTopRange(const class CritEncoding& enc) const;


  struct RunDump dump() const;
};


/**
   @brief Accumulates diagnostic statistics over the run vector.
 */
struct RunDump {
  IndexT sCount;
  double sum;
  vector<PredictorT> code;

  /**
     @brief Populates dump over specified vector range.
   */
  RunDump(const RunAccum* runAccum,
	  PredictorT runStart,
	  PredictorT runCount) : sCount(0), sum(0.0), code(vector<PredictorT>(runCount)) {
    for (PredictorT rc = runStart; rc < runStart + runCount; rc++) {
      code[rc] = runAccum->getCode(rc);
      sCount += runAccum->getSCount(rc);
      sum += runAccum->getSum(rc);
    }
  }
};

#endif
