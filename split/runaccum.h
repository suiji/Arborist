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
  const IndexT runCount; ///< # runs counted by repartitioning.
  IndexT sampledRuns; ///< # sampled runs.
  vector<RunNux> runZero; // SR block, partitioned by code.
  vector<BHPair<PredictorT>> heapZero; // Sorting workspace.
  vector<PredictorT> idxRank; // Slot rank, according to ad-hoc ordering.
  vector<double> cellSum; // Categorical:  run x ctg checkerboard.
  double* rvZero; // Non-binary wide runs:  random variates for sampling.

  PredictorT implicitSlot; // Which run, if any has no explicit SR range.
  PredictorT baseTrue; // Base of true-run slots.
  PredictorT runsTrue; // Count of true-run slots.
  PredictorT splitToken; // Cut or bits.
  IndexT implicitTrue; // # implicit true-sense indices:  post-encoding.
  vector<double> ctgSum; ///> per-category sum of responses in node.

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
     @brief Initializes a run from residual values.

     @param sumSlice is the per-category response over the frontier node.
  */
  void applyResidual(const vector<double>& sumSlice = vector<double>(0));


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
  void regRuns(const struct SFReg* sf,
	       const SplitNux& cand);


  /**
     @brief As above, but also tracks a residual slot.
   */
  void regRunsImplicit(const struct SFReg* sf,
		       const SplitNux& cand);

  
  /**
     @brief Determines split having highest weighted variance.

     Runs initially sorted by mean response.
   */
  void maxVar();
  

  /**
     @brief Accumulates runs for classification.

     @param sumSlice is the per-category response decomposition.
  */
  void ctgRuns(const class SFCtg* sf,
	       const class SplitNux& cand);

  
  /**
     @brief As above, but also tracks a residual slot
   */
  void ctgRunsImplicit(const class SFCtg* sf,
		       const class SplitNux& cand);


  /**
     @brief Gini-based splitting for categorical response and predictor.

     Nodes are now represented compactly as a collection of runs.
     For each node, subsets of these collections are examined, looking for the
     Gini argmax beginning from the pre-bias.

     Iterates over nontrivial subsets, coded by unsigneds as bit patterns.  By
     convention, the final run is incorporated into RHS of the split, if any.
     Excluding the final run, then, the number of candidate LHS subsets is
     '2^(sampledRuns-1) - 1'.

     @param ctgSum is the per-category sum of responses.
  */
  void ctgGini(const class SFCtg* sf, const SplitNux& cand);

  
  /**
     @brief As above, but specialized for binary response.
   */
  void binaryGini(const class SFCtg* sf, const SplitNux& cand);

  
  /**
     @brief Sorts by random variate ut effect sampling w/o replacement.

     @param nRun is the total number of runs.
   */
  void heapRandom();

  
  /**
     @brief Sorts by probability, binary response.
   */
  void heapBinary();

  
  /**
     @brief Sorts by mean response.
   */
  void heapMean();

  
  /**
     @brief Determines whether run denotes a residual.

     Redidual runs distinguished by out-of-bound range.
   */
  bool isImplicit(const RunNux& runNux) const {
    return runNux.range.idxStart >= obsEnd;
  }
  

public:
  static constexpr unsigned int maxWidth = 10; // Algorithmic threshold.


  /**
   */
  RunAccum(const class SplitFrontier* splitFrontier,
	   const class SplitNux& cand,
	   SplitStyle style);


  /**
     @brief Counts the number of wide level extents.

     @return level extent iff beyond the threshold else zero.
   */
  IndexT countWide() const {
    return runCount > maxWidth ? runCount : 0;
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
  PredictorT deWide();


  /**
     @brief Reorders the per-category response decomposition to compensate for run reordering.

     @param leadCount is the number of leading runs reordered.
   */
  void ctgReorder(PredictorT leadCount);


  /**
     @breif Static entry for regression splitting.
   */
  static void split(const struct SFReg* sf,
		    class SplitNux& cand);

  
  /**
     @brief Private entry for regression splitting.
   */
  void splitReg(const struct SFReg* sf,
		class SplitNux& cand);


  /**
     @brief Static entry for classification splitting.
   */
  static void split(const class SFCtg* sf,
		    class SplitNux& cand);
  

  /**
     @brief Private entry for categorical splitting.
   */
  void splitCtg(const class SFCtg* sf,
		class SplitNux& cand);


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
  void regRunsMasked(const struct SFReg* sf,
		     const SplitNux& cand,
		     const class BranchSense* branchSense,
		     IndexT edgeRight,
		     IndexT edgeLeft,
		     bool maskSense);


  /**
     @brief Writes to heap arbitrarily:  sampling w/o replacement.

     @param leadCount is the number of leading elments to reorder.
  */
  void orderRandom(PredictorT leadCount);


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
     @brief Getter for nRun.

     @return run count.
   */
  inline auto getRunCount() const {
    return sampledRuns;
  }


  inline auto getImplicitTrue() const {
    return implicitTrue;
  }
  

  inline void resetRunCount(PredictorT nRun) {
    this->sampledRuns = nRun;
  }
  

  /**
     @brief Safe-count getter.
   */
  inline auto getSafeCount() const {
    return runCount;
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
    if (runIdx != sampledRuns) {
      runZero[runStart] = runZero[runIdx]; // New top value.
      sampledRuns = runStart + 1;
    }
    else { // No new top, run-count restored.
      sampledRuns = runStart;
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
  inline void residCtg(const vector<double>& sumSlice);


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
     @return representative observation index within specified slot.
   */
  auto getObs(PredictorT slot) const {
    return runZero[slot].range.idxStart;
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

     Equivalent to  (~subset << (32 - sampledRuns))) >> (32 - sampledRuns).
     
     @param subset is a collection of sampledRuns-many bits.

     @return bit (ones) complement of subset.
  */
  inline unsigned int slotComplement(unsigned int subset) const {
    return (1 << sampledRuns) - (subset + 1);
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
  double subsetGini(unsigned int subset) const;


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
    return isImplicit(runZero[slot]) ? getExtent(slot) : 0;
  }

  
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
