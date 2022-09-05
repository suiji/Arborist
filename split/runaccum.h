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
protected:
  vector<RunNux> runNux; ///< obs block, partitioned by code.
  IndexT runSup; ///< # active runs, <= runNux size.
  vector<BHPair<PredictorT>> heapZero; ///< Sorting workspace.
  PredictorT implicitSlot; ///< Run, if any, without explicit obs range.

  // Post-splitting values.
  PredictorT baseTrue; ///> Base of true-run slots.
  PredictorT runsTrue; ///> Count of true-run slots.
  PredictorT splitToken; ///> Cut or bits.
  IndexT implicitTrue; ///> # implicit true-sense indices:  post-encoding.
  
  /**
     @brief Subtracts contents of top run from accumulators and sets its
     high terminal index.

     @param idxEnd is the high terminal index of the run.
   */
  inline void endRun(RunNux& nux,
		     SumCount& scExplicit,
		     IndexT idxEnd) {
    scExplicit.sCount -= nux.sCount;
    scExplicit.sum -= nux.sum;
    nux.endRange(idxEnd);
  }
  
  
  /**
     @brief Initializes a run from residual values.
  */
  void applyResidual(const SumCount& scResidual);


  /**
     @brief Looks up run parameters by indirection through output vector.
     
     N.B.:  should not be called with a dense run.

     @return index range associated with run.
  */
  IndexRange getBounds(PredictorT slot) const {
    return runNux[slot].getRange();
  }

  
  /**
     @brief Accumulates runs for regression.
   */
  void regRuns(const SplitNux& cand);


  /**
     @brief As above, but also tracks a residual slot.
   */
  void regRunsImplicit(const SplitNux& cand);

  
  /**
     @brief Determines split having highest weighted variance.

     Runs initially sorted by mean response.

     @return gain in weighted variance.
   */
  double maxVar();

  
  /**
     @brief Sorts by mean response.
   */
  void heapMean();

  
  /**
     @brief Determines whether run denotes a residual.

     Redidual runs distinguished by out-of-bound range.
   */
  bool isImplicit(const RunNux& runNux) const {
    return runNux.obsRange.idxStart >= obsEnd;
  }
  

public:
  static constexpr unsigned int maxWidth = 10; // Algorithmic threshold.


  /**
   */
  RunAccum(const class SplitFrontier* splitFrontier,
	   const class SplitNux& cand,
	   const class RunSet* runSet);


  /**
     @return extent of implicit slot, if in true branch, else zero.
   */
  IndexT getImplicitCut() const {
    return (implicitSlot >= baseTrue && implicitSlot < baseTrue + runsTrue) ? getExtent(implicitSlot) : 0;
  }
  

  /**
     @brief Determines whether run count must be truncated.

     @return true iff run count exceeds maximum.
   */
  static bool ctgWide(const class SplitFrontier* sf,
		      const class SplitNux& cand);


  /**
     @brief Depopulates the heap associated with a pair and places sorted ranks into rank vector.
  */
  void slotReorder();


  /**
     @brief Revises slot or bit contents for argmax accumulator.

     @param cand is a successful splitting candidate.
   */
  void update(const SplitNux& cand,
	      SplitStyle style);


  void initReg(IndexT runLeft,
	       PredictorT runIdx);

  
  /**
     @brief As above, but skips masked SR indices.

     @param maskSense indicates whether to screen set or unset mask.
   */  
  void regRunsMasked(const SplitNux& cand,
		     const class BranchSense* branchSense,
		     IndexT edgeRight,
		     IndexT edgeLeft,
		     bool maskSense);


  /**
     @brief Writes to heap, weighting by slot mean response.
  */
  void orderMean();


  /**
     @brief Sets splitting token for possible elaboration.
   */
  inline void setToken(PredictorT token) {
    splitToken = token;
  }
  

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
  

  inline void resetRunSup(PredictorT nRun) {
    this->runSup = nRun;
  }


  /**
     @brief Accumulates contents at position referenced by a given index.

     @param slot is a run index.

     @param scAccum[in, out] accumulates sample count and sum.
   */
  inline void sumAccum(PredictorT slot,
		       SumCount& scAccum) const {
    runNux[slot].accum(scAccum);
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
    @brief Outputs sample and index counts at a given slot.

    @param slot is the run slot in question.

    @return total SR index count subsumed.
  */
  inline IndexT getExtent(PredictorT slot) const {
    return runNux[slot].obsRange.getExtent();
  }


  /**
     @return representative observation index within specified slot.
   */
  auto getObs(PredictorT slot) const {
    return runNux[slot].obsRange.idxStart;
  }


  auto getSum(PredictorT slot) const {
    return runNux[slot].sum;
  }

  
  auto getSCount(PredictorT slot) const {
    return runNux[slot].sCount;
  }

  
  /**
     @brief Decodes bit vector of argmax factor.

     @param lhBits encodes sampled LH/RH slot indices as on/off bits, respectively.

     @param invertTest indicates whether to complement true branch bits:  EXIT.
  */
  void leadBits(bool invertTest);


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
    return isImplicit(runNux[slot]) ? getExtent(slot) : 0;
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


class RunAccumReg : public RunAccum {
public:
  RunAccumReg(const class SFReg* sfReg,
	      const class SplitNux& cand,
	      const class RunSet* runSet);


  
  /**
     @breif Static entry for regression splitting.
   */
  static void split(const class SFReg* sfReg,
		    class RunSet* runSet,
		    class SplitNux& cand);

  /**
     @brief Private splitting entry.
   */
  double split();
};



class RunAccumCtg : public RunAccum {
  const PredictorT nCtg;
  CtgNux ctgNux;

  // Initialized as a side-effect of RunNux construction:
  vector<double> runSum; ///>  run x ctg checkerboard.


  void sampleRuns(const class RunSet* runSet,
		  const class SplitNux& cand);

public:

  RunAccumCtg(const class SFCtg* sfCtg,
	      const class SplitNux& cand,
	      const class RunSet* runSet);

  /**
     @return checkerboard value at slot for category.
   */
  inline double getRunSum(PredictorT runIdx,
			   PredictorT yCtg) const {
    return runSum[runIdx * nCtg + yCtg];
  }


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
    sum0 += getRunSum(slot, 0);
    double cell1 = getRunSum(slot, 1);
    sum1 += cell1;

    // Two runs are deemed significantly different if their sample
    // counts differ. If identical, then checks whether the response
    // sums differ by some measure.
    PredictorT slotNext = slot+1;
    return (runNux[slot].sCount != runNux[slotNext].sCount) ||  getRunSum(slotNext, 1) > cell1;
  }


  /**
     @brief Writes to heap, weighting by category-1 probability.
  */
  void orderBinary();


  /**
     @brief Sorts by probability, binary response.
   */
  void heapBinary();


  /**
     @brief Static entry for classification splitting.
   */
  static void split(const class SFCtg* sf,
		    class RunSet* runSet,
		    class SplitNux& cand);
  

  double* initCtg(IndexT runLeft,
		  PredictorT runIdx);


  /**
     @brief Subtracts a run's per-category responses from the current run.
   */
  inline void residCtg();


  /**
     @brief Private entry for categorical splitting.
   */
  double split();


  /**
     @brief Accumulates runs for classification.

     @param sumSlice is the per-category response decomposition.
  */
  void ctgRuns(const class RunSet* runSet,
	       const class SplitNux& cand);

  
  /**
     @brief Builds runs without checking for implicit observations.
   */
  void runsExplicit(const class SplitNux& cand);


  /**
     @brief As above, but also tracks a residual slot
   */
  void runsImplicit(const class SplitNux& cand);


  /**
     @brief Gini-based splitting for categorical response and predictor.

     Nodes are now represented compactly as a collection of runs.
     For each node, subsets of these collections are examined, looking for the
     Gini argmax beginning from the pre-bias.

     Iterates over nontrivial subsets, coded by unsigneds as bit patterns.  By
     convention, the final run is incorporated into RHS of the split, if any.
     Excluding the final run, then, the number of candidate LHS subsets is
     '2^(runNux.size()-1) - 1'.

     @return Gini information gain.
  */
  double ctgGini();

  
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
     @brief As above, but specialized for binary response.

     @return Gini information gain.
   */
  double binaryGini();
};


#endif
