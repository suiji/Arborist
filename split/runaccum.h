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

#include "splitcoord.h"
#include "sumcount.h"
#include "accum.h"
#include "bheap.h"
#include "runsig.h"

#include <vector>

/**
   @brief Conveys results of a splitting operation.
 */
struct SplitRun {
  double gain; ///< Information gain of split.
  PredictorT token; ///< Cut or bit representation.
  PredictorT runsSampled; ///< # run participating in split.

  SplitRun(double gain_,
	   PredictorT token_,
	   PredictorT runsSampled_) :
    gain(gain_),
    token(token_),
    runsSampled(runsSampled_) {
  }
};


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

  vector<BHPair<PredictorT>> heapZero; ///< Sorting workspace.


  /**
     @brief Builds runs for regression.
   */
  vector<RunNux> regRuns(const SplitNux& cand);


  vector<RunNux> initRuns(class RunSet* runSet,
			  const class SplitNux& cand);


  vector<RunNux> regRunsExplicit(const SplitNux& cand);

  /**
     @brief As above, but also tracks a residual slot.
   */
  vector<RunNux> regRunsImplicit(const SplitNux& cand);


  /**
     @brief Determines split having highest weighted variance.

     Runs initially sorted by mean response.

     @return gain in weighted variance.
   */
  SplitRun maxVar(const vector<RunNux>& runNux);

  
  /**
     @brief Sorts by mean response.
   */
  void heapMean(const vector<RunNux>& runNux);

  
public:
  static constexpr unsigned int maxWidth = 10; // Algorithmic threshold.


  /**
   */
  RunAccum(const class SplitFrontier* splitFrontier,
	   const class SplitNux& cand);


  /**
     @brief Determines whether run count must be truncated.

     @return true iff run count exceeds maximum.
   */
  static bool ctgWide(const class SplitFrontier* sf,
		      const class SplitNux& cand);


  /**
     @brief Depopulates the heap associated with a pair and places sorted ranks into rank vector.
  */
  vector<RunNux> slotReorder(const vector<RunNux>& runNux);


  void initReg(IndexT runLeft,
	       RunNux& nux) const;

  
  /**
     @brief As above, but skips masked SR indices.

     @param maskSense indicates whether to screen set or unset mask.
   */  
  vector<RunNux> regRunsMasked(const SplitNux& cand,
			       const class BranchSense* branchSense,
			       bool maskSense);


  /**
     @brief Heap orders by target-mean encoding.
  */
  vector<RunNux> orderMean(const vector<RunNux>& runNux);
};


class RunAccumReg : public RunAccum {
public:
  RunAccumReg(const struct SFReg* sfReg,
	      const class SplitNux& cand);


  /**
     @breif Static entry for regression splitting.
   */
  static void split(const struct SFReg* sfReg,
		    class RunSet* runSet,
		    class SplitNux& cand);

  /**
     @brief Private splitting entry.
   */
  SplitRun split(const vector<RunNux>& runNux);
};



class RunAccumCtg : public RunAccum {
  const PredictorT nCtg; ///< Response category count.
  const bool sampling; ///< Whether to split sample.
  const PredictorT sampleCount; ///< # runs to sample.
  
  CtgNux ctgNux;

  // Initialized as a side-effect of RunNux construction:
  vector<double> runSum; ///<  run x ctg checkerboard.


  vector<RunNux> sampleRuns(const class RunSet* runSet,
			    const class SplitNux& cand,
			    const vector<RunNux>& runNux);


  vector<RunNux> initRuns(class RunSet* runSet,
			  const class SplitNux& cand);


public:

  RunAccumCtg(const class SFCtg* sfCtg,
	      const class SplitNux& cand);


  /**
     @return checkerboard value at slot for category.
   */
  double getRunSum(PredictorT runIdx,
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
  bool accumBinary(const vector<RunNux>& runNux,
			  PredictorT slot,
			  double& sum0,
			  double& sum1) {
    sum0 += getRunSum(slot, 0);
    double cell1 = getRunSum(slot, 1);
    sum1 += cell1;

    // Two runs are deemed significantly different if their sample
    // counts differ. If identical, then checks whether the response
    // sums differ by some measure.
    PredictorT slotNext = slot+1;
    return (runNux[slot].sumCount.sCount != runNux[slotNext].sumCount.sCount) ||  getRunSum(slotNext, 1) > cell1;
  }


  /**
     @brief Writes to heap, weighting by category-1 probability.
  */
  vector<RunNux> orderBinary(const vector<RunNux>& runNux);


  /**
     @brief Sorts by probability, binary response.
   */
  void heapBinary(const vector<RunNux>& runNux);


  /**
     @brief Static entry for classification splitting.
   */
  static void split(const class SFCtg* sf,
		    class RunSet* runSet,
		    class SplitNux& cand);
  

  double* initCtg(IndexT runLeft,
		  RunNux& nux,
		  PredictorT runIdx);


  /**
     @brief Subtracts a run's per-category responses from the current run.
   */
  void residualSums(const vector<RunNux>& runNux,
		       PredictorT implicitSlot);


  /**
     @brief Private entry for categorical splitting.
   */
  SplitRun split(const vector<RunNux>& runNux);


  /**
     @brief Accumulates runs for classification.

     @param sumSlice is the per-category response decomposition.
  */
  vector<RunNux> ctgRuns(class RunSet* runSet,
			 const class SplitNux& cand);

  
  /**
     @brief Builds runs without checking for implicit observations.
   */
  vector<RunNux> runsExplicit(const class SplitNux& cand);


  /**
     @brief As above, but also tracks a residual slot
   */
  vector<RunNux> runsImplicit(const class SplitNux& cand);


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
  SplitRun ctgGini(const vector<RunNux>& runNux);


  /**
     @brief Determines Gini of a subset of runs encoded as bits.

     @param sumSlice decomposes the partition node response by category.

     @param subset bit-encodes a collection of runs.

     N.B.:  Gini value should be symmetric w.r.t. fixed-size complements.
     
     N.B.:  trivial subsets, beside being uninformative, may precipitate
     division by zero.

     @return Gini coefficient of subset.
   */
  double subsetGini(const vector<RunNux>& runNux,
		    unsigned int subset) const;


  /**
     @brief As above, but specialized for binary response.

     @return Gini information gain.
   */
  SplitRun binaryGini(const vector<RunNux>& runNux);
};


#endif
