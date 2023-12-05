/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_ACCUM_H
#define SPLIT_ACCUM_H

/**
   @file accum.h

   @brief Generic accumulator class for computing splits.

   @author Mark Seligman
 */

#include "typeparam.h"
#include "sumcount.h"


/**
   @brief Accumulated values for categorical nodes.
 */
struct CtgNux {
  vector<double> ctgSum; ///> # per-category response sum
  double sumSquares; ///> Sum of squares over categories.

  CtgNux(vector<double>& ctgSum_,
	 double sumSquares_) :
  ctgSum(ctgSum_),
    sumSquares(sumSquares_) {
  }

  
  PredictorT nCtg() const {
    return ctgSum.size();
  }
};


struct Accum {
private:
  SumCount filterMissing(const class SplitNux& cand) const;
  
protected:
  // Information is initialized according to the splitting method.
  double info; ///< Information high watermark.

  
  CtgNux filterMissingCtg(const class SFCtg* sfCtg,
			  const SplitNux& cand) const;

public:
  const class Obs* obsCell;
  const IndexT* sampleIndex;
  const IndexT obsStart;///< Low terminus.
  const IndexT obsEnd; ///< sup.
  const SumCount sumCount; ///< Initialized from candidate, filtered.
  const IndexT cutResidual; ///< Rightmost position > any residual.
  const IndexT implicitCand;

  double sum; ///< Running sum of trial LHS response.
  IndexT sCount; ///< Running sum of trial LHS sample counts.

  Accum(const class SplitFrontier* splitFrontier,
	const class SplitNux& cand);

  /**
     @brief Computes weighted-variance for trial split.

     @param sumLeft is the sum of responses to the left of a trial split.

     @param sumRight is the sum of responses to the right.

     @param sCountLeft is number of samples to the left.

     @param sCountRight is the number of samples to the right.

     @param return weighted-variance information value.
   */
  static double infoVar(double sumLeft,
			       double sumRight,
			       IndexT sCountLeft,
			       IndexT sCountRight) {
    return (sumLeft * sumLeft) / sCountLeft + (sumRight * sumRight) / sCountRight;
  }


  /**
     @brief As above, but with running and initialized SumCounts.
   */
  static double infoVar(const SumCount& scAccum,
			       const SumCount& scInit) {
    return infoVar(scAccum.sum, scInit.sum - scAccum.sum, scAccum.sCount, scInit.sCount - scAccum.sCount);
  }


  /**
     @brief Evaluates trial splitting information as Gini.

     @param ssLeft is the sum of squared responses to the left of a trial split.

     @param ssRight is the sum of squared responses to the right.

     @param sumLeft is the sum of responses to the left.

     @param sumRight is the sum of responses to the right.
   */
  static double infoGini(double ssLeft,
				double ssRight,
				double sumLeft,
				double sumRight) {
    return ssLeft / sumLeft + ssRight / sumRight;
  }


  /**
     @brief Maintains maximum 'info' value.

     @return true iff value passed exceeds current information value.
  */
  bool trialSplit(double infoTrial) {
    if (infoTrial > info) {
      info = infoTrial;
      return true;
    }
    else {
      return false;
    }
  }


  /**
     @brief Finds outermost range matching given branch sense.
   */
  IndexRange findUnmaskedRange(const class BranchSense* branchSense,
			       bool sense) const;


  /**
     @brief Walks observations to match given branch sense.

     @param branchSense encodes branch sense at each index.

     @param leftward specifies whether walk is right-to-left.
     
     @param idxTerm is the terminus index from which to start.

     @param sense is the branch sense value to match.

     @param[out] edge is the first index matching sense, if any.

     @return whether a match was found.
   */
  bool findEdge(const class BranchSense* branchSense,
		bool leftward,
		bool sense,
		IndexT& edge) const;
};

#endif
