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

struct Accum {
  const class Obs* obsCell;
  const IndexT* sampleIndex;
  const IndexT rankResidual; ///< Rank of dense value, if any.
  const IndexT obsStart;///< Low terminus.
  const IndexT obsTop; ///< High terminus.
  const double sumCand;
  const IndexT sCountCand;
  const IndexT implicitCand;

  IndexT sCount; ///< Running sum of trial LHS sample counts.
  double sum; ///< Running sum of trial LHS response.
  double info; ///< Information high watermark.  Precipitates split iff > 0.0 after update.

  Accum(const class SplitFrontier* splitFrontier,
	const class SplitNux* cand);

  /**
     @brief Computes weighted-variance for trial split.

     @param sumLeft is the sum of responses to the left of a trial split.

     @param sumRight is the sum of responses to the right.

     @param sCountLeft is number of samples to the left.

     @param sCountRight is the number of samples to the right.

     @param info[in, out] outputs max of input and new information 
   */
  static constexpr double infoVar(double sumLeft,
				  double sumRight,
				  IndexT sCountLeft,
				  IndexT sCountRight) {
    return (sumLeft * sumLeft) / sCountLeft + (sumRight * sumRight) / sCountRight;
  }


  /**
     @brief Evaluates trial splitting information as Gini.

     @param ssLeft is the sum of squared responses to the left of a trial split.

     @param ssRight is the sum of squared responses to the right.

     @param sumLeft is the sum of responses to the left.

     @param sumRight is the sum of responses to the right.
   */
  static constexpr double infoGini(double ssLeft,
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
     @brief Walks Obs index range in specified direction to match given branch sens.

     @param branchSense encodes branch sense for each SR index.

     @param idxTerm is the terminus index from which to start.

     @param sense is the branch sense value to match.

     @param[out] edge is the first index matching sense, if any, else undefined.

     @return whether a match was found.
   */
  bool findEdge(const class BranchSense* branchSense,
		bool leftward,
		IndexT idxTerm,
		bool sense,
		IndexT& edge) const;
};


#endif
