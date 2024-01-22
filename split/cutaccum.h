/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_CUTACCUM_H
#define SPLIT_CUTACCUM_H

/**
   @file cutaccum.h

   @brief Base accumulator classes for cut-based (numeric) splitting workspaces.

   @author Mark Seligman

 */

#include "typeparam.h"
#include "accum.h"
#include "obs.h"

#include <vector>

/**
   @brief Persistent workspace for computing optimal split.

   Cells having implicit dense blobs are split in separate sections,
   calling for a re-entrant data structure to cache intermediate state.
   Accum is tailored for right-to-left index traversal.
 */
class CutAccum : public Accum {
protected:

  /**
     @brief As above, but directionless.
   */
  void argmaxBounds(double infoTrial,
		    IndexT obsRight,
		    IndexT obsLeft) {
    if (Accum::trialSplit(infoTrial)) {
      this->obsRight = obsRight;
      this->obsLeft = obsLeft;
    }
  }


  /**
     @brief Accumulates sum and sample-count state from observation.

     @return true iff rank is tied with that of left neighbor.
   */
  bool accumulateReg(const Obs& obs) {
    sum -= obs.getYSum();
    sCount -= obs.getSCount();
    return obs.isTied();
  }


  /**
     @brief Derives and applies residual contributions.
   */
  void applyResidual(const Obs* obsCell);


  /**
     @brief Revises argmax in right-to-left traversal.
   */
  void trialObsRL(double infoTrial,
		  IndexT obsLeft,
		  IndexT obsRight);


public:
  // Revised at each new local maximum of 'info':
  IndexT obsLeft; ///< sup left index.  Out of bounds (obsEnd + 1) iff left is dense.
  IndexT obsRight; ///< inf right index.  Out of bounds (obsEnd + 1) iff right is dense.
  bool residualLeft; ///< State of most recent residual argmax:  L/R.

  /**
     @param cand encapsulates candidate splitting parameters.

     @param splitFrontier looks up residual rank.
   */
  CutAccum(const class SplitNux& cand,
	   const class SplitFrontier* splitFrontier);

  
  IndexT lhImplicit(const class SplitNux& cand) const;


  /**
     @brief Derives splitting rank from cut bounds.

     @return fractional splitting rank.
   */
  double interpolateRank(const class InterLevel* interLevel,
			 const class SplitNux& cand) const;


  /**
     @brief Determines whether an argmax has been encountered since
     initialization.

     @return true iff argmax has been observed.
   */
  bool hasArgmax() const {
    return obsLeft != obsRight;
  }
};


class CutAccumCtg : public CutAccum {

protected:
  const CtgNux ctgNux; ///< Categorical sums, missing data filtered.
  vector<double> ctgAccum; ///< Accumulates per-category response.
  double ssL; ///< Left sum-of-squares accumulator.
  double ssR; ///< Right " ".


  /**
     @brief Trial argmax on decreasing index.

     @param obsLeft is the left bound.

     @param ungated is true iff caller not preempting computation.
     
     In CART-like splitting, right bound is implicitly one greater.
   */
  void argmaxRL(double infoTrial,
		IndexT obsLeft) {
    if (Accum::trialSplit(infoTrial)) {
      this->obsLeft = obsLeft;
      obsRight = obsLeft + 1;
    }
  }




  /**
     @brief Trial argmax involving residual.

     @param infoTrial is the information content of the trial.

     @param onLeft is true iff residual is the left observation.

     @param ungated is true iff caller no preempting computation.

     May be called twice for the same residual:  once right, once left.
   */
  void argmaxResidual(double infoTrial,
		      bool onLeft) {
    if (Accum::trialSplit(infoTrial)) {
      obsRight = cutResidual;
      // cutResidual > obsStart if residual lies to the right.
      obsLeft = (cutResidual == obsStart ? cutResidual : cutResidual - 1);
      residualLeft = onLeft;
    }
  }


  /**
     @brief Accumulates observation state.

     @return true iff rank ties with observation to left.
   */
  bool accumulateCtg(const Obs& obs) {
    sum -= obs.getYSum();
    sCount -= obs.getSCount();
    accumCtgSS(obs.getYSum(), obs.getCtg());

    return obs.isTied();
  }


  /**
     @brief Derives and applies residual contributions.
   */
  void applyResidual(const Obs* obsCell);

  
public:
  CutAccumCtg(const class SplitNux& cand,
	      class SFCtg* sfCtg);


  /**
     @brief Updtes category sum and squared sums.

     @param ySumCtg is the response sum for a category.

     @param yCtg is the response category.
   */
  void accumCtgSS(double ySumCtg,
		  PredictorT yCtg) {
    double ySum2 = ySumCtg * ySumCtg;
    ssR += ySum2 + 2.0 * ySumCtg * ctgAccum[yCtg];
    ssL += ySum2 - 2.0 * ySumCtg * (ctgNux.ctgSum[yCtg] - ctgAccum[yCtg]);
    ctgAccum[yCtg] += ySumCtg;
  }
};


class CutAccumReg : public CutAccum {

protected:
  const int monoMode; ///< Presence/direction of monotone constraint.

  /**
     @return false iff monotone and sense violated.
   */
  bool senseMonotone() const {
    if (monoMode == 0)
      return true;

    IndexT sCountR = sumCount.sCount - sCount;
    double sumR = sumCount.sum - sum;
    bool accumNonDecreasing = (sum * sCountR <= sumR * sCount);
    return monoMode > 0 ? accumNonDecreasing : !accumNonDecreasing;
  }

    /**
     @brief Trial argmax on decreasing index.

     @param obsLeft is the left bound.

     @param ungated is true iff caller not preempting computation.
     
     In CART-like splitting, right bound is implicitly one greater.
   */
  void argmaxRL(double infoTrial,
		IndexT obsLeft) {
    if (senseMonotone() && Accum::trialSplit(infoTrial)) {
      this->obsLeft = obsLeft;
      obsRight = obsLeft + 1;
    }
  }


  /**
     @brief Trial argmax involving residual.

     @param infoTrial is the information content of the trial.

     @param onLeft is true iff residual is the left observation.

     @param ungated is true iff caller no preempting computation.

     May be called twice for the same residual:  once right, once left.
   */
  void argmaxResidual(double infoTrial,
		      bool onLeft) {
    if (senseMonotone() && Accum::trialSplit(infoTrial)) {
      obsRight = cutResidual;
      // cutResidual > obsStart if residual lies to the right.
      obsLeft = (cutResidual == obsStart ? cutResidual : cutResidual - 1);
      residualLeft = onLeft;
    }
  }



public:
  CutAccumReg(const class SplitNux& splitCand,
	      const struct SFReg* spReg);
};


#endif

