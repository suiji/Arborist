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

#include <vector>


/**
   @brief Persistent workspace for computing optimal split.

   Cells having implicit dense blobs are split in separate sections,
   calling for a re-entrant data structure to cache intermediate state.
   Accum is tailored for right-to-left index traversal.
 */
class CutAccum : public Accum {
protected:
  IndexT cutResidual; ///< Rightmost position beyond residual, if any.

  /**
     @brief Trial argmax on right indices.
   */
  inline void argmaxRL(double infoTrial,
		       IndexT obsLeft,
		       IndexT rkIdxR) {
    if (Accum::trialSplit(infoTrial)) {
      rankIdxR = rkIdxR;
      rankIdxL = rkIdxR + 1; // CART-like, explicit.
      this->obsLeft = obsLeft;
      obsRight = obsLeft + 1;
    }
  }


  // Diagnostic.
  inline void argmaxRL(double infoTrial,
			   IndexT obsLeft,
			   IndexT rkIdxR,
		       IndexT rkIdxL) {
    if (Accum::trialSplit(infoTrial)) {
      rankIdxR = rkIdxR;
      rankIdxL = rkIdxL;
      this->obsLeft = obsLeft;
      obsRight = obsLeft + 1;
    }
  }


  /**
     @brief Revises argmax in right-to-left traversal.
   */
  void trialObsRL(double infoTrial,
		  IndexT obsLeft,
		  IndexT obsRight);


public:
  // Revised at each new local maximum of 'info':
  IndexT rankIdxL; ///< Left rank index.
  IndexT rankIdxR; ///< Right rank index.
  IndexT obsLeft; ///< sup left index.  Out of bounds (obsEnd + 1) iff left is dense.
  IndexT obsRight; ///< inf right index.  Out of bounds (obsEnd + 1) iff right is dense.

  /**
     @param cand encapsulates candidate splitting parameters.

     @param splitFrontier looks up residual rank.
   */
  CutAccum(const class SplitNux* cand,
	   const class SplitFrontier* splitFrontier);

  
  IndexT lhImplicit(const class SplitNux* cand) const;


  double interpolateRank(const class ObsFrontier* ofFront,
			 const class SplitNux* cand) const;

  /**
     @brief Determines whether an argmax has been encountered since
     initialization.

     @return true iff argmax has been observed.
   */
  bool hasArgmax() const {
    return rankIdxL != rankIdxR;
  }
};


class CutAccumCtg : public CutAccum {
protected:

  const PredictorT nCtg; ///< Cadinality of response.
  const vector<double>& nodeSum; ///< Per-category response sum at node.
  double* ctgAccum; ///< Slice of compressed accumulation data structure.
  double ssL; ///< Left sum-of-squares accumulator.
  double ssR; ///< Right " ".


  /**
     @brief Post-increments accumulated sum.

     @param yCtg is the category at which to increment.

     @param sumCtg is the sum by which to increment.
     
     @return value of accumulated sum prior to incrementing.
   */
  inline double accumCtgSum(PredictorT yCtg,
			    double sumCtg) {
    return exchange(ctgAccum[yCtg], ctgAccum[yCtg] + sumCtg);
  }


public:
  CutAccumCtg(const class SplitNux* cand,
	      class SFCtg* sfCtg);


  /**
     @brief Accumulates running sums of squares by category.

     @param ctgSum is the response sum for a category.

     @param yCtt is the response category.
   */
  inline void accumCtgSS(double ctgSum,
			 PredictorT yCtg) {
    double sumRCtg = exchange(ctgAccum[yCtg], ctgAccum[yCtg] + ctgSum);
    ssR += ctgSum * (ctgSum + 2.0 * sumRCtg);
    double sumLCtg = nodeSum[yCtg] - sumRCtg;
    ssL += ctgSum * (ctgSum - 2.0 * sumLCtg);
  }
};


class CutAccumReg : public CutAccum {

protected:
  const int monoMode; ///< Presence/direction of monotone constraint.

public:
  CutAccumReg(const class SplitNux* splitCand,
	      const class SFReg* spReg);
};


#endif

