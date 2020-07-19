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
  IndexT cutDense; // Rightmost position beyond implicit blob, if any.
  
  // Read locally but initialized, and possibly reset, externally.
  IndexT sCountThis; // Current sample count.
  FltVal ySumThis; // Current response value.


  /**
     @brief Updates split anywhere left of a residual, if any.
   */
  inline void trialRight(double infoTrial,
			 IndexT idxLeft,
			 IndexT rkThis,
			 IndexT rkRight) {
    if (infoTrial > info) {
      info = infoTrial;
      lhSCount = sCount;
      lhSum = sum;
      rankRH = rkRight;
      rankLH = rkThis;
      this->idxLeft = idxLeft;
      idxRight = rkRight == rankDense ? cutDense : idxLeft + 1;
    }
  }


  /**
     @brief As above, but with distinct index bounds.
   */
  void trialSplit(double infoTrial,
		  IndexT idxLeft,
		  IndexT idxRight);

  
  /**
     @brief Updates split just to the right of a residual.
   */
  inline void splitResidual(double infoTrial,
			   IndexT rkRight) {
    if (infoTrial > info) {
      info = infoTrial;
      lhSCount = sCount;
      lhSum = sum;
      rankRH = rkRight;
      rankLH = rankDense;
      idxRight = cutDense;
    }
  }
  
public:
  // Revised at each new local maximum of 'info':
  IndexT lhSCount; // Sample count of split LHS:  > 0.
  double lhSum; // Sum of responses over LHS.
  IndexT rankRH; // Maximum rank characterizing split.
  IndexT rankLH; // Minimum rank charactersizing split.
  IndexT idxLeft; // sup left index.  Out of bounds (idxEnd + 1) iff left is dense.
  IndexT idxRight; // inf right index.  Out of bounds (idxEnd + 1) iff right is dense.

  /**
     @param cand encapsulates candidate splitting parameters.

     @param splitFrontier looks up dense rank.
   */
  CutAccum(const class SplitNux* cand,
	   const class SplitFrontier* splitFrontier);

  ~CutAccum() {
  }

  
  IndexT lhImplicit(const class SplitNux* cand) const;


  double interpolateRank(const class SplitNux* cand) const;
};


class CutAccumCtg : public CutAccum {
protected:

  const PredictorT nCtg; // Cadinality of response.
  const unique_ptr<struct ResidualCtg> resid;
  const vector<double>& ctgSum; // Per-category response sum at node.
  double* ctgAccum; // Slice of compressed accumulation data structure.
  double ssL; // Left sum-of-squares accumulator.
  double ssR; // Right " ".


  /**
     @brief Accessor for node-wide sum for a given category.

     @param ctg is the category in question

     @return sum at category over node.
   */
  inline double getCtgSum(PredictorT ctg) const {
    return ctgSum[ctg];
  }


  /**
     @brief Post-increments accumulated sum.

     @param yCtg is the category at which to increment.

     @param sumCtg is the sum by which to increment.
     
     @return value of accumulated sum prior to incrementing.
   */
  inline double accumCtgSum(PredictorT yCtg,
                     double sumCtg) {
    double val = ctgAccum[yCtg];
    ctgAccum[yCtg] += sumCtg;
    return val;
  }


  /**
     @brief Imputes per-category dense rank statistics as residuals over cell.

     @param cand is the splitting candidate.

     @param spn is the splitting environment.

     @param spCtg summarizes the categorical response.

     @return new residual for categorical response over cell.
  */
  unique_ptr<struct ResidualCtg> makeResidual(const class SplitNux* cand,
					      const class SFCtg* sfCtg);



public:
  CutAccumCtg(const class SplitNux* cand,
	      class SFCtg* sfCtg);


  /**
     @brief Accumulates running sums of squares.

     @param ctgSum is the response sum for a category.

     @param yCtt is the response category.

     @param[in, out] ssL accumulates sums of squares from the left.

     @param[in, out] ssR accumulates sums of squares to the right.
   */
  inline void accumCtgSS(double ctgSum,
                  PredictorT yCtg,
                  double& ssL_,
                  double& ssR_) {
    double sumRCtg = accumCtgSum(yCtg, ySumThis);
    ssR += ctgSum * (ctgSum + 2.0 * sumRCtg);
    double sumLCtg = getCtgSum(yCtg) - sumRCtg;
    ssL += ctgSum * (ctgSum - 2.0 * sumLCtg);
  }
};


class CutAccumReg : public CutAccum {

protected:
  const int monoMode; // Presence/direction of monotone constraint.
  const unique_ptr<struct Residual> resid; // Current residual or null.

public:
  CutAccumReg(const class SplitNux* splitCand,
	      const class SFReg* spReg);

  ~CutAccumReg();
  /**
     @brief Creates a residual summarizing implicit splitting state.

     @param cand is the splitting candidate.

     @param spn is the splitting data set.
     
     @return new residual based on the current splitting data set.
   */
  unique_ptr<struct Residual> makeResidual(const class SplitNux* cand,
                                          const class SampleRank spn[]);

};


/**
   @brief Minimal information needed to reconstruct cut.
 */
struct CutSig {
  // In CART-like implementations, idxLeft and idxRight are adjacent.
  IndexT idxLeft; // sup of left SampleRank indices.
  IndexT idxRight;  // inf of right SampleRank indices.
  IndexT implicitTrue; // # implicit SampleRank indices associated with true sense.
  double quantRank; // Interpolated cut rank.
  bool cutLeft; // True iff cut encodes left portion.

  CutSig(const IndexRange& idxRange) :
    idxLeft(idxRange.getStart()),
    idxRight(idxRange.getEnd() - 1),
    cutLeft(true) { // Default.
  }

  CutSig() :
    cutLeft(true) {
  }

  void write(const class SplitNux* nux,
	     const class CutAccum* accum);
};

#endif

