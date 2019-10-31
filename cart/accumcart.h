// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CART_SPLITACCUM_H
#define CART_SPLITACCUM_H

/**
   @file splitaccum.h

   @brief Accumulator classes for cut-based (numeric) splitting workspaces.

   @author Mark Seligman

 */

#include "accum.h"

#include <vector>


/**
   @brief Auxiliary workspace information specific to regression.
 */
class AccumCartReg : public Accum {
  const int monoMode; // Presence/direction of monotone constraint.
  const unique_ptr<struct Residual> resid; // Current residual, if any, else null.

  inline void trialSplit(IndexT idx,
			 IndexT rkThis,
			 IndexT rkRight) {
    if (rkThis != rkRight) {
      Accum::trialRight(infoSplit(sumL, sum - sumL, sCountL, sCount - sCountL), idx, rkThis, rkRight);
    }
  }

  
  /**
     @brief Updates with residual and possibly splits.

     Current rank position assumed to be adjacent to dense rank, whence
     the application of the residual immediately to the left.

     @param rkThis is the rank of the current position.
   */
  void splitResidual(IndexT rkThis);


public:
  AccumCartReg(const class SplitNux* splitCand,
                const class SampleRank spn[],
                const class SFCartReg* spReg);

  ~AccumCartReg();

  
  /**
     @brief Evaluates trial splitting information as weighted variance.

     @param sumLeft is the sum of responses to the left of a trial split.

     @param sumRight is the sum of responses to the right.

     @param sCountLeft is number of samples to the left.

     @param sCountRight is the number of samples to the right.

     @param info[in, out] outputs max of input and new information 
   */
  static constexpr double infoSplit(double sumLeft,
                                    double sumRight,
                                    IndexT sCountLeft,
                                    IndexT sCountRight) {
    return (sumLeft * sumLeft) / sCountLeft + (sumRight * sumRight) / sCountRight;
  }


  static bool infoSplit(double sumLeft,
                        double sumRight,
                        IndexT sCountLeft,
                        IndexT sCountRight,
                        double& info) {
    double infoTemp = infoSplit(sumLeft, sumRight, sCountLeft, sCountRight);
    if (infoTemp > info) {
      info = infoTemp;
      return true;
    }
    else {
      return false;
    }
  }


  /**
     @brief Dispatches appropriate splitting method.
   */
  void split(const class SFCartReg* spReg,
             const class SampleRank spn[],
             class SplitNux* cand);
  

  /**
     @brief Splits a range of indices having an implicit blob either between
     the two bounds or immediately adjacent to one of them.

     @param resid summarizes the blob's residual statistics.
   */
  void splitImpl(const class SampleRank spn[],
                 const class SplitNux* cand);


  /**
     @brief Low-level splitting method for explicit block of indices.
   */
  void splitExpl(const SampleRank spn[],
                 IndexT rkThis,
                 IndexT idxInit,
                 IndexT idxFinal);

  /**
     @brief As above, but specialized for monotonicty constraint.
   */
  void splitMono(const SampleRank spn[],
                 IndexT rkThis,
                 IndexT idxInit,
                 IndexT idxFinal);
};


/**
   @brief Splitting accumulator for classification.
 */
class AccumCartCtg : public Accum {
  const PredictorT nCtg; // Cadinality of response.
  const unique_ptr<struct ResidualCtg> resid;
  const vector<double>& ctgSum; // Per-category response sum at node.
  double* ctgAccum; // Slice of compressed accumulation data structure.
  double ssL; // Left sum-of-squares accumulator.
  double ssR; // Right " ".


  /**
     @brief Updates a split if not tied and information increases.
   */
  inline void trialSplit(IndexT idx,
                         IndexT rkThis,
                         IndexT rkRight) {
    if (rkThis != rkRight) {
      Accum::trialRight(infoSplit(ssL, ssR, sumL, sum - sumL), idx, rkThis, rkRight);
    }
  }

  
  /**
     @brief Applies residual state and continues splitting left.
   */
  void residualAndLeft(const class SampleRank spn[],
		       IndexT idxLeft,
		       IndexT idxStart);

  
  /**
     @brief Imputes per-category dense rank statistics as residuals over cell.

     @param cand is the splitting candidate.

     @param spn is the splitting environment.

     @param spCtg summarizes the categorical response.

     @return new residual for categorical response over cell.
  */
  unique_ptr<struct ResidualCtg>
  makeResidual(const class SplitNux* cand,
               const class SampleRank spn[],
               const class SFCartCtg* spCtg);

public:

  AccumCartCtg(const class SplitNux* cand,
                const class SampleRank spn[],
                class SFCartCtg* spCtg);

  ~AccumCartCtg();

  
  /**
     @brief Evaluates trial splitting information as Gini.

     @param ssLeft is the sum of squared responses to the left of a trial split.

     @param ssRight is the sum of squared responses to the right.

     @param sumLeft is the sum of responses to the left.

     @param sumRight is the sum of responses to the right.
   */
  static constexpr double infoSplit(double ssLeft,
                                    double ssRight,
                                    double sumLeft,
                                    double sumRight) {
    return ssLeft / sumLeft + ssRight / sumRight;
  }


  static bool infoSplit(double ssLeft,
                        double ssRight,
                        double sumLeft,
                        double sumRight,
                        double& info) {
    double infoTemp = infoSplit(ssLeft, ssRight, sumLeft, sumRight);
    if (infoTemp > info) {
      info = infoTemp;
      return true;
    }
    else {
      return false;
    }
  }


  /**
     @brief Dispatches appropriate splitting method.
   */
  void split(const class SFCartCtg* spCtg,
             const class SampleRank spn[],
             class SplitNux* cand);

  
  /**
     @brief Splitting method for categorical response over an explicit
     block of numerical observation indices.

     @param rightCtg indicates whether a category has been set in an
     initialization or previous invocation.
   */
  void splitExpl(const class SampleRank spn[],
	    IndexT rkThs,
	    IndexT idxInit,
	    IndexT idxFinal);

  /**
     @brief As above, but with implicit dense blob.
   */
  void splitImpl(const class SampleRank spn[],
                 const class SplitNux* cand);

  /**
     @brief Accumulates right and left sums-of-squares from
     exposed state.
   */
  inline void stateNext(const class SampleRank spn[],
			IndexT idx);


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
  double accumCtgSum(PredictorT yCtg,
                     double sumCtg) {
    double val = ctgAccum[yCtg];
    ctgAccum[yCtg] += sumCtg;
    return val;
  }


  /**
     @brief Accumulates running sums of squares.

     @param ctgSum is the response sum for a category.

     @param yCtt is the response category.

     @param[out] ssL accumulates sums of squares from the left.

     @param[out] ssR accumulates sums of squares to the right.
   */
  void accumCtgSS(double ctgSum,
                  PredictorT yCtg,
                  double& ssL_,
                  double& ssR_) {
    double sumRCtg = accumCtgSum(yCtg, ySum);
    ssR += ctgSum * (ctgSum + 2.0 * sumRCtg);
    double sumLCtg = getCtgSum(yCtg) - sumRCtg;
    ssL += ctgSum * (ctgSum - 2.0 * sumLCtg);
  }

};

#endif
