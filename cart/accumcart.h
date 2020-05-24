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

#include "cutaccum.h"

#include <vector>


/**
   @brief Auxiliary workspace information specific to regression.
 */
class CutAccumReg : public CutAccum {
  const int monoMode; // Presence/direction of monotone constraint.
  const unique_ptr<struct Residual> resid; // Current residual or null.

  /**
     @brief Updates with residual and possibly splits.

     Current rank position assumed to be adjacent to dense rank, whence
     the application of the residual immediately to the left.

     @param rkThis is the rank of the current position.
   */
  void splitResidual(IndexT rkThis);


public:
  CutAccumReg(const class SplitNux* splitCand,
                const class SFRegCart* spReg);

  ~CutAccumReg();

  /**
     @brief Dispatches appropriate splitting method.
   */
  void split(const class SFRegCart* spReg,
             class SplitNux* cand);
  

  /**
     @brief Splits a range of indices having an implicit blob either between
     the two bounds or immediately adjacent to one of them.

     @param resid summarizes the blob's residual statistics.
   */
  void splitImpl(const class SplitNux* cand);


  /**
     @brief Low-level splitting method for explicit block of indices.
   */
  void splitExpl(IndexT rkThis,
                 IndexT idxInit,
                 IndexT idxFinal);

  /**
     @brief As above, but specialized for monotonicty constraint.
   */
  void splitMono(IndexT rkThis,
                 IndexT idxInit,
                 IndexT idxFinal);
};


/**
   @brief Splitting accumulator for classification.
 */
class CutAccumCtg : public CutAccum {
  const PredictorT nCtg; // Cadinality of response.
  const unique_ptr<struct ResidualCtg> resid;
  const vector<double>& ctgSum; // Per-category response sum at node.
  double* ctgAccum; // Slice of compressed accumulation data structure.
  double ssL; // Left sum-of-squares accumulator.
  double ssR; // Right " ".


  /**
     @brief Applies residual state and continues splitting left.
   */
  void residualAndLeft(IndexT idxLeft,
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
               const class SFCtgCart* spCtg);

public:

  CutAccumCtg(const class SplitNux* cand,
	      class SFCtgCart* spCtg);

  ~CutAccumCtg();


  /**
     @brief Dispatches appropriate splitting method.
   */
  void split(const class SFCtgCart* spCtg,
             class SplitNux* cand);

  
  /**
     @brief Splitting method for categorical response over an explicit
     block of numerical observation indices.

     @param rightCtg indicates whether a category has been set in an
     initialization or previous invocation.
   */
  void splitExpl(IndexT rkThs,
		 IndexT idxInit,
		 IndexT idxFinal);

  /**
     @brief As above, but with implicit dense blob.
   */
  void splitImpl(const class SplitNux* cand);

  /**
     @brief Accumulates right and left sums-of-squares from
     exposed state.
   */
  inline void stateNext(IndexT idx);


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
    double sumRCtg = accumCtgSum(yCtg, ySumThis);
    ssR += ctgSum * (ctgSum + 2.0 * sumRCtg);
    double sumLCtg = getCtgSum(yCtg) - sumRCtg;
    ssL += ctgSum * (ctgSum - 2.0 * sumLCtg);
  }

};

#endif
