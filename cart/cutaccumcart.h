// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CART_CUTACCUMCART_H
#define CART_CUTACCUMCART_H

/**
   @file cutaccumcart.h

   @brief Accumulator classes for cut-based (numeric) splitting workspaces.

   @author Mark Seligman

 */

#include "cutaccum.h"

#include <vector>


/**
   @brief Auxiliary workspace information specific to regression.
 */
class CutAccumRegCart : public CutAccumReg {

  /**
     @brief Updates with residual and possibly splits.

     Current rank position assumed to be adjacent to dense rank, whence
     the application of the residual immediately to the left.
   */
  void splitResidual();


  /**
     @return true iff accumulated and monotonicity senses agree.
   */
  inline bool senseMonotone() const {
    IndexT sCountR = sumCount.sCount - sCount;
    double sumR = sumCount.sum - sum;
    bool accumNonDecreasing = (sum * sCountR <= sumR * sCount);
    return monoMode > 0 ? accumNonDecreasing : !accumNonDecreasing;
  }
  

public:
  CutAccumRegCart(const class SplitNux& splitCand,
		  const struct SFRegCart* spReg);


  /**
     @brief Static entry for regression splitting.
   */
  static void split(const struct SFRegCart* spReg,
		    class SplitNux& cand);

  
  /**
     @brief Private regresion splitting method.

     @return information gain.
   */
  double splitReg(const struct SFRegCart* spReg,
		  const class SplitNux& cand);
  

  /**
     @brief Splits a range of indices having an implicit blob either between
     the two bounds or immediately adjacent to one of them.

     @param resid summarizes the blob's residual statistics.
   */
  void splitImpl();


  /**
     @brief As above, but checks monotonicity with argmax.
   */
  void splitImplMono();


  /**
     @brief Splits right to left, no residual.
   */
  void splitRL(IndexT idxStart,
	       IndexT idxEnd);


  /**
     @brief As above, but applies monotonicty constraint.
   */
  void splitRLMono(IndexT idxStart,
		   IndexT idxEnd);


  /**
     @brief Splits a range bounded to the right by a residual.
   */
  void residualRL();


  /**
     @brief As above, but applies monotonicity constraint.
   */
  void residualRLMono();
};


/**
   @brief Splitting accumulator for classification.
 */
class CutAccumCtgCart : public CutAccumCtg {

  /**
     @brief Attempts to split at residual; either monotone or non-.

     Current rank position assumed to be adjacent to residual rank,
     whence the application of the residual immediately to the left.
   */
  void splitResidual();


  /**
     @brief Applies residual state and continues splitting left.
   */
  void residualRL();


public:

  CutAccumCtgCart(const class SplitNux& cand,
		  class SFCtgCart* spCtg);


  /**
     @brief Static entry for classification splitting.
   */
  static void split(class SFCtgCart* spCtg,
		    class SplitNux& cand);
  

  /**
     @brief Private classification splitting method.

     @return information gain.
   */
  double splitCtg(const class SFCtgCart* spCtg,
		  const class SplitNux& cand);

  
  /**
     @brief Splitting method for categorical response over an explicit
     block of numerical observation indices.

     @param rightCtg indicates whether a category has been set in an
     initialization or previous invocation.
   */
  void splitRL(IndexT idxStart,
	       IndexT idxEnd);


  /**
     @brief As above, but with implicit dense blob.
   */
  void splitImpl();
};

#endif
