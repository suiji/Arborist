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
class CutAccumRegCart : public CutAccumReg {
  /**
     @brief Updates with residual and possibly splits.

     Current rank position assumed to be adjacent to dense rank, whence
     the application of the residual immediately to the left.

     @param rkThis is the rank of the current position.
   */
  void splitResidual(IndexT rkThis);


public:
  CutAccumRegCart(const class SplitNux* splitCand,
                const class SFRegCart* spReg);


  /**
     @brief Static entry for regression splitting.
   */
  static void split(const class SFRegCart* spReg,
		    class SplitNux* cand);

  
  /**
     @brief Private regresion splitting method.
   */
  void splitReg(const class SFRegCart* spReg,
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
class CutAccumCtgCart : public CutAccumCtg {
  /**
     @brief Applies residual state and continues splitting left.
   */
  void residualAndLeft(IndexT idxLeft,
		       IndexT idxStart);

  
public:

  CutAccumCtgCart(const class SplitNux* cand,
	      class SFCtgCart* spCtg);


  /**
     @brief Static entry for classification splitting.
   */
  static void split(class SFCtgCart* spCtg,
		    class SplitNux* cand);
  

  /**
     @brief Private classification splitting method.
   */
  void splitCtg(const class SFCtgCart* spCtg,
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

};

#endif
