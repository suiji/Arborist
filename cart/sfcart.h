// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CART_SFCART_H
#define CART_SFCART_H

/**
   @file sfcart.h

   @brief Manages CART-specific node splitting across the tree frontier.

   @author Mark Seligman

 */

#include "splitcoord.h"
#include "typeparam.h"
#include "sumcount.h"
#include "splitfrontier.h"

#include <vector>


/**
   @brief Splitting facilities specific regression trees.
 */
struct SFRegCart : public SFReg {
  SFRegCart(class Frontier* frontier_);

  ~SFRegCart() = default;


  /**
     @return enumeration indicating slot-style encoding.
   */
  SplitStyle getFactorStyle() const;


  void split(const CandType& cand,
	     class BranchSense& branchSense);


  /**
     @brief Collects splitable candidates from among all restaged cells.
   */
  void split(class SplitNux& cand);
};


/**
   @brief Splitting facilities for categorical trees.
 */
class SFCtgCart : public SFCtg {
// Numerical tolerances taken from A. Liaw's code:
  static constexpr double minDenom = 1.0e-5;
  static constexpr double minSumL = 1.0e-8;
  static constexpr double minSumR = 1.0e-5;

  /**
     @return slot-style for binary response, otherwise bit-style.
   */
  SplitStyle getFactorStyle() const;


  void split(const CandType& cand,
	     class BranchSense& branchSense);


  /**
     @brief Collects splitable candidates from among all restaged cells.
   */
  void split(class SplitNux& cand);


public:
  SFCtgCart(class Frontier* frontier_);

  ~SFCtgCart() = default;


  /**
     @brief Determine whether an ordered pair of sums is acceptably stable
     to appear in the denominator.

     Only relevant for instances of extreme case weighting.  Currently unused
     and may be obsolete.

     @param sumL is the left-hand sum.

     @param sumR is the right-hand sum.

     @return true iff both sums suitably stable.
   */
  inline bool stableSum(double sumL, double sumR) const {
    return sumL > minSumL && sumR > minSumR;
  }


  /**
     @brief Determines whether a pair of sums is acceptably stable to appear
     in the denominators.

     Only relevant for instances of extreme case weighting.  Currently unused
     and may not be useful if training responses are normalized.

     @param sumL is the left-hand sum.

     @param sumR is the right-hand sum.

     @return true iff both sums suitably stable.
   */
  inline bool stableDenom(double sumL, double sumR) const {
    return sumL > minDenom && sumR > minDenom;
  }
};


#endif
