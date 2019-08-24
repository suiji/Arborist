// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CART_SPLITNUX_H
#define CART_SPLITNUX_H

/**
   @file splitnux.h

   @brief Minimal container capable of characterizing split.

   @author Mark Seligman

 */

#include "typeparam.h"

class SplitNux {
  static constexpr double minRatioDefault = 0.0;
  static double minRatio;

  double info; // Weighted variance or Gini, currently.
  unsigned int predIdx;  // Core-order predictor index.
  unsigned char bufIdx;
  IndexType lhSCount;
  IndexType lhExtent;
  IndexType lhImplicit;
  IndexRange idxRange;

  IndexRange rankRange;  // Rank bounds:  numeric only.
  unsigned int setIdx; // Index into runSet vector for factor split.
  unsigned int cardinality; // Cardinality iff factor else zero.

public:
  static void immutables(double minRatio_);
  static void deImmutables();


  /**
     @brief Trivial constructor. 'info' value of 0.0 ensures ignoring.
   */
  SplitNux() :
  info(0.0),
  predIdx(0),
  bufIdx(0),
  lhSCount(0),
  lhExtent(0),
  lhImplicit(0),
  idxRange(IndexRange()),
  rankRange(IndexRange()),
  setIdx(0) {
  }

  /**
     @brief Constructor copies essential candidate components.

     @param argMax is the chosen splitting candidate.
   */
  SplitNux(const class SplitCand& argMax,
           const class SummaryFrame* frame);


  /**
     @brief Reports whether potential split be informative with respect to a threshold.

     @param minInfo is an information threshold.

     @return true iff information content exceeds the threshold.
   */
  bool isInformative(double minInfo) const {
    return info > minInfo;
  }


  /**
     @brief Consumes frontier node parameters associated with nonterminal.

     @param[out] minInfo outputs the information threshold for splitting.

     @param[out] lhSCount outputs the number of samples in LHS.

     @param[out] lhExtent outputs the number of indices in LHS.
  */
  void consume(IndexSet* iSet) const;


  /**
     @return true iff left side has no implicit indices.  Rank-based
     splits only.
   */
  bool leftIsExplicit() const {
    return lhImplicit == 0;
  }


  /**
     @brief Getters:
   */
  auto getInfo() const {
    return info;
  }

  auto getBufIdx() const {
    return bufIdx;
  }

  auto getPredIdx() const {
    return predIdx;
  }

  auto getRankRange() const {
    return rankRange;
  }  

  auto getSetIdx() const {
    return setIdx;
  }

  auto getCardinality() const {
    return cardinality;
  }

  auto getExtent() const {
    return idxRange.getEnd() - idxRange.getStart() - 1;
  }


  auto getLHExtent() const {
    return lhExtent;
  }

  
  /**
     @return Count of indices corresponding to LHS.

     Only applies to rank-based splits.
   */
  auto getLHExplicit() const {
    return lhExtent - lhImplicit;
  }

  /**
     @return Count of indices corresponding to RHS.  Rank-based splits
     only.
   */  
  auto getRHExplicit() const {
    return getExtent() - getLHExplicit();
  }


  /**
     @return Starting index of an explicit branch.  Defaults to left if
     both branches explicit.  Rank-based splits only.
   */
  auto getExplicitBranchStart() const {
    return lhImplicit == 0 ? idxRange.getStart() : idxRange.getStart() + getLHExplicit();
  }


  /**
     @return Extent of an explicit branch.  Defaults to left if both
     branches explicit.  Rank-based splits only.
   */
  auto getExplicitBranchExtent() const {
    return lhImplicit == 0 ? getLHExplicit() : getRHExplicit();
  }

  
  /**
     @return coordinate range of the explicit sample indices.
   */
  auto getExplicitRange() const {
    IndexRange range;
    range.set(getExplicitBranchStart(), getExplicitBranchExtent());
    return range;
  }
};


#endif
