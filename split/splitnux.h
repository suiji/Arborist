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

#include "splitcoord.h"
#include "typeparam.h"

struct SplitNux {
  SplitCoord splitCoord;
  IndexRange idxRange; // Indices into compressed ObsPart buffer.
  PredictorT setIdx; // Index into runSet vector for factor split.
  unsigned char bufIdx;

  double info; // Weighted variance or Gini, currently.

  // Accumulated during splitting:
  IndexT lhSCount; // # samples subsumed by split LHS:  > 0 iff split.
  IndexT lhExtent; // # " " indices " ".
  IndexT lhImplicit; // LHS implicit index count:  numeric only.

  // Copied to decision node, if arg-max.  Numeric only:
  //
  IndexRange rankRange;  // Rank bounds.
  
  static void immutables(double minRatio_);
  static void deImmutables();


  /**
     @brief Trivial constructor. 'info' value of 0.0 ensures ignoring.
  */
  SplitNux() : info(0.0) {
  }

  
  /**
     @brief Called by SplitCand constructor.
   */
  SplitNux(SplitCoord splitCoord_,
	   PredictorT setIdx_,
	   unsigned char bufIdx_,
	   double info_) :
  splitCoord(splitCoord_),
    setIdx(setIdx_),
    bufIdx(bufIdx_),
    info(info_) {
  }

  
  ~SplitNux() {
  }

  /**
     @brief Passes through to frame method.

     @return cardinality iff factor-valued predictor else zero.
   */
  PredictorT getCardinality(const class SummaryFrame*) const;


  /**
     @brief Decrements information field and reports whether still positive.

     @param splitFrontier determines pre-existing information value to subtract.

     @bool true iff decremented information field positive.
   */
  bool infoGain(const class SplitFrontier* splitFrontier);
  

  /**
     @brief Writes the left-hand characterization of a factor-based
     split with categorical response.

     @param lhBits is a compressed representation of factor codes for the LHS.
   */
  void writeBits(const class SplitFrontier* splitFrontier,
		 PredictorT lhBits);
  

  void writeNum(const SplitFrontier* splitFrontier,
		double info,
		IndexT rankLH,
		IndexT rankRH,
		IndexT lhScount,
		IndexT lhImplicit,
		IndexT rhMin);

  /**
     @brief Consumes frontier node parameters associated with nonterminal.
  */
  void consume(class IndexSet* iSet) const;


  /**
     @brief Reports whether potential split be informative with respect to a threshold.

     @param minInfo is an information threshold.

     @return true iff information content exceeds the threshold.
   */
  bool isInformative(double minInfo) const {
    return info > minInfo;
  }


  /**
     @return true iff left side has no implicit indices.  Rank-based
     splits only.
   */
  bool leftIsExplicit() const {
    return lhImplicit == 0;
  }


  auto getExtent() const {
    return idxRange.getExtent();
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

private:
  static constexpr double minRatioDefault = 0.0;
  static double minRatio;
};


#endif
