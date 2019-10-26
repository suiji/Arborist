// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_SPLITNUX_H
#define SPLIT_SPLITNUX_H

/**
   @file splitnux.h

   @brief Minimal container capable of characterizing split.

   @author Mark Seligman
 */

#include "splitcoord.h"
#include "typeparam.h"

#include <vector>

class SplitNux {
  static constexpr double minRatioDefault = 0.0;
  static double minRatio;
  static vector<double> splitQuant; // Where within CDF to split.

  const SplitCoord splitCoord;
  const unsigned char bufIdx;
  const IndexRange idxRange; // Indices into compressed ObsPart buffer.
  const PredictorT setIdx; // Index into runSet vector for factor split.
  const double sum; // node sum.

  IndexT lhSCount; // # samples in left split:  initialized to node value.
  double info; // Weighted variance or Gini, currently.
  IndexT lhImplicit; // # implicit indices in LHS:  initialized to node value at scheduling.

  // Accumulated during splitting:
  IndexT lhExtent; // total # indices in LHS.  Written on arg-max.

  // Copied to decision node, if arg-max.  Numeric only:
  //
  double quantRank;
  
  /**
     @brief Decrements information field and reports whether still positive.

     @param splitFrontier determines pre-existing information value to subtract.

     @bool true iff decremented information field positive.
   */
  bool infoGain(const class SplitFrontier* splitFrontier);


 public:  
/**
   @brief Builds static quantile splitting vector from front-end specification.

   @param feSplitQuant specifies the splitting quantiles for numerical predictors.
  */
  static void immutables(double minRatio_,
			 const vector<double>& feSplitQuant);

  
  /**
     @brief Empties the static quantile splitting vector.
   */
  static void deImmutables();


  /**
     @brief Trivial constructor. 'info' value of 0.0 ensures ignoring.
  */
  SplitNux() : splitCoord(SplitCoord(0,0)),
	       bufIdx(0),
	       setIdx(0),
	       sum(0.0),
	       lhSCount(0),
	       info(0.0) {
  }

  
  /**
     @brief Called by SplitCand constructor.
   */
  SplitNux(SplitCoord splitCoord_,
	   PredictorT setIdx_,
	   unsigned char bufIdx_,
	   double sum,
	   IndexT sCount,
	   double info_) :
  splitCoord(splitCoord_),
  bufIdx(bufIdx_),
  setIdx(setIdx_),
  sum(sum),
  lhSCount(sCount),
  info(info_) {
  }


  SplitNux(const DefCoord& preCand,
	   const class SplitFrontier* splitFrontier,
	   PredictorT setIdx_,
	   IndexRange range,
	   IndexT implicitCount);

  
  ~SplitNux() {
  }


  /**
     @brief Passes through to frame method.

     @return cardinality iff factor-valued predictor else zero.
   */
  PredictorT getCardinality(const class SummaryFrame*) const;


  /**
     @brief Writes the left-hand characterization of a factor-based
     split with categorical response.

     @param lhBits is a compressed representation of factor codes for the LHS.
   */
  void writeBits(const class SplitFrontier* splitFrontier,
		 PredictorT lhBits);


  /**
     @brief Writes the left-hand characterization of a factor-based
     split with numerical or binary response.

     @param cutSlot is the LHS/RHS separator position in the vector of
     factor codes maintained by the run-set.
   */
  void writeSlots(const class SplitFrontier* splitFrontier,
                  PredictorT cutSlot);
  

  void writeNum(const class SplitFrontier* sf,
		const class Accum* accum);


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
     @brief Resets trial information value of this greater.

     @param[out] runningMax holds the running maximum value.

     @return true iff value revised.
   */
  bool maxInfo(double& runningMax) const {
    if (info > runningMax) {
      runningMax = info;
      return true;
    }
    return false;
  }


  auto getPredIdx() const {
    return splitCoord.predIdx;
  }

  auto getNodeIdx() const {
    return splitCoord.nodeIdx;
  }
  

  auto getDefCoord() const {
    return DefCoord(splitCoord, bufIdx);
  }

  
  auto getSplitCoord() const {
    return splitCoord;
  }

  auto getBufIdx() const {
    return bufIdx;
  }
  
  auto getSetIdx() const {
    return setIdx;
  }

  /**
     @brief Reference getter for over-writing info member.
  */
  double& refInfo() {
    return info;
  }
  
  auto getInfo() const {
    return info;
  }
  
  /**
     @return true iff left side has no implicit indices.  Rank-based
     splits only.
   */
  bool leftIsExplicit() const {
    return lhImplicit == 0;
  }

  auto getIdxStart() const {
    return idxRange.getStart();
  }

  auto getExtent() const {
    return idxRange.getExtent();
  }

  auto getIdxEnd() const {
    return idxRange.getEnd() - 1;
  }


  auto getQuantRank() const {
    return quantRank;
  }
  

  auto getSCount() const {
    return lhSCount;
  }

  
  auto getSum() const {
    return sum;
  }
  

  auto getLHExtent() const {
    return lhExtent;
  }

  
  auto getImplicitCount() const {
    return lhImplicit;
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
    IndexRange range(getExplicitBranchStart(), getExplicitBranchExtent());
    return range;
  }
};


#endif
