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
#include "sumcount.h"
#include "mrra.h"
#include "typeparam.h"

#include <vector>


/**
   @brief Coordinates and node summary for a splitting candidate.

   Summary and coordinate members initialized and not changed.  Information
   value updated by splitting method.
 */
class SplitNux {
  static constexpr double minRatioDefault = 0.0;
  static double minRatio;

  MRRA mrra; // Cell coordinates of pre-cand; delIdx implicitly zero.
  IndexT implicitCount; // Extracted from StageCount; rank count to accumulator.
  IndexRange idxRange; // SampleRank index range of cell column.
  IndexT accumIdx; // Index into accumulator workspace.
  double sum; // Initial sum, fixed by index set (node).
  IndexT sCount; // Initial sample count, fixed by index set.
  IndexT ptId; // Index into tree:  offset from position given by index set.

  // Set during splitting:
  double info; // CART employs Weighted variance or Gini.
  
public:  
  static vector<double> splitQuant; // Where within CDF to cut.  MOVE to CutSet.
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
     @brief Revises information value from accumulator's contents.
   */
  void infoGain(const class Accum* accum);
  

  /**
     @return desired cut range.
   */
  IndexRange cutRange(const class CutSet* cutSet,
		      bool leftRange) const;
  

  /**
     @brief Computes cut-based left range for numeric splits.
   */
  IndexRange cutRangeLeft(const class CutSet* cutSet) const;


  /**
     @brief Computes cut-based right range for numeric splits.
   */
  IndexRange cutRangeRight(const class CutSet* cutSet) const;


  /**
     @brief Trivial constructor. 'info' value of 0.0 ensures ignoring.
  */  
  SplitNux() :
    mrra(MRRA()),
    implicitCount(0),
    accumIdx(0),
    sum(0.0),
    sCount(0),
    ptId(0),
    info(0.0) {
  }

  
  /**
     @brief Copy constructor:  post splitting.
   */
  SplitNux(const SplitNux& nux) :
    mrra(nux.mrra),
    implicitCount(nux.implicitCount),
    idxRange(nux.idxRange),
    accumIdx(nux.accumIdx),
    sum(nux.sum),
    sCount(nux.sCount),
    ptId(nux.ptId),
    info(nux.info) {
  }

  SplitNux& operator= (const SplitNux& nux) {
    mrra = nux.mrra;
    implicitCount = nux.implicitCount;
    idxRange = nux.idxRange;
    accumIdx = nux.accumIdx;
    sum = nux.sum;
    sCount = nux.sCount;
    ptId = nux.ptId;
    info = nux.info;

    return *this;
  }

  
  /**
     @brief Transfer constructor over iteratively-encoded IndexSet.

     @param idx positions nux within a multi-criterion set.
   */
  SplitNux(const SplitNux& parent,
	   const class SplitFrontier* sf,
	   bool sense,
	   IndexT idx = 0);


  /**
     @brief Pre-split constructor.
   */
  SplitNux(const class PreCand& preCand,
	   const class SplitFrontier* splitFrontier);


  /**
     @brief Reports whether frame identifies underlying predictor as factor-valued.

     @return true iff splitting predictor is a factor.
   */
  bool isFactor(const class SplitFrontier* sf) const;
  

  /**
     @brief Passes through to frame method.

     @return cardinality iff factor-valued predictor else zero.
   */
  PredictorT getCardinality(const class TrainFrame*) const;


  
  /**
     @brief Reports whether potential split be informative with respect to a threshold.

     @param minInfo is an information threshold.

     @return true iff information content exceeds the threshold.
   */
  bool isInformative(double minInfo) const {
    return info > minInfo;
  }


  /**
     @return minInfo threshold.
   */
  double getMinInfo() const {
    return minRatio * info;
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


  auto getPTId() const {
    return ptId;
  }

  
  auto getPredIdx() const {
    return mrra.splitCoord.predIdx;
  }

  auto getNodeIdx() const {
    return mrra.splitCoord.nodeIdx;
  }
  
  auto getMRRA() const {
    return mrra;
  }

  auto getBufIdx() const {
    return mrra.bufIdx;
  }
  
  auto getAccumIdx() const {
    return accumIdx;
  }

  
  auto getInfo() const {
    return info;
  }

  
  void setInfo(double info) {
    this->info = info;
  }


  /**
     @brief Indicates whether this is an empty placeholder.
   */
  inline bool noNux() const {
    return mrra.splitCoord.noCoord();
  }


  auto getRange() const {
    return idxRange;
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

  auto getSCount() const {
    return sCount;
  }
  

  auto getSum() const {
    return sum;
  }

  
  /**
     @return Count of implicit indices associated with IndexSet.
  */   
  IndexT getImplicitCount() const {
    return implicitCount;
  }


  auto getSplitQuant() const {
    return splitQuant[getPredIdx()];
  }
};


#endif
