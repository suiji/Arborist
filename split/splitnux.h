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

#include "stagedcell.h"
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

  const StagedCell* cell; // Copied from PreCand.
  uint32_t randVal;
  IndexT sigIdx; // Index into accumulator workspace.
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
     @retrun true iff run's range exceeds bounds.
   */
  bool isImplicit(const struct RunNux& nux) const;

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
    cell(nullptr),
    randVal(0),
    sigIdx(0),
    sum(0.0),
    sCount(0),
    ptId(0),
    info(0.0) {
  }

  
  /**
     @brief Copy constructor:  post splitting.

  SplitNux(const SplitNux& nux) :
    mrra(nux.mrra),
    randVal(nux.randVal),
    obsRange(nux.obsRange),
    sigIdx(nux.sigIdx),
    sum(nux.sum),
    sCount(nux.sCount),
    ptId(nux.ptId),
    info(nux.info) {
  }

  SplitNux& operator= (const SplitNux& nux) {
    mrra = nux.mrra;
    randVal = nux.randVal;
    obsRange = nux.obsRange;
    sigIdx = nux.sigIdx;
    sum = nux.sum;
    sCount = nux.sCount;
    ptId = nux.ptId;
    info = nux.info;

    return *this;
  }
  */
  
  /**
     @brief Transfer constructor over iteratively-encoded IndexSet.

     Post-splitting.

     @param idx positions nux within a multi-criterion set.
   */
  SplitNux(const SplitNux& parent,
	   const class SplitFrontier* sf,
	   bool sense,
	   IndexT idx = 0);


  /**
     @brief Pre-split constructor.
   */
  SplitNux(const StagedCell* cell_,
	   double randVal_,
	   const class SplitFrontier* splitFrontier);


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
     @brief Running argmax over info members.

     @param[in, out] amn holds the running argmax nux.

  inline void maxInfo(const SplitNux*& amn) const {
    if (info > amn->info || (info == amn->info && info > 0.0 && randVal > amn->randVal))
      amn = this;
  }
  */  

  inline bool maxInfo(const SplitNux& amn) const {
    return (info > amn.info) || (info == amn.info && info > 0.0 && randVal > amn.randVal);
  }
  

  auto getPTId() const {
    return ptId;
  }


  IndexT getNMissing() const {
    return cell->obsMissing;
  }

  
  /**
     @return # observations preceding implicit, if any.
   */
  IndexT getPreresidual() const {
    return cell->preResidual;
  }


  auto getRunCount() const {
    return cell->getRunCount();
  }


  auto getPredIdx() const {
    return cell->getPredIdx();
  }

  auto getNodeIdx() const {
    return cell->getNodeIdx();
  }

  
  auto getStagedCell() const {
    return cell;
  }

  
  auto getBufIdx() const {
    return cell->bufIdx;
  }
  
  auto getSigIdx() const {
    return sigIdx;
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
    return cell == nullptr || cell->coord.noCoord();
  }


  inline IndexRange getRange() const {
    return cell->getObsRange();
  }

  
  inline IndexT getObsStart() const {
    return cell->getObsRange().getStart();
  }

  
  inline IndexT getObsExtent() const {
    return cell->getObsRange().getExtent();
  }


  /**
     @return inattainable position beyond observation buffer top.
   */
  inline IndexT getObsEnd() const {
    return cell->getObsRange().getEnd();
  }


  /**
     @return Count of implicit observations associated with cell.
  */   
  IndexT getImplicitCount() const {
    return cell->obsImplicit;
  }


  inline IndexT getSCount() const {
    return sCount;
  }
  

  inline double getSum() const {
    return sum;
  }

  
  /**
     @brief Randomizes decision to invert the test at run time.

     A bit somewhat removed from the least-significant position is tested.
     A bias has been observed in the lowest-order bits for some PRNGs.

     @return true iff test is to be inverted.
   */
  bool invertTest() const {
    return randVal & 0x80000000;
  }
  
  
  /**
     @brief Looks up splitting quantile for associated predictor.
   */
  auto getSplitQuant() const {
    return splitQuant[getPredIdx()];
  }
};


#endif
