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
#include "typeparam.h"

#include <vector>

struct SplitEncoding {
  double sum;
  IndexT sCount;
  IndexT extent;

  void init() {
    sum = 0.0;
    sCount = 0;
    extent = 0;
  }
};


class SplitNux {
  static constexpr double minRatioDefault = 0.0;
  static double minRatio;
  static vector<double> splitQuant; // Where within CDF to split.

  const SplitCoord splitCoord;
  const IndexRange idxRange; // Index range of lower ObsPart buffer.
  const PredictorT setIdx; // Index into runSet vector for factor split.
  const double sum; // Initial sum, fixed by index set.
  const IndexT sCount; // Initial sample count, fixed by index set.
  const unsigned char bufIdx;

  unsigned char cutLeft; // True iff <= cut encoded, else > cut.
  unsigned char encTrue; // Whether split encoding characterizes true branch.
  IndexT implicitTrue; // # implicit indices on true branch:  initialized from index set at scheduling.
  IndexT ptId; // Index into tree:  offset from position given by index set.
  double info; // Weighted variance or Gini, currently.

  // Accumulated during splitting:
  IndexT cutExtent; // Left extent of cut point, in indices.

  // Accumulated during replay.
  SplitEncoding enc;
  
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


  void encAccum(double ySum,
		IndexT sCount) {
    enc.sum += ySum;
    enc.sCount += sCount;
  }


  void bumpExtent() {
    enc.extent++;
  }

  
  void encExtent(const IndexRange& range) {
    enc.extent += range.getExtent();
  }
  

  auto getEncodedSum() const {
    return enc.sum;
  }


  auto getEncodedSCount() const {
    return enc.sCount;
  }


  /**
     @return range associated with cut inequality.
   */
  auto getCutRange() const {
    return cutLeft ? IndexRange(idxRange.getStart(), cutExtent) :
      IndexRange(idxRange.getStart() + cutExtent, idxRange.getExtent() - cutExtent);
  }

  
  /**
     @brief Trivial constructor. 'info' value of 0.0 ensures ignoring.
  */  
  SplitNux() : splitCoord(SplitCoord()),
	       setIdx(0),
	       sum(0.0),
	       sCount(0),
	       bufIdx(0),
	       cutLeft(true),
	       encTrue(true),
	       implicitTrue(0),
	       info(0.0) {
    enc.init();
  }

  
  /**
     @brief Called by SplitCand constructor.
   */
  SplitNux(SplitCoord splitCoord_,
	   PredictorT setIdx_,
	   unsigned char bufIdx_,
	   double sum,
	   IndexT sCount_,
	   double info_) :
  splitCoord(splitCoord_),
  setIdx(setIdx_),
  sum(sum),
  sCount(sCount_),
  bufIdx(bufIdx_),
  cutLeft(true),
  encTrue(true),
  implicitTrue(0),
  info(info_) {
    enc.init();
  }


  /**
     @brief Post-split constructor for compund criteria.
   */
  SplitNux(const SplitNux* parent,
	   IndexT idx) :
    splitCoord(parent->splitCoord),
    idxRange(parent->getCutRange()),
    setIdx(parent->setIdx),
    sum(parent->getEncodedSum()),
    sCount(parent->getEncodedSCount()),
    bufIdx(parent->bufIdx),
    cutLeft(parent->cutLeft),
    encTrue(parent->encTrue),
    implicitTrue(parent->implicitTrue),
    ptId(parent->ptId + idx),
    info(sum / sCount), // ?
    cutExtent(0),
    quantRank(parent->quantRank) {
    enc.init();
  }

  /**
     @brief Transfer constructor over specified support.
   */
  SplitNux(const SplitNux* parent,
	   double supportSum,
	   IndexT supportSCount) :
    splitCoord(parent->splitCoord),
    idxRange(parent->idxRange),
    setIdx(parent->setIdx),
    sum(supportSum),
    sCount(supportSCount),
    bufIdx(parent->bufIdx),
    cutLeft(parent->cutLeft),
    encTrue(parent->encTrue),
    implicitTrue(parent->implicitTrue),
    ptId(parent->ptId),
    info(sum/sCount),
    cutExtent(0) {
    enc.init();
  }

  /**
     @brief Pre-split constructor.
   */
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
     @brief Writes the true-branch characterization of a factor-based
     split with categorical response.

     @param lhBits is a compressed representation of factor codes for the LHS.
   */

  void writeBits(const class SplitFrontier* splitFrontier,
		 PredictorT lhBits);

  
  /**
     @brief Writes the true-branch characterization of a factor-based
     split with numerical or binary response.

     @param cutSlot is the LHS/RHS separator position in the vector of
     factor codes maintained by the run-set.
   */
  void writeSlots(const class SplitFrontier* splitFrontier,
                  PredictorT cutSlot);


  void appendSlot(const class SplitFrontier* splitFrontier);

  
  /**
     @brief Fills out remaining data members from numeric split, if any.
   */
  void writeNum(const class SplitFrontier* sf,
		const class Accum* accum);


  /**
     @brief As above, but specifies sense of encoding.
   */
  void writeNum(const class Accum* accum,
		bool cutLeft,
                bool encTrue);


  /**
     @brief As above, but for factor-valued predictors.
   */
  void writeFac(IndexT sCountTrue,
		IndexT cutExtent,
		IndexT implicitTrue);

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
     @brief Indicates whether this is an empty placeholder.
   */
  inline bool noNux() const {
    return splitCoord.noCoord();
  }

  
  /**
     @return true iff true branch is encoded.  Rank-based splits only.
   */
  inline bool trueEncoding() const {
    return encTrue != 0;
  }


  inline bool leftCut() const {
    return cutLeft != 0;
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
    return sCount;
  }
  

  auto getSCountTrue() const {
    return encTrue ? getEncodedSCount() : sCount - getEncodedSCount();
  }


  auto getSum() const {
    return sum;
  }


  auto getSumTrue() const {
    return encTrue ? getEncodedSum() : sum - getEncodedSum();
  }

  
  auto getExtentTrue() const {
    return cutExtent;
  }

  
  /**
     @brief Getter for extent of encoded portion.
   */
  auto getEncodedExtent() const {
    return enc.extent;
  }

  
  auto getImplicitCount() const {
    return implicitTrue;
  }
  
  
  /**
     @return Count of indices corresponding to LHS of a rank-based split.
   */
  auto getTrueExplicit() const {
    return getExtentTrue() - implicitTrue;
  }

  /**
     @return Count of indices corresponding to FalseS of a rank-based split.
   */  
  auto getFalseExplicit() const {
    return getExtent() - getTrueExplicit();
  }


  auto getStartFalse() const {
    return idxRange.getStart() + getTrueExplicit();
  }

  
  /**
     @return Starting index of an explicit branch.  Defaults to left if
     both branches explicit.  Rank-based splits only.
   */
  auto getExplicitBranchStart() const {
    return trueEncoding() ? idxRange.getStart() : getStartFalse();
  }


  /**
     @return Extent of an explicit branch.  Defaults to left if both
     branches explicit.  Rank-based splits only.
   */
  auto getExplicitBranchExtent() const {
    return trueEncoding() ? getTrueExplicit() : getFalseExplicit();
  }


  /**
     @brief Accessor for left index range.
   */
  auto getRangeTrue() const {
    return IndexRange(idxRange.getStart(), getExtentTrue());
  }


  auto getRangeFalse() const {
    return IndexRange(getStartFalse(), getFalseExplicit());
  }
  

  auto getEncodedRange() const {
    return encTrue ? getRangeTrue() : getRangeFalse();
  }
};


#endif
