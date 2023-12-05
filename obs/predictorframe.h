// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file layout.h

   @brief Lays out predictor observations for staging.

   @author Mark Seligman
 */

#ifndef OBS_PREDICTORFRAME_H
#define OBS_PREDICTORFRAME_H

#include "typeparam.h"
#include "rleframe.h"

#include <vector>
#include <cmath>

using namespace std;

/**
   @brief Characterizes predictor contents via implicit rank and explicit count.
 */
struct Layout {
  IndexT rankImpl; // Implicit rank, if any.
  IndexT countExpl; // Count of explicit samples.
  IndexT rankMissing; ///< rank denoting missing data, if any.
  IndexT denseIdx;
  IndexT safeOffset; // Base of staged predictor.

  Layout() = default;

  
  Layout(IndexT rankImpl_,
	   IndexT countExpl_,
	   IndexT rankMissing_) :
    rankImpl(rankImpl_),
    countExpl(countExpl_),
    rankMissing(rankMissing_) {
  }
};


/**
  @brief Rank orderings of predictors.
*/
class PredictorFrame {
  const unique_ptr<struct RLEFrame> rleFrame;
  const IndexT nObs;
  const unique_ptr<class Coproc> coproc; // Stubbed, for now.
  const PredictorT nPredNum;
  const vector<PredictorT> factorTop; ///< # levels; 0 iff numeric.
  const vector<PredictorT> factorExtent; ///< # runs, per factor.
  const PredictorT nPredFac;
  const PredictorT nPred;
  const vector<PredictorT> feIndex; ///> Maps core predictor index to user position.
  const PredictorT noRank; // Inattainable rank value.
  const IndexT denseThresh; // Threshold run length for autocompression.

  vector<vector<IndexT>> row2Rank;
  PredictorT nonCompact;  // Total count of uncompactified predictors.
  IndexT lengthCompact;  // Sum of compactified lengths.
  vector<Layout> implExpl;
  
  
  double getNumVal(PredictorT predIdx,
                          IndexT rank) const {
    return rleFrame->numRanked[predIdx][rank];
  }


  /**
     @brief Assigns factor cardinalities RLE frame.

     @return vector of factor cardinalities.
   */
  vector<PredictorT> cardinalities() const;

  
  /**
     @brief Assigns factor extents from number of unique factor levels.

     @return vector of factor extents.
   */
  vector<PredictorT> extents() const;

  
  /**
     @brief Assigns mapping from core to front-end predictor index.

     @return vector of index mappings.
   */
  vector<PredictorT> mapPredictors(const vector<unsigned int>& factorTop) const;

  
  /**
     @brief Determines whether predictor to be stored densely and updates
     storage accumulators accordingly.
  */
  void obsPredictorFrame();


  /**
     @brief Walks the design matrix as RLE entries, merging adjacent entries of identical rank.
  */
  vector<Layout> denseBlock();


  /**
     @brief Determines a dense rank for the predictor, if any.
   */
  Layout surveyRanks(PredictorT predIdx);

  
public:

  // Factory parametrized by coprocessor state.
  static PredictorFrame *Factory(unique_ptr<RLEFrame> rleFrame,
				 const class Coproc *coproc,
				 double autoCompress,
				 vector<string>& diag);


  /**
     @brief Constructor for row, rank passed from front end as parallel arrays.

     @param feRow is the vector of rows allocated by the front end.

     @param feRank is the vector of ranks allocated by the front end.
 */
  PredictorFrame(unique_ptr<RLEFrame> rleFrame_,
		 double autoCompress,
		 bool enableCoproc,
		 vector<string>& diag);


  /**
     @return number of observation predictors.
  */
  PredictorT getNPred() const {
    return nPred;
  }

  
  /**
     @return number of factor predictors.
   */
  PredictorT getNPredFac() const {
    return nPredFac;
  }


  /**
     @return number of numerical predictors.
   */
  PredictorT getNPredNum() const {
    return nPredNum;
  }


  IndexT getNoRank() const {
    return noRank;
  }


  /**
     @return rank denoting missing data, if any.
   */
  IndexT getMissingRank(PredictorT predIdx) const {
    return implExpl[predIdx].rankMissing;
  }


  /**
     @brief Computes conservative offset for storing predictor-based
     information.

     @param predIdx is the predictor index.

     @param sampleCount serves as a multiplier for strided access.

     @return safe range.
  */
  IndexRange getSafeRange(PredictorT predIdx,
			  IndexT sampleCount) const;


  /**
     @brief Accessor for dense rank value associated with a predictor.

     @param predIdx is the predictor index.

     @return residual rank assignment for predictor.
   */
  IndexT getImplicitRank(PredictorT predIdx) const{
    return implExpl[predIdx].rankImpl;
  }

  
  /**
     @brief Computes a conservative buffer size, allowing strided access
     for noncompact predictors but full-width access for compact predictors.

     @param sampleCount is the desired strided access length.

     @return buffer size conforming to conservative constraints.
   */
  IndexT getSafeSize(IndexT sampleCount) const {
    return nonCompact * sampleCount + lengthCompact; // TODO:  align.
  }


  /**
     @brief Accessor for dense index vector.

     @return reference to vector.
   */
  vector<IndexT> getDenseIdx() const {
    vector<IndexT> denseIdx(nPred);
    PredictorT predIdx = 0;
    for (auto ie : implExpl) {
      denseIdx[predIdx++] = ie.denseIdx;
    }
    return denseIdx;
  }


  const vector<IndexT>& getRanks(PredictorT predIdx) const {
    return row2Rank[predIdx];
  }

  
  const vector<PredictorT>& getPredMap() const {
    return feIndex;
  }
  

  const vector<RLEVal<szType>>& getRLE(PredictorT predIdx) const {
    return rleFrame->getRLE(feIndex[predIdx]);
  }


  IndexT getRankMax(PredictorT predIdx) const {
    return rleFrame->getRLE(feIndex[predIdx]).back().val;
  }

  
  /**
     @brief Determines whether predictor is numeric or factor.

     @param predIdx is internal predictor index.

     @return true iff index references a factor.
   */
  bool isFactor(PredictorT predIdx)  const {
    return predIdx >= nPredNum;
  }


  /**
     @brief Looks up factorTop of a predictor.

     @param predIdx is the absolute predictor index.

     @return top observed index value (zero iff non-factor).
   */
  PredictorT getFactorExtent(const class SplitNux& nux) const;


  /**
     @brief Accessor for factorTop footprint.
   */
  auto getFactorExtent() const {
    return factorExtent.empty() ? 0 : *max_element(factorExtent.begin(), factorExtent.end());
  }
  

  /**
     @brief Determines a dense position for factor-valued predictors.

     @param predIdx is a predictor index.

     @param nStride is a stride value.

     @param[out] thisIsFactor is true iff predictor is factor-valued.

     @return strided factor offset, if factor, else predictor index.
   */
  unsigned int getFacStride(PredictorT predIdx,
				   unsigned int nStride,
				   bool& thisIsFactor) const {
    thisIsFactor = isFactor(predIdx);
    PredictorT blockIdx = rleFrame->getBlockIdx(feIndex[predIdx]);
    return thisIsFactor ? nStride * getNPredFac() + blockIdx : predIdx;
  }


  /**
     @brief Fixes contiguous factor ordering as numerical preceding factor.

     @return Position of first numerical predictor.
  */
  static constexpr PredictorT getNumFirst() {
    return 0ul;
  }


  /**
     @brief Positions predictor within typed block.

     @param predIdx is the core-ordered predictor index.

     @return Position of predictor within its block.
  */
  PredictorT getTypedIdx(PredictorT predIdx) const {
    return rleFrame->getBlockIdx(feIndex[predIdx]);
  }


  /**
     @brief Interpolates a numerical value from a fractional "rank".

     @param predIdx is the index of a numerical predictor.

     @param rank is a fractional rank value.

     @return interpolated value.
   */
  double interpolate(PredictorT predIdx,
			    double rank) const {
    IndexT rankFloor = floor(rank);
    IndexT rankCeil = ceil(rank);
    double valFloor = getNumVal(predIdx, rankFloor);
    double valCeil = getNumVal(predIdx, rankCeil);

    return valFloor + (rank - rankFloor) * (valCeil - valFloor);
  }

  /**
     @brief Passes through to local implementation.
   */
  bool isFactor(const class SplitNux& nux) const;
};


#endif

