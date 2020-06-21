// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file trainframe.h

   @brief Data frame representations for preformatting and training.

   @author Mark Seligman
 */

#ifndef OBS_TRAINFRAME_H
#define OBS_TRAINFRAME_H

#include "block.h"
#include "rankedframe.h"

#include <vector>
#include <memory>
using namespace std;

/**
   @brief Frame represented as row/rank summaries, with numeric block.
 */
class TrainFrame {
  const IndexT nRow;
  const PredictorT nPredNum;
  const vector<PredictorT> cardinality; // Factor predictor cardinalities.
  const PredictorT nPredFac;
  const PredictorT cardExtent; // Greatest factor footprint.
  const PredictorT nPred;
  const unique_ptr<class Coproc> coproc; // Stubbed, for now.
  unique_ptr<RankedFrame> rankedFrame;
  const unique_ptr<BlockJagged<double> > numRanked;


  inline double getNumVal(PredictorT predIdx,
                          IndexT rank) const {
    return numRanked->getVal(predIdx, rank);
  }

public:

  TrainFrame(const struct RLEFrame* rleFrame,
	     double autoCompress,
	     PredictorT predPermute,
	     bool enableCoproc,
	     vector<string>& diag);

  
  ~TrainFrame();

  
  /**
     @brief Getter for rankedFrame.
   */
  inline RankedFrame* getRankedFrame() const {
    return rankedFrame.get();
  }


  /**
     @brief Assumes numerical predictors packed in front of factor-valued.

     @return Position of fist factor-valued predictor.
  */
  inline PredictorT getFacFirst() const {
    return nPredNum;
  }

  
  /**
     @brief Determines whether predictor is numeric or factor.

     @param predIdx is internal predictor index.

     @return true iff index references a factor.
   */
  inline bool isFactor(PredictorT predIdx)  const {
    return predIdx >= getFacFirst();
  }

  /**
     @brief Looks up cardinality of a predictor.

     @param predIdx is the absolute predictor index.

     @return cardinality iff factor else 0.
   */
  inline PredictorT getCardinality(PredictorT predIdx) const {
    return predIdx < getFacFirst() ? 0 : cardinality[predIdx - getFacFirst()];
  }


  /**
     @brief Accessor for cardinality footprint.
   */
  inline auto getCardExtent() const {
    return cardExtent;
  }
  

  /**
     @brief Computes block-relative position for a predictor.

     @param[out] thisIsFactor outputs true iff predictor is factor-valued.

     @return block-relative index.
  */
  inline PredictorT getIdx(PredictorT predIdx, bool &thisIsFactor) const{
    thisIsFactor = isFactor(predIdx);
    return thisIsFactor ? predIdx - getFacFirst() : predIdx;
  }


  /**
     @brief Determines a dense position for factor-valued predictors.

     @param predIdx is a predictor index.

     @param nStride is a stride value.

     @param[out] thisIsFactor is true iff predictor is factor-valued.

     @return strided factor offset, if factor, else predictor index.
   */
  inline unsigned int getFacStride(PredictorT predIdx,
				unsigned int nStride,
				bool &thisIsFactor) const {
    PredictorT facIdx = getIdx(predIdx, thisIsFactor);
    return thisIsFactor ? nStride * getNPredFac() + facIdx : predIdx;
  }


  /**
     @return number or observation rows.
   */
  inline IndexT getNRow() const {
    return nRow;
  }

  /**
     @return number of observation predictors.
  */
  inline PredictorT getNPred() const {
    return nPred;
  }

  /**
     @return number of factor predictors.
   */
  inline PredictorT getNPredFac() const {
    return nPredFac;
  }

  /**
     @return number of numerical predictors.
   */
  inline PredictorT getNPredNum() const {
    return nPredNum;
  }


  /**
     @brief Fixes contiguous factor ordering as numerical preceding factor.

     @return Position of first numerical predictor.
  */
  static constexpr PredictorT getNumFirst() {
    return 0ul;
  }


  /**
     @brief Positions predictor within numerical block.

     @param predIdx is the core-ordered index of a predictor assumed to be numeric.

     @return Position of predictor within numerical block.
  */
  inline PredictorT getNumIdx(PredictorT predIdx) const {
    return predIdx - getNumFirst();
  }


  /**
     @brief Interpolates a numerical value from a fractional "rank".

     @param predIdx is the index of a numerical predictor.

     @param rank is a fractional rank value.

     @return interpolated value.
   */
  inline double interpolate(PredictorT predIdx,
			    double rank) const {
    IndexT rankFloor = floor(rank);
    IndexT rankCeil = ceil(rank);
    double valFloor = getNumVal(predIdx, rankFloor);
    double valCeil = getNumVal(predIdx, rankCeil);

    return valFloor + (rank - rankFloor) * (valCeil - valFloor);
  }
};

#endif
