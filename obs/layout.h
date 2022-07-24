// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file layout.h

   @brief Lays out observations for staging.

   @author Mark Seligman
 */

#ifndef OBS_LAYOUT_H
#define OBS_LAYOUT_H

#include "typeparam.h"
#include "rleframe.h"


#include <vector>

using namespace std;

/**
   @brief Characterizes predictor contents via implicit rank and explicit count.
 */
struct ImplExpl {
  IndexT rankImpl; // Implicit rank, if any.
  IndexT countExpl; // Count of explicit samples.
  IndexT denseIdx;
  IndexT safeOffset; // Base of staged predictor.

  ImplExpl() = default;

  
  ImplExpl(IndexT rankImpl_,
	   IndexT countExpl_) :
    rankImpl(rankImpl_),
    countExpl(countExpl_) {
  }
};


/**
  @brief Rank orderings of predictors.
*/
class Layout {
  const class TrainFrame* trainFrame;
  const IndexT nRow;
  const PredictorT nPred;
  const PredictorT noRank; // Inattainable rank value.
  vector<vector<IndexT>> row2Rank;
  PredictorT nPredDense;

  PredictorT nonCompact;  // Total count of uncompactified predictors.
  IndexT lengthCompact;  // Sum of compactified lengths.
  const IndexT denseThresh; // Threshold run length for autocompression.

  vector<ImplExpl> implExpl;
  
  /**
     @brief Walks the design matrix as RLE entries, merging adjacent entries of identical rank.
  */
  vector<ImplExpl> denseBlock(const class TrainFrame* trainFrame);


  /**
     @brief Determines a dense rank for the predictor, if any.
   */
  ImplExpl setDense(const class TrainFrame* trainFrame,
		    PredictorT predIdx);


public:

  /**
     @brief Determines whether predictor to be stored densely and updates
     storage accumulators accordingly.
  */
  void accumOffsets();


  // Factory parametrized by coprocessor state.
  static Layout *Factory(const class Coproc *coproc,
			      const class TrainFrame* trainFrame,
			 double autoCompress);

  
  /**
     @brief Constructor for row, rank passed from front end as parallel arrays.

     @param feRow is the vector of rows allocated by the front end.

     @param feRank is the vector of ranks allocated by the front end.
 */
  Layout(const class TrainFrame* trainFrame,
	 double autoCompress);


  inline IndexT getNRow() const {
    return nRow;
  }
  
  
  inline PredictorT getNPred() const {
    return nPred;
  }


  inline IndexT getNoRank() const {
    return noRank;
  }


  /**
     @brief Computes conservative offset for storing predictor-based
     information.

     @param predIdx is the predictor index.

     @param bagCount serves as a multiplier for strided access.

     @return safe range.
  */
  IndexRange getSafeRange(PredictorT predIdx,
			  IndexT bagCount) const;


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

     @param bagCount is the desired strided access length.

     @return buffer size conforming to conservative constraints.
   */
  IndexT getSafeSize(IndexT bagCount) const {
    return nonCompact * bagCount + lengthCompact; // TODO:  align.
  }

  
  /**
     @brief Getter for count of dense predictors.

     @return number of dense predictors.
   */
  inline PredictorT getNPredDense() const {
    return nPredDense;
  }


  /**
     @brief Accessor for dense index vector.

     @return reference to vector.
   */
  inline vector<IndexT> getDenseIdx() const {
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

  
  const vector<RLEVal<unsigned int>>& getRLE(PredictorT predIdx) const;
};


#endif

