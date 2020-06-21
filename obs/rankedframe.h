// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rankedframe.h

   @brief Class definitions for maintenance of predictor ordering.

   @author Mark Seligman
 */

#ifndef PARTITION_RANKEDFRAME_H
#define PARTITION_RANKEDFRAME_H

#include <vector>
#include <cmath>

#include "rleframe.h"

using namespace std;

#include "typeparam.h"


/**
  @brief Rank orderings of predictors.
*/
class RankedFrame {
  const class RLEFrame* rleFrame;
  const IndexT nRow;
  const PredictorT nPred;
  const PredictorT noRank; // Inattainable rank value.
  const PredictorT predPermute; // Predictor undergoing permutation.
  PredictorT nPredDense;
  vector<IndexT> denseIdx;

  PredictorT nonCompact;  // Total count of uncompactified predictors.
  IndexT lengthCompact;  // Sum of compactified lengths.
  vector<PredictorT> denseRank;
  vector<IndexT> explicitCount; // Refreshed at each tree.
  vector<IndexT> safeOffset; // Predictor offset within SamplePred[].
  const IndexT denseThresh; // Threshold run length for autocompression.

  /**
     @brief Walks the design matrix as RLE entries, merging adjacent entries of identical rank.
  */
  void denseBlock();


  /**
     @brief Determines a dense rank for the predictor, if any.
   */
  void setDense(PredictorT predIdx);


  /**
     @brief Determines whether predictor to be stored densely and updates
     storage accumulators accordingly.

     @param predIdx is the predictor under consideration.
  */
  void accumOffsets(PredictorT predIdx);

 public:

  // Factory parametrized by coprocessor state.
  static RankedFrame *Factory(const class Coproc *coproc,
			      const class RLEFrame* rleFrame,
                              double autoCompress,
			      PredictorT predPermute);

  /**
     @brief Constructor for row, rank passed from front end as parallel arrays.

     @param feRow is the vector of rows allocated by the front end.

     @param feRank is the vector of ranks allocated by the front end.
 */
  RankedFrame(const class RLEFrame* rleFrame,
              double autoCompress,
	      PredictorT predPermute);

  virtual ~RankedFrame();


  inline IndexT getNRow() const {
    return nRow;
  }
  
  
  inline PredictorT getNPred() const {
    return nPred;
  }


  inline IndexT NoRank() const {
    return noRank;
  }


  /**
     @brief Accessor for dense rank value associated with a predictor.

     @param predIdx is the predictor index.

     @return dense rank assignment for predictor.
   */
  IndexT getDenseRank(PredictorT predIdx) const{
    return denseRank[predIdx];
  }

  
  /**
     @brief Computes a conservative buffer size, allowing strided access
     for noncompact predictors but full-width access for compact predictors.

     @param stride is the desired strided access length.

     @return buffer size conforming to conservative constraints.
   */
  IndexT safeSize(IndexT stride) const {
    return nonCompact * stride + lengthCompact; // TODO:  align.
  }

  
  /**
     @brief Computes conservative offset for storing predictor-based
     information.

     @param predIdx is the predictor index.

     @param stride is the multiplier for strided access.

     @param extent outputs the number of slots avaiable for staging.

     @return safe range.
  */
  IndexRange getSafeRange(PredictorT predIdx,
			  IndexT stride) const {
    if (denseRank[predIdx] == noRank) {
      return IndexRange(safeOffset[predIdx] * stride, stride);
    }
    else {
      return IndexRange(nonCompact * stride + safeOffset[predIdx], explicitCount[predIdx]);
    }
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
  inline const vector<IndexT> &getDenseIdx() const {
    return denseIdx;
  }

  /**
     @brief Loops through the predictors to stage.
  */
  vector<IndexT> stage(const class Sample* sample,
		       class ObsPart* obsPart);

  
  /**
     @brief Stages ObsPart objects in non-decreasing predictor order.

     @param predIdx is the predictor index.
  */
  IndexT stage(const class Sample* sample,
	       PredictorT predIdx,
	       class ObsPart* obsPart) const;
};


#endif

