// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file obscell.h

   @brief Compact observation representation for splitting.

   @author Mark Seligman
 */

#ifndef OBS_OBSCELL_H
#define OBS_OBSCELL_H

#include "typeparam.h"
#include "samplenux.h"

#include <cmath>

#include<iostream>
/**
   @brief Compact representation for splitting.
 */
class ObsCell {
  static IndexT maxSCount;
  static unsigned int ctgBits; // Pack:  nonzero iff categorical.
  static unsigned int ctgMask;
  static unsigned int multBits;
  static unsigned int multMask; // Masks bits not encoding multiplicity.

  static double scale; // Coefficent scaling response to < 0.5.
  static double recipScale; // Reciprocal simplifies division.
  
  IndexT rank;
  FltVal yVal;


  /**
     @brief Unpacks float into numerical representation.
   */
  inline void refReg(unsigned int& sCount,
		     double& ySum) const {
    unsigned int rounded = round(yVal); // Rounds nearest.
    sCount = (rounded >> ctgBits) & multMask;
    ySum = scale * (yVal - rounded);
  }


  /**
     @brief Unpacks float into categorical representation.

     The fractional component of yVal is a scaled class weight, and
     is therefore positive, so truncation (round-toward-zero) may be
     used instead of a slower call to round().

     Class weights are proportional, so it may be possible to avoid
     descaling.
   */
  inline void refCtg(unsigned int& sCount,
		     double& ySum,
		     unsigned int& yCtg) const {
    unsigned int rounded = yVal;  // Rounds toward zero.
    sCount = (rounded >> ctgBits) & multMask;
    ySum = scale * (yVal - rounded);
    yCtg = rounded & ctgMask;
  }

 public:

  /**
     @brief Sets internal packing parameters.
   */
  static void setShifts(IndexT maxSCount,
			unsigned int ctgBits_,
		        unsigned int ctgMask_);


  static void setScale(double yMax);

  
  static void deImmutables();


  /**
     @brief Initializes by copying response and joining sampled rank.

     Rank is only used to break ties and elaborate argmax summaries.
     It may be possible to exclude them.

     @param sNux summarizes response sampled at row.

     @param rank_ is the predictor rank sampled at a given row.
  */
  inline void join(const SampleNux& sNux,
		   IndexT rank_) {
    rank = rank_;
    yVal = sNux.getYSum() * recipScale + sNux.getRight();
  }


  /**
     @brief Derives sample count from internal encoding.

     @return sample count.
   */
  inline IndexT getSCount() const {
    unsigned int rounded = round(yVal);
    return (rounded >> ctgBits) & multMask;
  }


  /**
     @brief Produces sum of y-values over sample.

     @return sum of y-values for sample.
   */
  inline double getYSum() const {
    return scale * (yVal - round(yVal));
  }


  /**
     @brief Getter for rank or factor group.

     @return rank value.
   */
  inline IndexT getRank() const {
    return rank;
  }


  /**
     @brief Derives response category from internal encouding.

     @return response cardinality.
   */
  inline PredictorT getCtg() const {
    unsigned int rounded = round(yVal);
    return rounded & ctgMask;
  }


  /**
     @brief Outputs statistics appropriate for regression.

     @return true iff run state changes.
   */
  inline void regInit(RunNux& nux) const {
    nux.code = rank;
    refReg(nux.sCount, nux.sum);
  }

  
  /**
     @brief Accumulates statistics for an existing run.

     @return true iff the current cell continues a run.
   */
  inline bool regAccum(RunNux& nux) const {
    if (nux.code == rank) {
      unsigned int sCount;
      double ySum;
      refReg(sCount, ySum);
      nux.sum += ySum;
      nux.sCount += sCount;
      return true;
    }
    else {
      return false;
    }
  }


  /**
     @brief Compound accessor for regression.  Cannot be used for
     classification, as 'sCount' value reported here not unpacked.

     @param[out] ySum outputs the response value.

     @param[out] sCount outputs the multiplicity of the row in this sample.

     @return rank of predictor value at sample.
   */
  inline double regFields(IndexT& sCount,
			  IndexT& rank_) const {
    double ySum;
    refReg(sCount, ySum);
    rank_ = rank;
    return ySum;
  }


  /**
     @brief Outputs statistics appropriate for classification.

     @param[out] nux accumulates run statistics.

     @param[in, out] sumBase accumulates run response by category.
   */
  inline void ctgInit(RunNux& nux,
		      double* sumBase) const {
    unsigned int sCount, yCtg;
    double ySum;
    refCtg(sCount, ySum, yCtg);
      
    nux.code = rank;
    nux.sum = ySum;
    nux.sCount = sCount;
    sumBase[yCtg] = ySum;
  }


  /**
     @brief Accumulates statistics for an existing run.

     @return true iff the current cell continues a run.
   */
  inline bool ctgAccum(RunNux& nux,
		       double* sumBase) const {
    if (nux.code == rank) {
      unsigned int sCount, yCtg;
      double ySum;
      refCtg(sCount, ySum, yCtg);

      nux.sum += ySum;
      nux.sCount += sCount;
      sumBase[yCtg] += ySum;
      return true;
    }
    else {
      return false;
    }
  }


  /**
     @brief Compound accessor for classification.  Can be
     called for regression if '_yCtg' value ignored.

     @param[out] ySum is the proxy response value.

     @param[out] sCount the sample count.

     @param[out] yCtg is the true response value.

     @return predictor rank.
   */
  inline IndexT ctgFields(double& ySum,
			  IndexT& sCount,
			  PredictorT& yCtg) const {
    refCtg(sCount, ySum, yCtg);
    return rank;
  }
};

#endif
