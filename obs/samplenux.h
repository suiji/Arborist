// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplenux.h

   @brief Class definitions for sample-related containers.

   @author Mark Seligman
 */

#ifndef FRONTIER_SAMPLENUX_H
#define FRONTIER_SAMPLENUX_H

#include "typeparam.h"
#include "samplernux.h"

#include <vector>

/**
   @brief Container for compressed sampled response.
 */
class SampleNux {
  static unsigned int ctgBits; ///< Pack:  nonzero iff categorical.
  static unsigned int ctgMask;
  static unsigned int multMask; ///< Masks bits not used to encode multiplicity.
  static unsigned int rightBits; ///< # bits to shift for left-most value.
  static unsigned int rightMask; ///< Mask bits not used by multiplicity, ctg.

  // Integer-sized container is likely overkill:  typically << #rows,
  // although sample weighting might yield run sizes approaching #rows.
  PackedT packed; ///< Packed sample count, ctg.
  double ySum; ///< Sum of values selected:  sample-count * y-value.
  
 public:

  /**
     @brief Computes a packing width sufficient to hold all (zero-based) response category values.

     @param ctgWidth is the response cardinality.
  */
  static void setShifts(PredictorT ctgWidth,
			IndexT maxSCount);
  
  /**
    @brief Resets to static initialization.
  */
  static void deImmutables();

  
  /**
     @brief Initializes to summary values passed.

     @param yVal is reponse value.

     @param nux encodes sampleCount and row delta.

     @param ctg is the response category, if classification.
  */
  SampleNux(double yVal,
	    const SamplerNux& nux,
            PredictorT ctg = 0) :
    packed((PackedT(nux.getDelRow() << rightBits) | (nux.getSCount() << ctgBits) | ctg)),
    ySum(yVal * nux.getSCount()) {
  }


  SampleNux() = default;


  /**
     @brief Derives sample count from internal encoding.

     @return sample count.
   */
  IndexT getSCount() const {
    return (packed >> ctgBits) & multMask;
  }


  /**
     @brief Compound accessor.

     @param[out] ctg is category value / default:  classification / regression.

     @return sample sum.
  */
  double refCtg(PredictorT& ctg) const {
    ctg = getCtg();
    return getYSum();
  }


  /**
     @brief Accessor for packed sCount/ctg member.
   */
  auto getRight() const {
    return packed & rightMask;
  }


  /**
     @brief Produces sum of y-values over sample.

     @return sum of y-values for sample.
   */
  double getYSum() const {
    return ySum;
  }


  /**
     @brief Derives response category from internal encouding.

     @return response cardinality.
   */
  PredictorT getCtg() const {
    return packed & ctgMask;
  }


  auto getDelRow() const {
    return packed >> rightBits;
  }


  /**
     @brief Decrements sum value.

     @param decr is the per-sample amount to decrement.
     
     @return decremented sum value.
   */
  double decrementSum(double decr) {
    ySum -= decr * getSCount();
    return ySum;
  }
};


#endif
