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
   @brief Single node type for regression and classification.

   For simplicity, regression and classification variants are distinguished
   only by method name and not by subtype.  The only distinction is the
   value (and interpretation) of the 'ctg' field.  Care should be taken
   to call the appropriate method, as 'ctg' is only used as a packing
   parameter (with value zero) in the case of regression.  Subtyping seems
   to complicate the code needlessly, with a per-tree size savings of only
   'nSamp' * sizeof(uint).
 */

class SampleNux {
  static unsigned int ctgBits; // Pack:  nonzero iff categorical.
  static unsigned int ctgMask;
  static unsigned int multMask; // Masks bits not used to encode multiplicity.
  static unsigned int rightBits; // # bits to shift for left-most value.
  static unsigned int rightMask; // Mask bits not used by multiplicity, ctg.

  // Integer-sized container is likely overkill:  typically << #rows,
  // although sample weighting might yield run sizes approaching #rows.
  PackedT packed; // Packed sample count, ctg.
  double ySum; // Sum of values selected:  sample-count * y-value.
  
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
  inline IndexT getSCount() const {
    return (packed >> ctgBits) & multMask;
  }


  /**
     @brief Compound accessor.

     @param[out] ctg is category value / default:  classification / regression.

     @return sample sum.
  */
  inline double refCtg(PredictorT& ctg) const {
    ctg = getCtg();
    return getYSum();
  }


  /**
     @brief Accessor for packed sCount/ctg member.
   */
  inline auto getRight() const {
    return packed & rightMask;
  }


  /**
     @brief Produces sum of y-values over sample.

     @return sum of y-values for sample.
   */
  inline double getYSum() const {
    return ySum;
  }


  /**
     @brief Derives response category from internal encouding.

     @return response cardinality.
   */
  inline PredictorT getCtg() const {
    return packed & ctgMask;
  }


  inline auto getDelRow() const {
    return packed >> rightBits;
  }
};


#endif
