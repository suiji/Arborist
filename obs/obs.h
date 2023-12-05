// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file obs.h

   @brief Compact observation representation for splitting.

   @author Mark Seligman
 */

#ifndef OBS_OBS_H
#define OBS_OBS_H

#include "typeparam.h"
#include "samplenux.h"
#include "runsig.h"

#include <cmath>


/**
   @brief Masks lowest-order bits for non-numeric values.

   Ideally, the observation statistics would be encapsulated within
   two 16-bit floating-point containers, permitting the sample count
   to take on non-integer values.
 */
union ObsPacked {
  FltVal num;
  uint32_t bits;
};


/**
   @brief Compact representation for splitting.
 */
class Obs {
  static const unsigned int tieMask = 1ul; ///< Mask bit for tie encoding.
  static const unsigned int ctgLow = 1ul; ///< Low bit position of ctg.
  static unsigned int ctgMask; ///< Masks bits not encoding category.
  static unsigned int multLow; ///< Low bit position of multiplicity.
  static unsigned int multMask; ///< Masks bits not encoding multiplicity.
  static unsigned int numMask; ///< Masks bits not encoding numeric.
  
  ObsPacked obsPacked;
  
 public:

  /**
     @brief Derives sample count from internal encoding.

     @return sample count.
   */
  unsigned int getSCount() const {
    return 1 + ((obsPacked.bits >> multLow) & multMask);
  }


  /**
     @return sum of y-values for sample.
   */
  FltVal getYSum() const {
    ObsPacked fltPacked = obsPacked;
    fltPacked.bits &= numMask;
    return fltPacked.num;
  }


  /**
     @brief Derives response category from internal encouding.

     @return response cardinality.
   */
  PredictorT getCtg() const {
    return (obsPacked.bits >> ctgLow) & ctgMask;
  }


  /**
     @brief Sets internal packing parameters.
   */
  static void setShifts(unsigned int ctgBits,
		        unsigned int multBits);

  
  static void deImmutables();


  bool isTied() const {
    return (obsPacked.bits & tieMask) != 0;
  }


  /**
     @brief Packs sample and tie information.

     @param sNux summarizes response sampled at row.

     @param tie indicates whether previous obs has same rank.
  */
  void join(const SampleNux& sNux,
		   bool tie) {
    ObsPacked fltPacked;
    fltPacked.num = sNux.getYSum();
    obsPacked.bits = (fltPacked.bits & numMask) + ((sNux.getSCount() - 1) << multLow) + (sNux.getCtg() << ctgLow) + (tie ? 1 : 0); 
  }

  
  /**
     @brief Sets/unsets tie bit.
  */
  void setTie(bool tie) {
    if (tie)
      obsPacked.bits |= 1ul;
    else
      obsPacked.bits &= ~1ul;
  }


  /**
     @brief Outputs statistics appropriate for regression.

     @return true iff run state changes.
   */
  void regInit(RunNux& nux) const {
    nux.sumCount = SumCount(getYSum(), getSCount());
  }

  
  /**
     @brief Accumulates statistics for an existing run.

     @return true iff the current cell continues a run.
   */
  bool regAccum(RunNux& nux) const {
    if (isTied()) {
      nux.sumCount += SumCount(getYSum(), getSCount());
      return true;
    }
    else {
      return false;
    }
  }


  /**
     @brief Outputs statistics appropriate for classification.

     @param[out] nux accumulates run statistics.

     @param[in, out] sumBase accumulates run response by category.
   */
  void ctgInit(RunNux& nux,
		      double* sumBase) const {
    nux.sumCount = SumCount(getYSum(), getSCount());
    sumBase[getCtg()] = nux.sumCount.sum;
  }


  /**
     @brief Accumulates statistics for an existing run.

     @return true iff the current cell continues a run.
   */
  bool ctgAccum(RunNux& nux,
		       double* sumBase) const {
    if (isTied()) {
      double ySum = getYSum();
      nux.sumCount += SumCount(ySum, getSCount());
      sumBase[getCtg()] += ySum;
      return true;
    }
    else {
      return false;
    }
  }
};

#endif
