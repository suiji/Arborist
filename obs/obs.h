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

#include <cmath>


struct ObsReg {
  double ySum;
  unsigned int sCount;
  bool tied;

  ObsReg(double ySum_,
	 unsigned int sCount_,
	 bool tied_ = false) :
    ySum(ySum_),
    sCount(sCount_),
    tied(tied_) {
  }
};


struct ObsCtg {
  PredictorT yCtg;
  double ySum;
  unsigned int sCount;
  bool tied;

  ObsCtg(unsigned int yCtg_,
	 double ySum_,
	 unsigned int sCount_,
	 bool tied_) :
    yCtg(yCtg_),
    ySum(ySum_),
    sCount(sCount_),
    tied(tied_) {
  }
};


union ObsPacked {
  float num;
  uint32_t bits;
};


/**
   @brief Compact representation for splitting.
 */
class Obs {
  static const unsigned int tieMask = 1ul; ///< Mask bit for tie encoding.
  static const unsigned int ctgLow = 1ul; ///< Low bit position of ctg.
  static unsigned int ctgMask;
  static unsigned int multLow; ///< Low bit position of multiplicity.
  static unsigned int multMask; ///< Masks bits not encoding multiplicity.
  static unsigned int numMask; ///< Masks bits not encoding numeric.
  
  ObsPacked obsPacked;

  /**
     @brief Totals a range of observations into regression format.
   */
  static ObsReg regTotal(const Obs* obsStart,
			 IndexT extent) {
    double ySum = 0.0;
    IndexT sCount = 0;
    for (const Obs* obs = obsStart; obs != obsStart + extent; obs++) {
      ObsPacked fltPacked = obs->obsPacked;
      fltPacked.bits &= numMask;
      ySum += fltPacked.num;
      sCount += 1 + ((obs->obsPacked.bits >> multLow) & multMask);
    }

    return ObsReg(ySum, sCount);
  }

  
  static void ctgResidual(const Obs* obsStart,
			  IndexT extent,
			  double& sum,
			  IndexT& sCount,
			  double ctgImpl[]) {
    double ySumExpl = 0.0;
    IndexT sCountExpl = 0;
    for (const Obs* obs = obsStart; obs != obsStart + extent; obs++) {
      ObsPacked obsPacked = obs->obsPacked;
      double ySumThis = obsPacked.num;
      PredictorT yCtg = (obsPacked.bits >> ctgLow) & ctgMask;
      ctgImpl[yCtg] -= ySumThis;
      ySumExpl += ySumThis;
      sCountExpl += 1 + ((obsPacked.bits >> multLow) & multMask);
    }
    sum -= ySumExpl;
    sCount -= sCountExpl;
  }

  
 public:

  /**
     @brief Unpacks a single observation into regression format.
   */
  inline ObsReg unpackReg() const {
    ObsPacked fltPacked = obsPacked;
    fltPacked.bits &= numMask;
    return ObsReg(fltPacked.num,
		  1 + ((obsPacked.bits >> multLow) & multMask),
		  (obsPacked.bits & tieMask) != 0);
  }


  static ObsReg residualReg(const Obs* obsCell,
			    const class SplitNux* nux);

  /**
     @brief Subtracts explicit sum and count values from node totals.
   */
  static void residualCtg(const Obs* obsCell,
			  const class SplitNux* nux,
			  double& sum,
			  IndexT& sCount,
			  vector<double>& ctgImpl);


  /**
     @brief Unpacks float into numerical representation.
   */
  inline void refReg(unsigned int& sCount,
		     double& ySum) const {
    ObsPacked fltPacked = obsPacked;
    fltPacked.bits &= numMask;
    sCount = 1 + ((obsPacked.bits >> multLow) & multMask);
    ySum = fltPacked.num;
  }


  /**
     @brief Unpacks float into categorical representation.

     Class weights are proportional, so it may be possible to avoid
     descaling.
   */
  inline void refCtg(unsigned int& sCount,
		     double& ySum,
		     unsigned int& yCtg) const {
    sCount = 1 + ((obsPacked.bits >> multLow) & multMask);
    ObsPacked obsNum = obsPacked;
    obsNum.bits &= numMask;
    ySum = obsNum.num;
    yCtg = (obsPacked.bits >> ctgLow) & ctgMask;
  }

  inline ObsCtg unpackCtg() const {
    ObsPacked obsNum = obsPacked;
    obsNum.bits &= numMask;
    return ObsCtg((obsPacked.bits >> ctgLow) & ctgMask,
		  obsNum.num,
		  1 + ((obsPacked.bits >> multLow) & multMask),
		  (obsPacked.bits & tieMask) != 0);
  }

  /**
     @brief Sets internal packing parameters.
   */
  static void setShifts(unsigned int ctgBits_,
		        unsigned int ctgMask_);

  
  static void deImmutables();


  bool isTied() const {
    return (obsPacked.bits & tieMask) != 0;
  }


  /**
     @brief Initializes by copying response and joining sampled rank.

     Rank is only used to break ties and elaborate argmax summaries.
     It may be possible to exclude them.

     @param sNux summarizes response sampled at row.

     @param rank_ is the predictor rank sampled at a given row.

     @param tie indicates whether previous obs has same rank.
  */
  inline void join(const SampleNux& sNux,
		   bool tie) {
    ObsPacked fltPacked;
    fltPacked.num = sNux.getYSum();
    obsPacked.bits = (fltPacked.bits & numMask) + ((sNux.getSCount()-1) << multLow) + (sNux.getCtg() << ctgLow) + (tie ? 1 : 0); 
  }


  void setTie(bool tie) {
    if (tie)
      obsPacked.bits |= 1ul;
    else
      obsPacked.bits &= ~1ul;
  }
  

  /**
     @brief Derives sample count from internal encoding.

     @return sample count.
   */
  inline IndexT getSCount() const {
    return 1 + ((obsPacked.bits >> multLow) & multMask);
  }


  /**
     @brief Produces sum of y-values over sample.

     @return sum of y-values for sample.
   */
  inline double getYSum() const {
    ObsPacked fltPacked = obsPacked;
    fltPacked.bits &= numMask;
    return fltPacked.num;
  }


  /**
     @brief Derives response category from internal encouding.

     @return response cardinality.
   */
  inline PredictorT getCtg() const {
    return (obsPacked.bits >> ctgLow) & ctgMask;
  }


  /**
     @brief Outputs statistics appropriate for regression.

     @return true iff run state changes.
   */
  inline void regInit(RunNux& nux,
		      IndexT code) const {
    nux.setCode(code);
    refReg(nux.sCount, nux.sum);
  }

  
  /**
     @brief Accumulates statistics for an existing run.

     @return true iff the current cell continues a run.
   */
  inline bool regAccum(RunNux& nux) const {
    if (isTied()) {
      IndexT sCount;
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
     @brief Outputs statistics appropriate for classification.

     @param[out] nux accumulates run statistics.

     @param[in, out] sumBase accumulates run response by category.
   */
  inline void ctgInit(RunNux& nux,
		      IndexT code,
		      double* sumBase) const {
    IndexT sCount;
    PredictorT yCtg;
    double ySum;
    refCtg(sCount, ySum, yCtg);
      
    nux.setCode(code);
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
    if (isTied()) {
      IndexT sCount;
      PredictorT yCtg;
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
};

#endif
