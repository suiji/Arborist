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


/**
   @brief Compact representation for splitting.
 */
class Obs {
  static const unsigned int tieMask = 1ul; ///< Mask bit for tie encoding.
  static const unsigned int ctgLow = 1ul; ///< Low bit position of ctg.
  static IndexT maxSCount;
  static unsigned int ctgMask;
  static unsigned int multLow; ///< Low bit position of multiplicity.
  static unsigned int multMask; ///< Masks bits not encoding multiplicity.
  static double scale; ///< Coefficent scaling response to < 0.5.
  static double recipScale; ///< Reciprocal simplifies division.
  
  FltVal obsPacked; ///< Packed representation.


  /**
     @brief Totals a range of observations into regression format.
   */
  static ObsReg regTotal(const Obs* obsStart,
			 IndexT extent) {
    double ySum = 0.0;
    IndexT sCount = 0;
    for (const Obs* obs = obsStart; obs != obsStart + extent; obs++) {
      FltVal obsPacked = obs->obsPacked;
      unsigned int rounded = round(obsPacked);
      ySum += scale * (obsPacked - rounded);
      sCount += 1 + ((rounded >> multLow) & multMask);
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
      FltVal obsPacked = obs->obsPacked;
      unsigned int rounded = round(obsPacked);
      double ySumThis = scale * (obsPacked - rounded);
      PredictorT yCtg = (rounded >> ctgLow) & ctgMask;
      ctgImpl[yCtg] -= ySumThis;
      ySumExpl += ySumThis;
      sCountExpl += 1 + ((rounded >> multLow) & multMask);
    }
    sum -= ySumExpl;
    sCount -= sCountExpl;
  }

  
 public:

  /**
     @brief Unpacks a single observation into regression format.
   */
  inline ObsReg unpackReg() const {
    unsigned int rounded = round(obsPacked); // Rounds nearest.
    return ObsReg(scale * (obsPacked - rounded),
		  1 + ((rounded >> multLow) & multMask),
		  (rounded & tieMask) != 0);
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
    unsigned int rounded = round(obsPacked); // Rounds nearest.
    sCount = 1 + ((rounded >> multLow) & multMask);
    ySum = scale * (obsPacked - rounded);
  }


  /**
     @brief Unpacks float into categorical representation.

     The fractional component of obsPacked is a scaled class weight, and
     is therefore positive, so truncation (round-toward-zero) may be
     used instead of a slower call to round().

     Class weights are proportional, so it may be possible to avoid
     descaling.
   */
  inline void refCtg(unsigned int& sCount,
		     double& ySum,
		     unsigned int& yCtg) const {
    unsigned int rounded = obsPacked;  // Rounds toward zero.
    sCount = 1 + ((rounded >> multLow) & multMask);
    ySum = scale * (obsPacked - rounded);
    yCtg = (rounded >> ctgLow) & ctgMask;
  }

  inline ObsCtg unpackCtg() const {
    unsigned int rounded = obsPacked;  // Rounds toward zero.
    return ObsCtg((rounded >> ctgLow) & ctgMask,
		  scale * (obsPacked - rounded),
		  1 + ((rounded >> multLow) & multMask),
		  (rounded & tieMask) != 0);
  }

  /**
     @brief Sets internal packing parameters.
   */
  static void setShifts(IndexT maxSCount,
			unsigned int ctgBits_,
		        unsigned int ctgMask_);


  static void setScale(double yMax);

  
  static void deImmutables();


  bool isTied() const {
    unsigned int rounded = round(obsPacked);
    return (rounded & tieMask) != 0;
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
    obsPacked = sNux.getYSum() * recipScale + ((sNux.getSCount()-1) << multLow) + (sNux.getCtg() << ctgLow) + (tie ? 1 : 0); 
  }


  void setTie(bool tie) {
    unsigned int rounded = round(obsPacked);
    if (rounded & 1) {
      if (!tie)
	obsPacked--;
    }
    else {
      if (tie)
	obsPacked++;
    }
  }
  

  /**
     @brief Derives sample count from internal encoding.

     @return sample count.
   */
  inline IndexT getSCount() const {
    unsigned int rounded = round(obsPacked);
    return 1 + ((rounded >> multLow) & multMask);
  }


  /**
     @brief Produces sum of y-values over sample.

     @return sum of y-values for sample.
   */
  inline double getYSum() const {
    return scale * (obsPacked - round(obsPacked));
  }


  /**
     @brief Derives response category from internal encouding.

     @return response cardinality.
   */
  inline PredictorT getCtg() const {
    unsigned int rounded = round(obsPacked);
    return (rounded >> ctgLow) & ctgMask;
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
