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

#ifndef CORE_SAMPLENUX_H
#define CORE_SAMPLENUX_H

#include "typeparam.h"
#include "runnux.h"
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

 protected:
  static unsigned int ctgBits; // Pack:  nonzero iff categorical.
  static unsigned int ctgMask;
  static unsigned int multMask; // Masks bits not used to encode multiplicity.

  static unsigned int rightBits; // # bits to shift for left-most value.
  static unsigned int rightMask; // Mask bits not used by multiplicity, ctg.
  // Integer-sized container is likely overkill:  typically << #rows,
  // although sample weighting might yield run sizes approaching #rows.
  PackedT packed; // Packed sample count, ctg.
  FltVal ySum; // Sum of values selected:  sample-count * y-value.

  
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

     @param sampleCount (>0) is the number of times value sampled.

     @param ctg is the response category, if classification.
  */ 
  SampleNux(FltVal yVal,
            IndexT sampleCount,
            PredictorT ctg = 0) :
    packed((sampleCount << ctgBits) | ctg),
    ySum(yVal * sampleCount) {
  }

  SampleNux(FltVal yVal,
	    IndexT leftVal,
            IndexT sampleCount,
            PredictorT ctg = 0) :
    packed((PackedT(leftVal) << rightBits) | (sampleCount << ctgBits) | ctg),
    ySum(yVal * sampleCount) {
  }

  SampleNux() {
  }


  /**
     @brief Compound accessor.

     @param[out] ctg is category value / default:  classification / regression.

     @return sample sum.
  */
  inline FltVal refCtg(PredictorT& ctg) const {
    ctg = getCtg();
    return ySum;
  }


  /**
     @brief Accessor for packed sCount/ctg member.
   */
  inline auto getRight() const {
    return packed & rightMask;
  }


  /**
     @brief Getter for sampled sum.

     @return ySum value.
   */
  inline double getSum() const {
    return ySum;
  }


  /**
     @brief Derives sample count from internal encoding.

     @return sample count.
   */
  inline IndexT getSCount() const {
    return (packed >> ctgBits) & multMask;
  }


  /**
     @brief Derives response category from internal encouding.

     @return response cardinality.
   */
  inline PredictorT getCtg() const {
    return packed & ctgMask;
  }
};


/**
   @brief Specialized for Sampler input:  row delta value.

   Unless rows are sampled with widely disparate weights, the
   values of 'delRow' are likely to require only a few bits.
 */
struct SampledNux : public SampleNux {
  SampledNux(IndexT delRow,
	     FltVal yVal,
	     IndexT sampleCount,
	     PredictorT ctg = 0) :
    SampleNux(yVal, delRow, sampleCount, ctg) {
  }

  inline auto getDelRow() const {
    return packed >> rightBits;
  }
};


/**
   @brief <Response value, observation rank> pair at row/predictor coordinate.
 */
struct SampleRank : public SampleNux {

  /**
     @brief Getter for rank or factor group.

     @return rank value.
   */
  inline auto getRank() const {
    return packed >> rightBits;
  }


  /**
     @brief Outputs statistics appropriate for regression.

     @return true iff run state changes.
   */
  inline void regInit(RunNux& nux) const {
    nux.code = getRank();
    nux.sum = ySum;
    nux.sCount = getSCount();
  }

  
  /**
     @brief Outputs statistics appropriate for classification.

     @param[out] nux accumulates run statistics.

     @param[in, out] sumBase accumulates run response by category.
   */
  inline void ctgInit(RunNux& nux,
		      double* sumBase) const {
    nux.code = getRank();
    nux.sum = ySum;
    nux.sCount = getSCount();
    sumBase[getCtg()] = ySum;
  }


  /**
     @brief Accumulates statistics for an existing run.

     @return true iff the current cell continues a run.
   */
  inline bool regAccum(RunNux& nux) const {
    if (nux.code == getRank()) {
      nux.sum += ySum;
      nux.sCount += getSCount();
      return true;
    }
    else {
      return false;
    }
  }

  
  /**
     @brief Accumulates statistics for an existing run.

     @return true iff the current cell continues a run.
   */
  inline bool ctgAccum(RunNux& nux,
		       double* sumBase) const {
    if (nux.code == getRank()) {
      nux.sum += ySum;
      nux.sCount += getSCount();
      sumBase[getCtg()] += ySum;
      return true;
    }
    else {
      return false;
    }
  }

  
  /**
     @brief Getter for 'ySum' field

     @return sum of y-values for sample.
   */
  inline auto getYSum() const {
    return ySum;
  }


  /**
     @brief Initializes by copying response and joining sampled rank.

     @param rank is the predictor rank sampled at a given row.

     @param sNode summarizes response sampled at row.
  */
  inline void join(const SampleNux& sNode,
		   IndexT rank) {
    packed = (PackedT(rank) << rightBits) | sNode.getRight();
    ySum = sNode.getSum();
  }


  // These methods should only be called when the response is known
  // to be regression, as it relies on a packed representation specific
  // to that case.
  //

  /**
     @brief Compound accessor for regression.  Cannot be used for
     classification, as 'sCount' value reported here not unpacked.

     @param[out] ySum outputs the response value.

     @param[out] sCount outputs the multiplicity of the row in this sample.

     @return rank of predictor value at sample.
   */
  inline auto regFields(FltVal& ySum,
                        IndexT& sCount) const {
    ySum = this->ySum;
    sCount = getSCount();

    return getRank();
  }


  // These methods should only be called when the response is known
  // to be categorical, as it relies on a packed representation specific
  // to that case.
  //
  /**
     @brief Reports SamplePred contents for categorical response.  Can
     be called with regression response if '_yCtg' value ignored.

     @param[out] ySum is the proxy response value.

     @param[out] yCtg is the response value.

     @return sample count.
   */
  inline auto ctgFields(FltVal &ySum,
                        PredictorT& yCtg) const {
    ySum = this->ySum;
    yCtg = getCtg();

    return getSCount();
  }


  /**
     @brief Compound accessor for classification.  Can be
     called for regression if '_yCtg' value ignored.

     @param[out] ySum_ is the proxy response value.

     @param[out] sCount_ the sample count.

     @param[out] yCtg_ is the true response value.

     @return predictor rank.
   */
  inline auto ctgFields(FltVal& ySum_,
                        IndexT& sCount_,
                        PredictorT& yCtg_) const {
    sCount_ = ctgFields(ySum_, yCtg_);
    return getRank();
  }
};

#endif
