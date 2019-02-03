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

#ifndef ARBORIST_SAMPLENUX_H
#define ARBORIST_SAMPLENUX_H

#include "typeparam.h"


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
  static unsigned int nCtg; // Number of categories; 0 for regression.

 protected:
  static unsigned int ctgShift; // Pack:  nonzero iff categorical.

  // Integer-sized container is likely overkill:  typically << #rows,
  // although sample weighting might yield run sizes approaching #rows.
  unsigned int sCount;
  FltVal ySum; // Sum of values selected:  sCount * y-value.

  
 public:

  /**
     @brief Computes a packing width sufficient to hold all (zero-based) response category values.

     @param ctgWidth is the response cardinality.

     @return void.
  */
  static void immutables(unsigned int ctgWidth);
  
  /**
    @brief Resets to static initialization.
  */
  static void deImmutables();

  /**
     @brief Accessor for number of response training categories.
   */
  static inline unsigned int getNCtg() {
    return nCtg;
  }
  
  inline double init(FltVal yVal,
                     unsigned int sampleCount,
                     unsigned int ctg = 0) {
    ySum = yVal * sampleCount;
    sCount = (sampleCount << ctgShift) | ctg; 
    return ySum;
  }


  /**
     @brief Compound accessor.

     @param[out] sum is the sample sum.

     @return Category value or default:  classification / regression.
  */
  inline unsigned int refLeaf(FltVal &sum) const {
    sum = ySum;
    return getCtg();
  }


  /**
     @brief Compound accessor for sampled sum and count.

     @param[out] ySum is the sampled sum.

     @param[out] sCount is the sampled count.

     @return void.
   */
  inline void ref(FltVal &ySum, unsigned int &sCount) const {
    ySum = this->ySum;
    sCount = this->sCount;
  }


  /**
     @brief Accessor for sampled sum.
   */
  inline double getSum() const {
    return ySum;
  }
  

  /**
     @brief Accessor for sample count.
   */
  inline unsigned int getSCount() const {
    return sCount >> ctgShift;
  }


  /**
     @brief Accessor for response category.
   */
  inline unsigned int getCtg() const {
    return sCount & ((1 << ctgShift) - 1);
  }
};


/**
 */
class SampleRank : public SampleNux {
 protected:
  unsigned int rank; // Rank, up to tie, or factor group.

 public:

  /**
     @brief Accessor for 'rank' field

     @return rank value.
   */
  inline unsigned int getRank() const {
    return rank;
  }


  /**
     @brief Accessor for 'ySum' field

     @return sum of y-values for sample.
   */
  inline FltVal getYSum() const {
    return ySum;
  }


  /**
     @brief Initializes node by joining sampled rank and response.

     @param rank is the predictor rank sampled at a given row.

     @param sNode summarizes response sampled at row.

     @return void.
  */
  inline void join(unsigned int rank, const SampleNux &sNode) {
    this->rank = rank;
    sNode.ref(ySum, sCount);
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
  inline auto regFields(FltVal &ySum,
                        unsigned int &sCount) const {
    ySum = this->ySum;
    sCount = this->sCount;

    return rank;
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
  inline unsigned int ctgFields(FltVal &ySum,
                                unsigned int &yCtg) const {
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
  inline unsigned int ctgFields(FltVal &ySum_,
                                unsigned int &sCount_,
                                unsigned int &yCtg_) const {
    sCount_ = ctgFields(ySum_, yCtg_);
    return rank;
  }
};

#endif