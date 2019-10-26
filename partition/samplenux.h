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

  
  /**
     @brief Initializes to summary values passed.

     @param yVal is reponse value.

     @param sampleCount (>0) is the number of times value sampled.

     @param ctg is the response category, if classification.
  */ 
  SampleNux(FltVal yVal,
            unsigned int sampleCount,
            unsigned int ctg = 0) :
    sCount((sampleCount << ctgShift) | ctg),
    ySum(yVal * sampleCount) {
  }

  SampleNux() {
  }


  /**
     @brief Compound accessor.

     @param[out] ctg is category value / default:  classification / regression.

     @return sample sum.
  */
  inline FltVal refCtg(unsigned int &ctg) const {
    ctg = getCtg();
    return ySum;
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
  inline unsigned int getSCount() const {
    return sCount >> ctgShift;
  }


  /**
     @brief Derives response category from internal encouding.

     @return response category.
   */
  inline unsigned int getCtg() const {
    return sCount & ((1 << ctgShift) - 1);
  }
};


/**
   @brief <Response value, observation rank> pair at row/predictor coordinate.
 */
class SampleRank : public SampleNux {
 protected:
  IndexT rank; // Rank, up to tie, or factor group.

 public:

  /**
     @brief Getter for rank or factor group.

     @return rank value.
   */
  inline auto getRank() const {
    return rank;
  }


  /**
     @brief Getter for 'ySum' field

     @return sum of y-values for sample.
   */
  inline auto getYSum() const {
    return ySum;
  }


  /**
     @brief Initializes node by joining sampled rank and response.

     @param rank is the predictor rank sampled at a given row.

     @param sNode summarizes response sampled at row.
  */
  inline void join(IndexT rank,
		   const SampleNux &sNode) {
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
  inline auto ctgFields(FltVal &ySum,
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
  inline auto ctgFields(FltVal &ySum_,
                        unsigned int &sCount_,
                        unsigned int &yCtg_) const {
    sCount_ = ctgFields(ySum_, yCtg_);
    return rank;
  }

  
  /**
     @brief Accumulates this cell's contents in per-category vector.

     @param[in, out] ctgExpl accumulates sample counts and values by category.

     @return ySum value.
   */
  FltVal accum(vector<class SumCount>& ctgExpl) const;
};

#endif
