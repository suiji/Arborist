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
   only by method name and not by subtyping.  The only distinction is the
   value (and interpretation) of the 'ctg' value.  Care should be taken
   to call the appropriate method, as 'ctg' is only used as a packing
   parameter (with value zero) in the case of regression.  Subtyping seems
   to complicate the code needlessly, with a per-tree size savings of only
   'nSamp' * sizeof(uint).
 */
class SampleNux {
  static unsigned int nCtg;

 protected:
  static unsigned int ctgShift; // Pack:  nonzero iff categorical.

  // Integer-sized container is likely overkill:  typically << #rows,
  // although sample weighting might yield run sizes approaching #rows.
  unsigned int sCount;
  FltVal ySum; // Sum of values selected:  sCount * y-value.

  
 public:
  static void Immutables(unsigned int ctgWidth);
  static void DeImmutables();

  static inline unsigned int NCtg() {
    return nCtg;
  }
  
  inline double Init(FltVal _yVal, unsigned int _sCount, unsigned int _ctg = 0) {
    ySum = _yVal * _sCount;
    sCount = (_sCount << ctgShift) | _ctg; 
    return ySum;
  }


  /**
     @brief Compound accessor.

     @param _sum outputs sum.

     @return Category value or default:  classification / regression, plus output reference parameters.
   
  */
  inline unsigned int RefLeaf(FltVal &_sum) const {
    _sum = ySum;
    return sCount & ((1 << ctgShift) - 1);
  }


  inline void Ref(FltVal &_ySum, unsigned int &_sCount) const {
    _ySum = ySum;
    _sCount = sCount;
  }


  inline double getSum() const {
    return ySum;
  }
  

  /**
     @brief Accessor for sample count.
     
   */
  inline unsigned int getSCount() const {
    return sCount >> ctgShift;
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
  inline unsigned int Rank() const {
    return rank;
  }


  /**
     @brief Accessor for 'ySum' field

     @return sum of y-values for sample.
   */
  inline FltVal YSum() {
    return ySum;
  }


  /**
     @brief Initializes node by joining sampled rank and response.

     @param _rank is the predictor rank sampled at a given row.

     @param _sNode summarizes response sampled at row.

     @return void.
  */
  inline void Join(unsigned int _rank, const SampleNux &_sNode) {
    rank = _rank;
    _sNode.Ref(ySum, sCount);
  }


  // These methods should only be called when the response is known
  // to be regression, as it relies on a packed representation specific
  // to that case.
  //

  /**
     @brief Reports SamplePred contents for regression response.  Cannot
     be used with categorical response, as 'sCount' value reported here
     is unshifted.

     @param _ySum outputs the response value.

     @param _rank outputs the predictor rank.

     @param _sCount outputs the multiplicity of the row in this sample.

     @return void.
   */
  inline void RegFields(FltVal &_ySum, unsigned int &_rank, unsigned int &_sCount) const {
    _ySum = ySum;
    _rank = rank;
    _sCount = sCount;
  }


  // These methods should only be called when the response is known
  // to be categorical, as it relies on a packed representation specific
  // to that case.
  //
  /**
     @brief Reports SamplePred contents for categorical response.  Can
     be called with regression response if '_yCtg' value ignored.

     @param _ySum outputs the proxy response value.

     @param _rank outputs the predictor rank.

     @param _yCtg outputs the response value.

     @return sample count, with output reference parameters.
   */
  inline unsigned int CtgFields(FltVal &_ySum, unsigned int &_yCtg) const {
    _ySum = ySum;
    _yCtg = sCount & ((1 << ctgShift) - 1);

    return getSCount();
  }


  /**
     @brief Reports SamplePred contents for categorical response.  Can
     be called with regression response if '_yCtg' value ignored.

     @param _ySum outputs the proxy response value.

     @param _rank outputs the predictor rank.

     @param _yCtg outputs the response value.

     @return sample count, with output reference parameters.
   */
  inline unsigned int CtgFields(FltVal &_ySum, unsigned int &_rank, unsigned int &_yCtg) const {
    _rank = rank;
    return CtgFields(_ySum, _yCtg);
  }
};

#endif
