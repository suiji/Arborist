// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sumcount.h

   @brief Class definition sample sum and count item.

   @author Mark Seligman
 */

#ifndef CORE_SUMCOUNT_H
#define CORE_SUMCOUNT_H

/**
   @brief Row sum / count record for categorical indices.
 */
class SumCount {
  double sum;
  unsigned int sCount;

 public:
  void init() {
    sum = 0.0;
    sCount = 0;
  }


  /**
     @brief Determines whether a node is splitable and accesses sum field.

     @param sCount is the containing node's sample count.

     @param[out] sum_ outputs the sum at this category.

     @return true iff not all samples belong to this category.
   */
  inline bool splitable(unsigned int sCount, double& sum) const {
    sum = this->sum;
    return sCount != this->sCount;
  }
  

  /**
     @brief Accumulates running sum and sample-count values.

     @param[in, out] _sum accumulates response sum over sampled indices.

     @param[in, out] _sCount accumulates sample count.
   */
  inline void accum(double sum, unsigned int sCount) {
    this->sum += sum;
    this->sCount += sCount;
  }


  /**
     @brief Subtracts contents of vector passed.

     @param subtrahend is the value to subtract.
   */
  void decr(const SumCount &subtrahend) {
    sum -= subtrahend.sum;
    sCount -= subtrahend.sCount;
  }
};

#endif
