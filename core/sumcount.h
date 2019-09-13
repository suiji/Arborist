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

#include <vector>
using namespace std;

/**
   @brief Row sum / count record for categorical indices.
 */
class SumCount {
  double sum;
  unsigned int sCount;

 public:

  SumCount(double sum_,
           unsigned int sCount_) : sum(sum_), sCount(sCount_) {
  }

  SumCount() : sum(0.0), sCount(0) {
  }


  inline auto getSum() const {
    return sum;
  }


  inline auto getSCount() const {
    return sCount;
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
  
  static SumCount minus(const SumCount& minuend,
                        const SumCount& subtrahend) {
    return SumCount(minuend.sum - subtrahend.sum, minuend.sCount - subtrahend.sCount);
  }

  
  /**
     @brief Subtracts contents of vector passed.

     @param subtrahend is the value to subtract.

     @return difference.
   */
  SumCount& operator-=(const SumCount &subtrahend) {
    sum -= subtrahend.sum;
    sCount -= subtrahend.sCount;

    return *this;
  }


  /**
     @brief As above, with addition.

     @return sum.
   */
  SumCount& operator+=(const SumCount& addend) {
    sum += addend.sum;
    sCount += addend.sCount;

    return *this;
  }


  static void decr(vector<SumCount>& minuend,
                   const vector<SumCount>& subtrahend) {
    size_t idx = 0;
    for (auto & sc : minuend) {
      sc -= subtrahend[idx++];
    }
  }

  static void incr(vector<SumCount>& sum,
                   const vector<SumCount>& addend) {
    size_t idx = 0;
    for (auto & sc : sum) {
      sc += addend[idx++];
    }
  }


  static vector<SumCount> minus(const vector<SumCount>& minuend,
                                const vector<SumCount>& subtrahend) {
    vector<SumCount> difference(minuend.size());
    for (size_t idx = 0; idx < difference.size(); idx++) {
      difference[idx] = minus(minuend[idx], subtrahend[idx]);
    }

    return difference;
  }
};

#endif
