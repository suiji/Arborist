// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_CRITENCODING_H
#define SPLIT_CRITENCODING_H

/**
   @file critencoding.h

   @brief Minimal information needed to encode splitting criterion.

   @author Mark Seligman
 */

#include "typeparam.h"
#include "sumcount.h"

#include <vector>

enum class EncodingStyle { direct, trueBranch };


/**
   @brief Encapsulates contributions of an individual split to frontier.
 */
struct CritEncoding {
  double sum; // sum of responses over encoding.
  IndexT sCount; // # samples encoded.
  IndexT extent; // # SR indices encoded.
  const class SplitNux& nux;
  vector<SumCount> scCtg; // Response sum decomposed by category.
  const IndexT implicitTrue; // # implicit SR indices.
  const bool increment; // True iff encoding is additive else subtractive.
  const bool exclusive; // True iff update is masked.
  const EncodingStyle style; // Whether direct or true-branch.
  
  CritEncoding(const class SplitFrontier* frontier,
	       const class SplitNux& nux,
	       bool incr = true);


  inline bool trueEncoding() const {
    return implicitTrue == 0;
  };

  /**
     @brief Sample count getter.
   */
  IndexT getSCountTrue() const;


  /**
     @return sum of responses contributing to true branch.
   */
  double getSumTrue() const;


  /**
     @return # SR inidices contributing to true branch.
   */
  IndexT getExtentTrue() const;
  

  /**
     @brief Accumulates encoding statistics for a single SR index.
   */
  inline void accum(double ySum,
		    IndexT sCount,
		    PredictorT ctg) {
    this->sum += ySum;
    this->sCount += sCount;
    extent++;
    if (!scCtg.empty()) {
      scCtg[ctg] += SumCount(ySum, sCount);
    }
  }


  void getISetVals(IndexT& sCountTrue,
		   double& sumTrue,
		   IndexT& extentTrue,
		   bool& encodeTrue,
		   double& minInfo) const;


  void branchUpdate(const class SplitFrontier* sf,
		    const IndexRange& range,
		    class BranchSense& branchSense);


  void branchUpdate(const class ObsPart* obsPart,
		    const IndexRange& range,
		    class BranchSense& branchSense);


private:  
  void branchSet(IndexT* sIdx,
		 class Obs* spn,
		 const IndexRange& range,
		 class BranchSense& branchSense);


  void branchUnset(IndexT* sIdx,
		   class Obs* spn,
		   const IndexRange& range,
		   class BranchSense& branchSense);


  void encode(const class Obs& obs);

  
  /**
     @brief Outputs the internal contents.
   */
  void accumDirect(IndexT& sCountTrue,
		   double& sumTrue,
		   IndexT& extentTrue) const;


  /**
     @brief Outputs the contributions to the true branch.
   */
  void accumTrue(IndexT& sCountTrue,
		 double& sumTrue,
		 IndexT& extentTrue) const ;
};

#endif
