// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file crit.h

   @brief Class defintion for generic splitting criteria.

   @author Mark Seligman
 */

#ifndef SPLIT_CRIT_H
#define SPLIT_CRIT_H

#include "typeparam.h"


/**
   @brief Untagged union of split encodings; fields keyed by predictor type.

   Numerical splits begin as rank ranges and are later adjusted to double.
   Factor splits are tree-relative offsets.

   Reading and writing requires context from containing node.
 */
typedef union EncodingU {
  IndexT leafIdx; ///< Terminals only.
  double num; ///< Rank-derived splitting value:  quantile or cut.
  size_t offset; ///< Tree-relative bit-vector offset:  factor.

  double getNum() const {
    return num;
  }
    
  void setNum(double num) {
    this->num = num;
  }

  size_t getOffset() const {
    return offset;
  }


  void setOffset(size_t offset) {
    this->offset = offset;
  }


  IndexT getLeafIdx() const {
    return leafIdx;
  }


  void setLeafIdx(IndexT leafIdx) {
    this->leafIdx = leafIdx;
  }
} SplitValU;


/** 
   @brief Encodes integer values as doubles.

   This limits the range to 52 bits, but enables context-free reading and writing.
*/
struct SplitValD {
  double dVal; // Rank-derived splitting value:  quantile or cut.

  SplitValD(double val) : dVal(val) {
  }


  double getVal() const {
    return dVal;
  }

  
  void setNum(double num) {
    dVal = num;
  }

  
  double getNum() const {
    return dVal;
  }

  
  size_t getOffset() const {
    return dVal;
  }

  
  void setOffset(size_t offset) {
    dVal = offset;
  }

  
  IndexT getLeafIdx() const {
    return dVal;
  }


  void setLeafIdx(IndexT leafIdx) {
    dVal = leafIdx;
  }
};


/**
   @brief Splitting criterion.

   Branch sense implicitly less-than-equal left.
 */
struct Crit {
  SplitValD val;

  Crit(double crit) : val(crit) {
  }


  double getVal() const {
    return val.getVal();
  }

  
  void critCut(const class SplitNux& nux,
	       const class SplitFrontier* splitFrontier);


  void critBits(size_t bitPos) {
    val.setOffset(bitPos);
  }
  
  
  void setNum(double num) {
    val.setNum(num);
  }


  double getNumVal() const {
    return val.getNum();
  }

  
  size_t getBitOffset() const {
    return val.getOffset();
  }


  IndexT getLeafIdx() const {
    return val.getLeafIdx();
  }


  void setLeafIdx(IndexT leafIdx) {
    val.setLeafIdx(leafIdx);
  }
  
  
  void setQuantRank(const class PredictorFrame* predictor,
		    PredictorT predIdx);
};


#endif

