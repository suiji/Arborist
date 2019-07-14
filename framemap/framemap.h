// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file framemap.h

   @brief Data frame representations built from type-parametrized blocks.

   @author Mark Seligman
 */

#ifndef ARBORIST_FRAMEMAP_H
#define ARBORIST_FRAMEMAP_H

#include <vector>
#include <algorithm> // max()
using namespace std;


/**
   @brief Summarizes frame contents by predictor type.
 */
class FrameMap {
  unsigned int nRow;
  const vector<unsigned int> &feCard; // Factor predictor cardinalities.
  unsigned int nPredFac;
  unsigned int nPredNum;

  // Greatest cardinality extent, irrespective of gaps.  Useful for packing.
  const unsigned int cardExtent;

 public:

  FrameMap(const vector<unsigned int> &feCard_,
           unsigned int nPred,
           unsigned int nRow_) :
    nRow(nRow_),
    feCard(feCard_),
    nPredFac(feCard.size()),
    nPredNum(nPred - feCard.size()),
    cardExtent(nPredFac > 0 ? *max_element(feCard.begin(), feCard.end()) : 0) {
  }

  
  /**
     @brief Assumes numerical predictors packed in front of factor-valued.

     @return Position of fist factor-valued predictor.
  */
  inline unsigned int getFacFirst() const {
    return nPredNum;
  }

  
  /**
     @brief Determines whether predictor is numeric or factor.

     @param predIdx is internal predictor index.

     @return true iff index references a factor.
   */
  inline bool isFactor(unsigned int predIdx)  const {
    return predIdx >= getFacFirst();
  }


  /**
     @brief Computes block-relative position for a predictor.

     @param[out] thisIsFactor outputs true iff predictor is factor-valued.

     @return block-relative index.
  */
  inline unsigned int getIdx(unsigned int predIdx, bool &thisIsFactor) const{
    thisIsFactor = isFactor(predIdx);
    return thisIsFactor ? predIdx - getFacFirst() : predIdx;
  }


  /**
     @return number or observation rows.
   */
  inline unsigned int getNRow() const {
    return nRow;
  }

  /**
     @return number of observation predictors.
  */
  inline unsigned int getNPred() const {
    return nPredFac + nPredNum;
  }

  /**
     @return number of factor predictors.
   */
  inline unsigned int getNPredFac() const {
    return nPredFac;
  }

  /**
     @return number of numerical predictors.
   */
  inline unsigned int getNPredNum() const {
    return nPredNum;
  }


  /**
     @brief Fixes contiguous factor ordering as numerical preceding factor.

     @return Position of first numerical predictor.
  */
  static unsigned int constexpr getNumFirst() {
    return 0ul;
  }


  /**
     @brief Positions predictor within numerical block.

     @param predIdx is the core-ordered index of a predictor assumed to be numeric.

     @return Position of predictor within numerical block.
  */
  inline unsigned int getNumIdx(unsigned int predIdx) const {
    return predIdx - getNumFirst();
  }


  /**
     @brief Computes cardinality of factor-valued predictor, or zero if not a
     factor.

     @param predIdx is the internal predictor index.

     @return factor cardinality or zero.
  */
  inline auto getFacCard(unsigned int predIdx) const {
    return isFactor(predIdx) ? feCard[predIdx - getFacFirst()] : 0;
  }

  
  /**
     @brief Accessor for greatest cardinality extent.
  */
  inline auto getCardExtent() const {
    return cardExtent;
  }
};

#endif
