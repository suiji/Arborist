// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file blockframe.h

   @brief Data frame representations built from type-parametrized blocks.

   @author Mark Seligman
 */

#ifndef FRAMEMAP_BLOCKFRAME_H
#define FRAMEMAP_BLOCKFRAME_H

#include "block.h"

/**
   @brief Frame represented as collections of simply-typed blocks.

   Currently implemented as numeric and factor only, but may potentially
   support arbitrary collections.
 */

class BlockFrame {
  BlockWindow<double>* blockNum;
  BlockWindow<unsigned int>* blockFac;
  const unsigned int nRow;

public:
  BlockFrame(BlockWindow<double>* blockNum_,
             BlockWindow<unsigned int>* blockFac,
             unsigned nRow_);

  /**
     @brief Accessor for row count.
   */
  inline auto getNRow() const {
    return nRow;
  }
  
  /**
     @brief Assumes numerical predictors packed in front of factor-valued.

     @return Position of fist factor-valued predictor.
  */
  inline unsigned int getNPredFac() const {
    return blockFac->getNCol();
  }


  /**
     @brief Assumes numerical predictors packed in front of factor-valued.

     @return Position of fist factor-valued predictor.
  */
  inline unsigned int getNPredNum() const {
    return blockNum->getNCol();
  }


  inline bool isFactor(unsigned int predIdx) const {
    return predIdx >= getNPredNum();
  }
  
  /**
     @brief Computes block-relative position for a predictor.

     @param[out] thisIsFactor outputs true iff predictor is factor-valued.

     @return block-relative index.
   */
  inline unsigned int getIdx(unsigned int predIdx, bool &predIsFactor) const{
    predIsFactor = isFactor(predIdx);
    return predIsFactor ? predIdx - getNPredNum() : predIdx;
  }


  /**
     @brief Updates windowing state on respective blocks.
   */
  void reWindow(unsigned int rowStart,
                unsigned int rowEnd,
                unsigned int rowBlock) const;

  /**
     @return base address for (transposed) numeric values at row.
   */
  const double* baseNum(unsigned int rowOff) const;


  /**
     @return base address for (transposed) factor values at row.
   */
  const unsigned int* baseFac(unsigned int rowOff) const;
};

#endif
