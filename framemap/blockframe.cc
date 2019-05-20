// This file is part of framemap

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file blockframe.cc

   @brief Methods for blocks of similarly-typed predictors.

   @author Mark Seligman
 */

#include "blockframe.h"

BlockFrame::BlockFrame(BlockWindow<double>* blockNum_,
                       BlockWindow<unsigned int>* blockFac_,
                       unsigned int nRow_) :
  blockNum(blockNum_),
  blockFac(blockFac_),
  nRow(nRow_) {
}

void BlockFrame::reWindow(unsigned int rowStart,
                         unsigned int rowEnd,
                         unsigned int rowBlock) const {
  blockNum->reWindow(rowStart, rowEnd, rowBlock);
  blockFac->reWindow(rowStart, rowEnd, rowBlock);
}


/**
   @brief Computes pointer to base of row of numeric values.

   @param rowOff is a block-relative row offset.

   @return base address for numeric values at row.
*/
const double* BlockFrame::baseNum(unsigned int rowOff) const {
  return blockNum->rowBase(rowOff);
}


/**
   @brief Computes pointer to base of row of factor values.

   @param rowOff is a block-relative row offset.

   @return base address for factor values at row.
*/
const unsigned int* BlockFrame::baseFac(unsigned int rowOff) const {
  return blockFac->rowBase(rowOff);
}
