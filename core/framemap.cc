// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file framemap.cc

   @brief Methods for blocks of similarly-typed predictors.

   @author Mark Seligman
 */

#include <algorithm>

#include "framemap.h"
#include "block.h"

/**
   @brief

   @return void.
 */
FrameTrain::FrameTrain(const vector<unsigned int> &_feCard,
		 unsigned int _nPred,
		 unsigned int _nRow) :
  FrameMap(_nRow, _nPred - _feCard.size(), _feCard.size()),
  feCard(_feCard),
  cardMax(nPredFac > 0 ? *max_element(feCard.begin(), feCard.end()) : 0) {
}


/**
   @brief Static initialization for prediction.

   @return void.
 */
FramePredict::FramePredict(BlockNum *blockNum_,
			   BlockFac *blockFac_,
			   unsigned int nRow_) :
  FrameMap(nRow_, blockNum_->getNCol(), blockFac_->getNCol()),
  blockNum(blockNum_),
  blockFac(blockFac_) {
}


FramePredict::~FramePredict() {
}

void FramePredict::transpose(unsigned int rowStart,
                                  unsigned int rowEnd) const {
  blockNum->transpose(rowStart, rowEnd);
  blockFac->transpose(rowStart, rowEnd);
}


  /**
     @return base address for (transposed) numeric values at row.
   */
const double *FramePredict::baseNum(unsigned int rowOff) const {
  return blockNum->rowBase(rowOff);
}


  /**
     @return base address for (transposed) factor values at row.
   */
const unsigned int *FramePredict::baseFac(unsigned int rowOff) const {
  return blockFac->rowBase(rowOff);
}
