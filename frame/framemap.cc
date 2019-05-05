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

FrameMap::FrameMap(const vector<unsigned int> &feCard_,
                   unsigned int nPred,
                   unsigned int nRow_) :
  nRow(nRow_),
  feCard(feCard_),
  nPredFac(feCard.size()),
  nPredNum(nPred - feCard.size()),
  cardMax(nPredFac > 0 ? *max_element(feCard.begin(), feCard.end()) : 0) {
}
