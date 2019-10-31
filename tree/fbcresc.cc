// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file fboresc.cc

   @brief Methods for growing the crescent factor blocks.

   @author Mark Seligman
 */

#include "fbcresc.h"
#include "bv.h"


FBCresc::FBCresc(unsigned int treeChunk) :
  fac(vector<unsigned int>(0)),
  height(vector<size_t>(treeChunk)) {
}


void FBCresc::treeCap(unsigned int tIdx) {
  height[tIdx] = fac.size();
}


void FBCresc::dumpRaw(unsigned char facRaw[]) const {
  for (size_t i = 0; i < fac.size() * sizeof(unsigned int); i++) {
    facRaw[i] = ((unsigned char*) &fac[0])[i];
  }
}


void FBCresc::appendBits(const BV* splitBits,
                         size_t bitEnd,
                         unsigned int tIdx) {
  splitBits->consume(fac, bitEnd);
  treeCap(tIdx);
}
