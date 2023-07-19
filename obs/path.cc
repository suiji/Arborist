// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file path.cc

   @brief Methods involving paths from index sets and to individual
   indices.

   @author Mark Seligman
 */

#include <numeric>

#include "frontier.h"
#include "path.h"

IndexT NodePath::noSplit = 0;


void NodePath::setNoSplit(IndexT bagCount) {
  noSplit = bagCount;
}


IdxPath::IdxPath(IndexT idxLive_) :
  idxLive(idxLive_),
  smIdx(vector<IndexT>(idxLive)),
  pathFront(vector<PathT>(idxLive)) {
  iota(smIdx.begin(), smIdx.end(), 0);
}


void NodePath::init(const IndexSet& iSet, IndexT endIdx) {
  frontIdx = iSet.getSplitIdx();
  bufRange = iSet.getBufRange();
  idxStart = endIdx;
}
  

