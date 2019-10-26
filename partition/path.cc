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

#include "path.h"


IdxPath::IdxPath(IndexT idxLive_) :
  idxLive(idxLive_),
  relFront(vector<unsigned int>(idxLive)),
  pathFront(vector<unsigned char>(idxLive)),
  offFront(vector<uint_least16_t>(idxLive)) {
  iota(relFront.begin(), relFront.end(), 0);
}
