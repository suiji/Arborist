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

#include "path.h"
#include <numeric>


IdxPath::IdxPath(unsigned int _idxLive) : idxLive(_idxLive), relFront(std::vector<unsigned int>(idxLive)), pathFront(std::vector<unsigned char>(idxLive)), offFront(std::vector<uint_least16_t>(idxLive)) {
  std::iota(relFront.begin(), relFront.end(), 0);
}


/**
   @brief Localizes copies of the paths to each index position.  Also
   localizes index positions themselves, if in a node-relative regime.

   @param reachBase is non-null iff index offsets enter as node relative.

   @param idxUpdate is true iff the index is to be updated.

   @param startIdx is the beginning index of the cell.

   @param extent is the count of indices in the cell.

   @param pathMask mask the relevant bits of the path value.

   @param idxVec inputs the index offsets, relative either to the
   current subtree or the containing node and may output an updated
   value.

   @param prePath outputs the (masked) path reaching the current index.

   @param pathCount enumerates the number of times a path is hit.  Only
   client is currently dense packing.

   @return void.
 */
void IdxPath::Prepath(const unsigned int reachBase[], bool idxUpdate, unsigned int startIdx, unsigned int extent, unsigned int pathMask, unsigned int idxVec[], PathT prepath[], unsigned int pathCount[]) const {
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int path = IdxUpdate(idxVec[idx], pathMask, reachBase, idxUpdate);
    prepath[idx] = path;
    if (path != NodePath::noPath) {
      pathCount[path]++;
    }
  }
}
