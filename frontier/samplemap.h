// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplemap.h

   @brief Partitions samples along tree frontier.

   @author Mark Seligman

 */

#ifndef FRONTIER_SAMPLEMAP_H
#define FRONTIER_SAMPLEMAP_H

#include "typeparam.h"

#include <algorithm>
#include <vector>


/**
   @brief Maps to and from sample indices and tree nodes.

   Easy access to node contents simplifies the task of tracking sample histories.
   Nonterminal scores provide a prediction for premature termination, as in the
   case of missing observations.

   Nonterminal components are maintained via a double-buffer scheme, updated
   following splitting.  The update performs a stable partition to improve
   locality.  A buffer initially lists all sample indices, but continues to
   shrink as terminal nodes absorb the contents.  The terminal component is
   initially empty, but continues to grow as nonterminal contents are absorbed.

   Index assignments become sparser as training proceeds, although stable
   partitioning preserves a monotone-increasing order.  Attempts to dereference
   the indices within a node will therefore incur increasingly irregular
   accesses.  This problem can be largely overcome by looping over the nodes in
   parallel, which exhibits excellent strong scaling - likely due to
   opportunites for line reuse across the nodes.

   Extent vectors record the number of sample indices associated with each node.
 */
struct SampleMap {
  vector<IndexT> sampleIndex;
  vector<IndexRange> range;
  vector<IndexT> ptIdx;

  /**
     @brief Constructor with optional index count.
   */
  SampleMap(IndexT nIdx = 0) :
    sampleIndex(vector<IndexT>(nIdx)),
    range(vector<IndexRange>(0)),
    ptIdx(vector<IndexT>(0)) {
  }


  IndexT getEndIdx() const {
    return range.empty() ? 0 : range.back().getEnd();
  }


  void addNode(IndexT extent,
	       IndexT ptId) {
    range.emplace_back(getEndIdx(), extent);
    ptIdx.push_back(ptId);
  }

  
  IndexT* getWriteStart(IndexT idx) {
    return &sampleIndex[range[idx].getStart()];
  }
  
  
  IndexT getNodeCount() const {
    return range.size();
  }

};


#endif
