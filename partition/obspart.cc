// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file obspart.cc

   @brief Methods to repartition observation frame by tree node.

   @author Mark Seligman
 */

#include "obspart.h"
#include "layout.h"
#include "splitfrontier.h"
#include "frontier.h"
#include "splitnux.h"
#include "path.h"
#include "branchsense.h"
#include "ompthread.h"

#include <numeric>


/**
   @brief Base class constructor.
 */
ObsPart::ObsPart(const Layout* layout,
		 IndexT bagCount_) :
  bagCount(bagCount_),
  bufferSize(layout->getSafeSize(bagCount)),
  pathIdx(bufferSize),
  stageRange(layout->getNPred()) {
  indexBase = new IndexT[2* bufferSize];
  nodeVec = new SampleRank[2 * bufferSize];

  // Coprocessor variants:
  destRestage = new unsigned int[bufferSize];
  //destSplit = new unsigned int[bufferSize];
}


/**
  @brief Base class destructor.
 */
ObsPart::~ObsPart() {
  delete [] nodeVec;
  delete [] indexBase;

  delete [] destRestage;
  //delete [] destSplit;
}


IndexT* ObsPart::getBufferIndex(const SplitNux* nux) const {
  return bufferIndex(nux->getPreCand());
}


SampleRank* ObsPart::getBuffers(const SplitNux* nux, IndexT*& sIdx) const {
  return buffers(nux->getPreCand(), sIdx);
}


SampleRank* ObsPart::getPredBase(const SplitNux* nux) const {
  return getPredBase(nux->getPreCand());
}


void ObsPart::prepath(const IdxPath *idxPath,
		      const unsigned int reachBase[],
		      const PreCand& mrra,
		      const IndexRange& idxRange,
		      unsigned int pathMask,
		      bool idxUpdate,
		      unsigned int pathCount[]) {
  prepath(idxPath, reachBase, idxUpdate, idxRange, pathMask, bufferIndex(mrra), &pathIdx[getStageOffset(mrra.splitCoord.predIdx)], pathCount);
}

void ObsPart::prepath(const IdxPath *idxPath,
                         const unsigned int *reachBase,
                         bool idxUpdate,
                         const IndexRange& idxRange,
                         unsigned int pathMask,
                         unsigned int idxVec[],
                         PathT prepath[],
                         unsigned int pathCount[]) const {
  for (IndexT idx = idxRange.getStart(); idx < idxRange.getEnd(); idx++) {
    PathT path = idxPath->update(idxVec[idx], pathMask, reachBase, idxUpdate);
    prepath[idx] = path;
    if (NodePath::isActive(path)) {
      pathCount[path]++;
    }
  }
}


void ObsPart::branchUpdate(const SplitNux* nux,
			   const vector<IndexRange>& range,
			   BranchSense* branchSense,
			   CritEncoding& enc) const {
  for (auto rg : range) {
    branchUpdate(nux, rg, branchSense, enc);
  }
}


void ObsPart::branchUpdate(const SplitNux* nux,
			   const IndexRange& range,
			   BranchSense* branchSense,
			   CritEncoding& enc) const {
  enc.increment ? branchSet(nux, range, branchSense, enc) : branchUnset(nux, range, branchSense, enc);
}


void ObsPart::branchSet(const SplitNux* nux,
			const IndexRange& range,
			BranchSense* branchSense,
			CritEncoding& enc) const {
  IndexT* sIdx;
  SampleRank* spn = getBuffers(nux, sIdx);
  if (enc.exclusive) {
    for (IndexT opIdx = range.getStart(); opIdx != range.getEnd(); opIdx++) {
      if (branchSense->setExclusive(sIdx[opIdx], enc.trueEncoding())) {
	spn[opIdx].encode(enc);
      }
    }
  }
  else {
    for (IndexT opIdx = range.getStart(); opIdx != range.getEnd(); opIdx++) {
      branchSense->set(sIdx[opIdx], enc.trueEncoding());
      spn[opIdx].encode(enc);
    }
  }
}


void ObsPart::branchUnset(const SplitNux* nux,
			   const IndexRange& range,
			   BranchSense* branchSense,
			   CritEncoding& enc) const {
  IndexT* sIdx;
  SampleRank* spn = getBuffers(nux, sIdx);
  if (enc.exclusive) {
    for (IndexT opIdx = range.getStart(); opIdx != range.getEnd(); opIdx++) {
      if (branchSense->isExplicit(sIdx[opIdx])) {
	branchSense->unset(sIdx[opIdx], enc.trueEncoding());
	spn[opIdx].encode(enc);
      }
    }
  }
  else {
    for (IndexT opIdx = range.getStart(); opIdx != range.getEnd(); opIdx++) {
      branchSense->unset(sIdx[opIdx], enc.trueEncoding());
      spn[opIdx].encode(enc);
    }
  }
}


void ObsPart::rankRestage(const PreCand& mrra,
                          const IndexRange& idxRange,
                          unsigned int reachOffset[],
                          unsigned int rankPrev[],
                          unsigned int rankCount[]) {
  SampleRank *source, *targ;
  IndexT *idxSource, *idxTarg;
  buffers(mrra, source, idxSource, targ, idxTarg);

  PathT *pathBlock = &pathIdx[getStageOffset(mrra.splitCoord.predIdx)];
  for (IndexT idx = idxRange.idxStart; idx < idxRange.getEnd(); idx++) {
    unsigned int path = pathBlock[idx];
    if (NodePath::isActive(path)) {
      SampleRank spNode = source[idx];
      IndexT rank = spNode.getRank();
      rankCount[path] += (rank == rankPrev[path] ? 0 : 1);
      rankPrev[path] = rank;
      IndexT destIdx = reachOffset[path]++;
      targ[destIdx] = spNode;
      idxTarg[destIdx] = idxSource[idx];
    }
  }
}


void ObsPart::indexRestage(const IdxPath *idxPath,
                           const unsigned int reachBase[],
                           const PreCand& mrra,
                           const IndexRange& idxRange,
                           unsigned int pathMask,
                           bool idxUpdate,
                           unsigned int reachOffset[],
                           unsigned int splitOffset[]) {
  unsigned int *idxSource, *idxTarg;
  indexBuffers(mrra, idxSource, idxTarg);

  for (IndexT idx = idxRange.idxStart; idx < idxRange.getEnd(); idx++) {
    IndexT sIdx = idxSource[idx];
    PathT path = idxPath->update(sIdx, pathMask, reachBase, idxUpdate);
    if (NodePath::isActive(path)) {
      unsigned int targOff = reachOffset[path]++;
      idxTarg[targOff] = sIdx; // Semi-regular:  split-level target store.
      destRestage[idx] = targOff;
      //      destSplit[idx] = splitOffset[path]++; // Speculative.
    }
    else {
      destRestage[idx] = bagCount;
      //destSplit[idx] = bagCount;
    }
  }
}

