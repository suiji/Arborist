// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file partition.cc

   @brief Methods to repartition observation frame by tree node.

   @author Mark Seligman
 */

#include "deffrontier.h"
#include "partition.h"
#include "layout.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "path.h"
#include "branchsense.h"

#include <numeric>


/**
   @brief Base class constructor.
 */
ObsPart::ObsPart(const Layout* layout,
		 IndexT bagCount_) :
  bagCount(bagCount_),
  bufferSize(layout->getSafeSize(bagCount)),
  pathIdx(bufferSize),
  stageRange(layout->getNPred()),
  noRank(layout->getNoRank()) {
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
  return bufferIndex(nux->getMRRA());
}


SampleRank* ObsPart::getBuffers(const SplitNux& nux, IndexT*& sIdx) const {
  return buffers(nux.getMRRA(), sIdx);
}


SampleRank* ObsPart::getPredBase(const SplitNux* nux) const {
  return getPredBase(nux->getMRRA());
}


void ObsPart::prepath(const DefFrontier* layer,
		      const IdxPath *idxPath,
		      const unsigned int reachBase[],
		      const MRRA& mrra,
		      unsigned int pathMask,
		      bool idxUpdate,
		      unsigned int pathCount[]) {
  prepath(idxPath, reachBase, idxUpdate, layer->getRange(mrra), pathMask, bufferIndex(mrra), &pathIdx[getStageOffset(mrra.splitCoord.predIdx)], pathCount);
}

void ObsPart::prepath(const IdxPath *idxPath,
                         const unsigned int *reachBase,
                         bool idxUpdate,
                         const IndexRange& idxRange,
                         unsigned int pathMask,
                         unsigned int idxVec[],
                         PathT prepath[],
                         unsigned int pathCount[]) const {
  for (IndexT idx = idxRange.getStart(); idx != idxRange.getEnd(); idx++) {
    PathT path = idxPath->update(idxVec[idx], pathMask, reachBase, idxUpdate);
    prepath[idx] = path;
    if (NodePath::isActive(path)) {
      pathCount[path]++;
    }
  }
}


void ObsPart::rankRestage(const DefFrontier* layer,
			  const MRRA& mrra,
                          unsigned int reachOffset[],
                          unsigned int rankCount[]) {
  SampleRank *srSource, *srTarg;
  IndexT *idxSource, *idxTarg;
  buffers(mrra, srSource, idxSource, srTarg, idxTarg);

  IndexT rankPrev[NodePath::pathMax()];
  fill(rankPrev, rankPrev + layer->backScale(1), noRank);
  fill(rankCount, rankCount + layer->backScale(1), 0);

  PathT *pathBlock = &pathIdx[getStageOffset(mrra.splitCoord.predIdx)];
  IndexRange idxRange = layer->getRange(mrra);
  for (IndexT idx = idxRange.idxStart; idx < idxRange.getEnd(); idx++) {
    unsigned int path = pathBlock[idx];
    if (NodePath::isActive(path)) {
      SampleRank sourceNode = srSource[idx];
      IndexT rank = sourceNode.getRank();
      rankCount[path] += (rank == rankPrev[path] ? 0 : 1);
      rankPrev[path] = rank;
      IndexT destIdx = reachOffset[path]++;
      srTarg[destIdx] = sourceNode;
      idxTarg[destIdx] = idxSource[idx];
    }
  }
}


void ObsPart::indexRestage(const IdxPath *idxPath,
                           const unsigned int reachBase[],
                           const MRRA& mrra,
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


IndexT ObsPart::countRanks(PredictorT predIdx,
			   unsigned int bufIdx,
			   IndexT rank,
			   IndexT idxExpl) const {
  SampleRank* srStart = bufferNode(predIdx, bufIdx);
  IndexT rankCount = 0; // # explicit ranks observed.
  for (SampleRank* sr = srStart; sr != srStart + idxExpl; sr++) {
    IndexT rankPrev = exchange(rank, sr->getRank());
    rankCount += rank == rankPrev ? 0 : 1;
  }

  return rankCount;
}
