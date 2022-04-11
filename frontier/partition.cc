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
  nodeVec = new ObsCell[2 * bufferSize];

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


ObsCell* ObsPart::getBuffers(const SplitNux& nux, IndexT*& sIdx) const {
  return buffers(nux.getMRRA(), sIdx);
}


ObsCell* ObsPart::getPredBase(const SplitNux* nux) const {
  return getPredBase(nux->getMRRA());
}


void ObsPart::prepath(const DefFrontier* layer,
		      const IdxPath *idxPath,
		      const IndexT reachBase[],
		      const MRRA& mrra,
		      unsigned int pathMask,
		      bool idxUpdate,
		      IndexT pathCount[]) {
  prepath(idxPath, reachBase, idxUpdate, layer->getRange(mrra), pathMask, bufferIndex(mrra), &pathIdx[getStageOffset(mrra.splitCoord.predIdx)], pathCount);
}

void ObsPart::prepath(const IdxPath *idxPath,
		      const IndexT* reachBase,
		      bool idxUpdate,
		      const IndexRange& idxRange,
		      unsigned int pathMask,
		      IndexT idxVec[],
		      PathT prepath[],
		      IndexT pathCount[]) const {
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
                          IndexT reachOffset[],
                          IndexT rankCount[]) {
  ObsCell *srSource, *srTarg;
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
      ObsCell sourceNode = srSource[idx];
      IndexT rank = sourceNode.getRank();
      rankCount[path] += (rank == rankPrev[path] ? 0 : 1);
      rankPrev[path] = rank;
      IndexT destIdx = reachOffset[path]++;
      srTarg[destIdx] = sourceNode;
      idxTarg[destIdx] = idxSource[idx];
    }
  }
}

IndexT ObsPart::countRanks(PredictorT predIdx,
			   unsigned int bufIdx,
			   IndexT rank,
			   IndexT idxExpl) const {
  ObsCell* srStart = bufferNode(predIdx, bufIdx);
  IndexT rankCount = 0; // # explicit ranks observed.
  for (ObsCell* sr = srStart; sr != srStart + idxExpl; sr++) {
    IndexT rankPrev = exchange(rank, sr->getRank());
    rankCount += rank == rankPrev ? 0 : 1;
  }

  return rankCount;
}
