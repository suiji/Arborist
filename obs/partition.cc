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
  obsCell = new ObsCell[2 * bufferSize];

  // Coprocessor variants:
  //  vector<unsigned int> destRestage(bufferSize);
  //  vector<unsigned int> destSplit(bufferSize);
}


/**
  @brief Base class destructor.
 */
ObsPart::~ObsPart() {
  delete [] obsCell;
  delete [] indexBase;
}


IndexT* ObsPart::getIdxBuffer(const SplitNux* nux) const {
  return idxBuffer(nux->getMRRA());
}


ObsCell* ObsPart::getBuffers(const SplitNux& nux, IndexT*& sIdx) const {
  return buffers(nux.getMRRA(), sIdx);
}


ObsCell* ObsPart::getPredBase(const SplitNux* nux) const {
  return getPredBase(nux->getMRRA());
}


vector<IndexT> ObsPart::prepath(const DefFrontier* dfAncestor,
				const DefFrontier* dfCurrent,
				const MRRA& mrra) {
  IndexRange idxRange = dfAncestor->getRange(mrra);
  IdxPath* idxPath = dfAncestor->getIndexPath();
  IndexT* indexVec = idxBuffer(mrra);
  PathT* prePath = &pathIdx[getStageOffset(mrra.splitCoord.predIdx)];
  vector<IndexT> pathCount(dfAncestor->backScale(1));
  for (IndexT idx = idxRange.getStart(); idx != idxRange.getEnd(); idx++) {
    PathT path;
    if (idxPath->pathSucc(indexVec[idx], dfAncestor->pathMask(), path)) {
      pathCount[path]++;
    }
    prePath[idx] = path;
  }

  return pathCount;
}


vector<IndexT> ObsPart::rankRestage(const DefFrontier* dfAncestor,
				    const MRRA& mrra,
				    vector<IndexT>& reachOffset) {
  ObsCell *srSource, *srTarg;
  IndexT *idxSource, *idxTarg;
  buffers(mrra, srSource, idxSource, srTarg, idxTarg);

  vector<IndexT> rankCount(dfAncestor->backScale(1));
  vector<IndexT> rankPrev(dfAncestor->backScale(1));
  fill(rankPrev.begin(), rankPrev.end(), noRank);

  PathT *pathBlock = &pathIdx[getStageOffset(mrra.splitCoord.predIdx)];
  IndexRange idxRange = dfAncestor->getRange(mrra);
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
  return rankCount;
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
