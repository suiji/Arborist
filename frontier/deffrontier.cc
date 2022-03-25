// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file deffontier.cc

   @brief Methods involving individual definition layers.

   @author Mark Seligman
 */

#include "layout.h"
#include "deffrontier.h"
#include "path.h"
#include "defmap.h"
#include "partition.h"


DefFrontier::DefFrontier(IndexT nSplit_,
		   PredictorT nPred_,
		   IndexT bagCount,
		   IndexT idxLive,
		   bool nodeRel_,
		   DefMap* defMap_) :
  defMap(defMap_),
  nPred(nPred_),
  nSplit(nSplit_),
  noIndex(bagCount),
  defCount(0), del(0),
  rangeAnc(vector<IndexRange>(nSplit)),
  mrra(vector<LiveBits>(nSplit * nPred)),
  denseCoord(vector<DenseCoord>(nSplit * defMap->getNPredDense())),
  relPath(make_unique<IdxPath>(idxLive)),
  nodeRel(nodeRel_) {
  NodePath::setNoSplit(bagCount);
  LiveBits df;

  // Coprocessor only.
  //  fill(mrra.begin(), mrra.end(), df);
}
    

void DefFrontier::rootDefine(PredictorT predIdx,
			     const StageCount& stageCount) {
  mrra[predIdx].init(0, stageCount.getRunCount() == 1);
  setDense(SplitCoord(0, predIdx), stageCount.idxImplicit);
  defCount++;
}


bool DefFrontier::nonreachPurge() {
  bool purged = false;
  for (IndexT mrraIdx = 0; mrraIdx < nSplit; mrraIdx++) {
    if (liveCount[mrraIdx] == 0) {
      for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
        undefine(SplitCoord(mrraIdx, predIdx)); // Harmless if undefined.
        purged = true;
      }
    }
  }

  return purged;
}


void DefFrontier::flush(DefMap* defMap) {
  for (IndexT mrraIdx = 0; mrraIdx < nSplit; mrraIdx++) {
    for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
      flushDef(SplitCoord(mrraIdx, predIdx), defMap);
    }
  }
}


void DefFrontier::flushDef(const SplitCoord& splitCoord,
			DefMap* defMap) {
  if (!isDefined(splitCoord)) {
    return;
  }
  if (defMap == nullptr) {
    undefine(splitCoord);
    return;
  }
  if (del == 0) {
    return;
  }
  bool singleton;
  MRRA preCand = consume(splitCoord, singleton);
  unsigned int pathStart = preCand.splitCoord.backScale(del);
  for (unsigned int path = 0; path < backScale(1); path++) {
    defMap->addDef(MRRA(SplitCoord(nodePath[pathStart + path].getSplitIdx(), preCand.splitCoord.predIdx), preCand.compBuffer()), singleton);
  }
  if (!singleton) {
    defMap->appendAncestor(preCand);
  }
}


void DefFrontier::setStageCount(const SplitCoord& splitCoord,
			    const StageCount& stageCount) {
  mrra[splitCoord.strideOffset(nPred)].setSingleton(stageCount);
}


void LiveBits::setSingleton(const StageCount& stageCount) {
  setSingleton(stageCount.isSingleton());
}


bool DefFrontier::backdate(const IdxPath *one2Front) {
  if (nodeRel) {
    relPath->backdate(one2Front);
    return true;
  }
  else
    return false;
}


void DefFrontier::relExtinct(IndexT idx) {
  relPath->setExtinct(idx);
}


void DefFrontier::relLive(IndexT idx, PathT path, IndexT targIdx, IndexT ndBase) {
  relPath->setLive(idx, path, targIdx, targIdx - ndBase);
}


void DefFrontier::reachingPaths() {
  del++;
  
  nodePath = vector<NodePath>(backScale(nSplit));
  liveCount = vector<IndexT>(nSplit);
}


void DefFrontier::pathInit(IndexT splitIdx,
			   PathT path,
			   const IndexRange& bufRange,
			   IndexT idxStart) {
  IndexT mrraIdx = defMap->getHistory(this, splitIdx);
  IndexT pathOff = backScale(mrraIdx);
  unsigned int pathBits = path & pathMask();
  nodePath[pathOff + pathBits].init(splitIdx, bufRange, idxStart);
  liveCount[mrraIdx]++;
}


void DefFrontier::rankRestage(ObsPart* obsPart,
			      const MRRA& mrra,
			      DefFrontier* levelFront) {
  IndexT reachOffset[NodePath::pathMax()];
  if (nodeRel) { // Both levels employ node-relative indexing.
    IndexT reachBase[NodePath::pathMax()];
    offsetClone(mrra.splitCoord, reachOffset, reachBase);
    rankRestage(obsPart, mrra, levelFront, reachOffset, reachBase);
  }
  else { // Source level employs subtree indexing.  Target may or may not.
    offsetClone(mrra.splitCoord, reachOffset);
    rankRestage(obsPart, mrra, levelFront, reachOffset);
  }
}


void DefFrontier::offsetClone(const SplitCoord &mrra,
			      IndexT reachOffset[],
			      IndexT reachBase[]) {
  IndexT nodeStart = mrra.backScale(del);
  for (unsigned int i = 0; i < backScale(1); i++) {
    reachOffset[i] = nodePath[nodeStart + i].getIdxStart();
  }
  if (reachBase != nullptr) {
    for (unsigned int i = 0; i < backScale(1); i++) {
      reachBase[i] = nodePath[nodeStart + i].getNodeStart();
    }
  }
}


void DefFrontier::rankRestage(ObsPart* obsPart,
			   const MRRA& mrra,
			   DefFrontier* levelFront,
			   IndexT reachOffset[],
			   const IndexT reachBase[]) {
  IndexT pathCount[NodePath::pathMax()];
  fill(pathCount, pathCount + backScale(1), 0);

  obsPart->prepath(this, nodeRel ?  getFrontPath() : defMap->getSubtreePath(), reachBase, mrra, pathMask(), reachBase == nullptr ? levelFront->isNodeRel() : true, pathCount);

  // Successors may or may not themselves be dense.
  packDense(pathCount, levelFront, mrra, reachOffset);

  IndexT rankCount[NodePath::pathMax()];
  obsPart->rankRestage(this, mrra, reachOffset, rankCount);
  setStageCounts(mrra, pathCount, rankCount);
}


void DefFrontier::packDense(const IndexT pathCount[],
			    DefFrontier* levelFront,
			    const MRRA& mrra,
			    IndexT reachOffset[]) const {
  if (!isDense(mrra)) {
    return;
  }
  IndexT idxStart = getRange(mrra).getStart();
  const NodePath* pathPos = &nodePath[mrra.splitCoord.backScale(del)];
  PredictorT predIdx = mrra.splitCoord.predIdx;
  for (unsigned int path = 0; path < backScale(1); path++) {
    IndexRange idxRange;
    SplitCoord coord;
    if (pathPos[path].getCoords(predIdx, coord, idxRange)) {
      IndexT margin = idxRange.getStart() - idxStart;
      IndexT extentDense = pathCount[path];
      levelFront->setDense(coord, idxRange.getExtent() - extentDense, margin);
      reachOffset[path] -= margin;
      idxStart += extentDense;
    }
  }
}


void DefFrontier::setStageCounts(const MRRA& mrra, const IndexT pathCount[], const IndexT rankCount[]) const {
  SplitCoord coord = mrra.splitCoord;
  const NodePath* pathPos = &nodePath[coord.backScale(del)];
  for (unsigned int path = 0; path < backScale(1); path++) {
    IndexRange idxRange;
    SplitCoord outCoord;
    if (pathPos[path].getCoords(coord.predIdx, outCoord, idxRange)) {
      defMap->setStageCount(outCoord, idxRange.getExtent() - pathCount[path], rankCount[path]);
    }
  }
}


/**
     @brief Sets the density-associated parameters for a reached node.
  */
void DefFrontier::setDense(const SplitCoord& splitCoord,
			IndexT idxImplicit,
			IndexT margin) {
  if (idxImplicit > 0 || margin > 0) {
    mrra[splitCoord.strideOffset(nPred)].setDense();
    denseCoord[defMap->denseOffset(splitCoord)].init(idxImplicit, margin);
  }
}


void DefFrontier::adjustRange(const MRRA& cand,
			   IndexRange& idxRange) const {
  if (isDense(cand)) {
    denseCoord[defMap->denseOffset(cand)].adjustRange(idxRange);
  }
}

  
IndexT DefFrontier::getImplicit(const MRRA& cand) const {
  return isDense(cand) ? denseCoord[defMap->denseOffset(cand)].getImplicit() : 0;
}

