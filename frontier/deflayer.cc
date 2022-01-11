// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file deflayer.cc

   @brief Methods involving individual definition layers.

   @author Mark Seligman
 */

#include "layout.h"
#include "deflayer.h"
#include "path.h"
#include "deffrontier.h"
#include "partition.h"


DefLayer::DefLayer(IndexT nSplit_,
		   PredictorT nPred_,
		   IndexT bagCount,
		   IndexT idxLive,
		   bool nodeRel_,
		   DefFrontier* defMap_) :
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
    

void DefLayer::rootDefine(PredictorT predIdx,
			  const StageCount& stageCount) {
  mrra[predIdx].init(0, stageCount.getRunCount() == 1);
  setDense(SplitCoord(0, predIdx), stageCount.idxImplicit);
  defCount++;
}


bool DefLayer::nonreachPurge() {
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


void DefLayer::flush(DefFrontier* defMap) {
  for (IndexT mrraIdx = 0; mrraIdx < nSplit; mrraIdx++) {
    for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
      flushDef(SplitCoord(mrraIdx, predIdx), defMap);
    }
  }
}


void DefLayer::flushDef(const SplitCoord& splitCoord,
			DefFrontier* defMap) {
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


void DefLayer::setStageCount(const SplitCoord& splitCoord,
			    const StageCount& stageCount) {
  mrra[splitCoord.strideOffset(nPred)].setSingleton(stageCount);
}


void LiveBits::setSingleton(const StageCount& stageCount) {
  setSingleton(stageCount.isSingleton());
}



bool DefLayer::backdate(const IdxPath *one2Front) {
  if (!nodeRel)
    return false;

  relPath->backdate(one2Front);
  return true;
}


void DefLayer::reachingPaths() {
  del++;
  
  nodePath = vector<NodePath>(backScale(nSplit));
  liveCount = vector<unsigned int>(nSplit);
}


void DefLayer::setExtinct(unsigned int idx) {
  relPath->setExtinct(idx);
}


void DefLayer::pathInit(IndexT splitIdx, unsigned int path, const IndexRange& bufRange, IndexT relBase) {
  IndexT mrraIdx = defMap->getHistory(this, splitIdx);
  unsigned int pathOff = backScale(mrraIdx);
  unsigned int pathBits = path & pathMask();
  nodePath[pathOff + pathBits].init(splitIdx, bufRange, relBase);
  liveCount[mrraIdx]++;
}


void DefLayer::setLive(IndexT idx, unsigned int path, IndexT targIdx, IndexT ndBase) {
  relPath->setLive(idx, path, targIdx, targIdx - ndBase);
}


void DefLayer::rankRestage(ObsPart* obsPart,
			   const MRRA& mrra,
			   DefLayer* levelFront) {
  unsigned int reachOffset[NodePath::pathMax()];
  if (nodeRel) { // Both levels employ node-relative indexing.
    unsigned int reachBase[NodePath::pathMax()];
    offsetClone(mrra.splitCoord, reachOffset, reachBase);
    rankRestage(obsPart, mrra, levelFront, reachOffset, reachBase);
  }
  else { // Source level employs subtree indexing.  Target may or may not.
    offsetClone(mrra.splitCoord, reachOffset);
    rankRestage(obsPart, mrra, levelFront, reachOffset);
  }
}


void DefLayer::offsetClone(const SplitCoord &mrra,
		   IndexT reachOffset[],
		   IndexT reachBase[]) {
  unsigned int nodeStart = mrra.backScale(del);
  for (unsigned int i = 0; i < backScale(1); i++) {
    reachOffset[i] = nodePath[nodeStart + i].getIdxStart();
  }
  if (reachBase != nullptr) {
    for (unsigned int i = 0; i < backScale(1); i++) {
      reachBase[i] = nodePath[nodeStart + i].getRelBase();
    }
  }
}


void DefLayer::rankRestage(ObsPart* obsPart,
			   const MRRA& mrra,
			   DefLayer* levelFront,
			   unsigned int reachOffset[],
			   const unsigned int reachBase[]) {
  unsigned int pathCount[NodePath::pathMax()];
  fill(pathCount, pathCount + backScale(1), 0);

  obsPart->prepath(this, nodeRel ?  getFrontPath() : defMap->getSubtreePath(), reachBase, mrra, pathMask(), reachBase == nullptr ? levelFront->isNodeRel() : true, pathCount);

  // Successors may or may not themselves be dense.
  packDense(pathCount, levelFront, mrra, reachOffset);

  IndexT rankCount[NodePath::pathMax()];
  obsPart->rankRestage(this, mrra, reachOffset, rankCount);
  setStageCounts(mrra, pathCount, rankCount);
}


void DefLayer::packDense(const unsigned int pathCount[],
			 DefLayer* levelFront,
			 const MRRA& mrra,
			 unsigned int reachOffset[]) const {
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


void DefLayer::setStageCounts(const MRRA& mrra, const unsigned int pathCount[], const unsigned int rankCount[]) const {
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


void DefLayer::indexRestage(ObsPart *obsPart,
			    const MRRA &mrra,
			    const DefLayer *levelFront,
			    const vector<IndexT>& offCand) {
  unsigned int reachOffset[NodePath::pathMax()];
  unsigned int splitOffset[NodePath::pathMax()];
  if (nodeRel) { // Both levels employ node-relative indexing.
    IndexT reachBase[NodePath::pathMax()];
    offsetClone(mrra.splitCoord, offCand, reachOffset, splitOffset, reachBase);
    indexRestage(obsPart, mrra, levelFront, reachBase, reachOffset, splitOffset);
  }
  else { // Source level employs subtree indexing.  Target may or may not.
    offsetClone(mrra.splitCoord, offCand, reachOffset, splitOffset);
    indexRestage(obsPart, mrra, levelFront, nullptr, reachOffset, splitOffset);
  }
}


// COPROC:
/**
   @brief Clones offsets along path reaching from ancestor node.

   @param mrra is an MRRA coordinate.

   @param reachOffset holds the starting offset positions along the path.
 */
void DefLayer::offsetClone(const SplitCoord& mrra,
			   const vector<IndexT>& offCand,
			   IndexT reachOffset[],
			   IndexT splitOffset[],
			   IndexT reachBase[]) {
  IndexT nodeStart = mrra.backScale(del);
  for (unsigned int i = 0; i < backScale(1); i++) {
    reachOffset[i] = nodePath[nodeStart + i].getIdxStart();
    splitOffset[i] = offCand[mrra.strideOffset(nPred)];
  }
  if (reachBase != nullptr) {
    for (unsigned int i = 0; i < backScale(1); i++) {
      reachBase[i] = nodePath[nodeStart + i].getRelBase();
    }
  }
}


void DefLayer::indexRestage(ObsPart* obsPart,
			    const MRRA& mrra,
			    const DefLayer *levelFront,
			    const unsigned int reachBase[],
			    unsigned int reachOffset[],
			    unsigned int splitOffset[]) {
  obsPart->indexRestage(nodeRel ? getFrontPath() : defMap->getSubtreePath(),
                        reachBase, mrra, getRange(mrra),
                        pathMask(),
                        reachBase == nullptr ? levelFront->isNodeRel() : true,
                        reachOffset,
                        splitOffset);
}


/**
     @brief Sets the density-associated parameters for a reached node.
  */
void DefLayer::setDense(const SplitCoord& splitCoord,
			IndexT idxImplicit,
			IndexT margin) {
  if (idxImplicit > 0 || margin > 0) {
    mrra[splitCoord.strideOffset(nPred)].setDense();
    denseCoord[defMap->denseOffset(splitCoord)].init(idxImplicit, margin);
  }
}


void DefLayer::adjustRange(const MRRA& cand,
			   IndexRange& idxRange) const {
  if (isDense(cand)) {
    denseCoord[defMap->denseOffset(cand)].adjustRange(idxRange);
  }
}

  
IndexT DefLayer::getImplicit(const MRRA& cand) const {
  return isDense(cand) ? denseCoord[defMap->denseOffset(cand)].getImplicit() : 0;
}


