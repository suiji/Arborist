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

#include "deflayer.h"
#include "path.h"
#include "defmap.h"
#include "obspart.h"


DefLayer::DefLayer(IndexT nSplit_,
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
  indexAnc(vector<IndexRange>(nSplit)),
  def(vector<MRRA>(nSplit * nPred)),
  denseCoord(vector<DenseCoord>(nSplit * defMap->getNPredDense())),
  relPath(make_unique<IdxPath>(idxLive)),
  nodeRel(nodeRel_) {
  NodePath::setNoSplit(bagCount);
  MRRA df;

  // Coprocessor only.
  fill(def.begin(), def.end(), df);
}
    

DefLayer::~DefLayer() {
}


bool DefLayer::nonreachPurge() {
  bool purged = false;
  for (IndexT mrraIdx = 0; mrraIdx < nSplit; mrraIdx++) {
    if (liveCount[mrraIdx] == 0) {
      for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
        undefine(SplitCoord(mrraIdx, predIdx)); // Harmless if already undefined.
        purged = true;
      }
    }
  }

  return purged;
}


void DefLayer::flush(DefMap* defMap) {
  for (IndexT mrraIdx = 0; mrraIdx < nSplit; mrraIdx++) {
    for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
      SplitCoord splitCoord(mrraIdx, predIdx);
      if (isDefined(splitCoord)) {
	if (defMap != nullptr) {
	  flushDef(splitCoord, defMap);
	}
	else {
	  undefine(splitCoord);
	}
      }
    }
  }
}


void DefLayer::flushDef(const SplitCoord& splitCoord,
			DefMap* defMap) {
  if (del == 0) {
    return;
  }
  bool singleton;
  PreCand preCand = consume(splitCoord, singleton);
  unsigned int pathStart = preCand.splitCoord.backScale(del);
  for (unsigned int path = 0; path < backScale(1); path++) {
    defMap->addDef(PreCand(SplitCoord(nodePath[pathStart + path].getSplitIdx(), preCand.splitCoord.predIdx), preCand.compBuffer()), singleton);
  }
  if (!singleton) {
    defMap->restageAppend(preCand);
  }
}


void DefLayer::setSingleton(const SplitCoord& splitCoord) {
  def[splitCoord.strideOffset(nPred)].setSingleton();
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


void
DefLayer::pathInit(IndexT splitIdx, unsigned int path, const IndexRange& bufRange, IndexT relBase) {
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
		   const PreCand& mrra,
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
		   const PreCand& mrra,
		   DefLayer* levelFront,
		   unsigned int reachOffset[],
		   const unsigned int reachBase[]) {
  IndexRange idxRange = getRange(mrra);
  unsigned int pathCount[NodePath::pathMax()];
  fill(pathCount, pathCount + backScale(1), 0);

  obsPart->prepath(nodeRel ?  getFrontPath() : defMap->getSubtreePath(), reachBase, mrra, idxRange, pathMask(), reachBase == nullptr ? levelFront->isNodeRel() : true, pathCount);

  // Successors may or may not themselves be dense.
  packDense(idxRange.getStart(), pathCount, levelFront, mrra, reachOffset);

  IndexT rankPrev[NodePath::pathMax()];
  IndexT rankCount[NodePath::pathMax()];
  fill(rankPrev, rankPrev + backScale(1), defMap->getNoRank());
  fill(rankCount, rankCount + backScale(1), 0);

  obsPart->rankRestage(mrra, idxRange, reachOffset, rankPrev, rankCount);
  setRunCounts(mrra.splitCoord, pathCount, rankCount);
}


void DefLayer::packDense(IndexT idxStart,
                      const unsigned int pathCount[],
                      DefLayer* levelFront,
                      const PreCand& mrra,
                      unsigned int reachOffset[]) const {
  if (!isDense(mrra)) {
    return;
  }
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


void DefLayer::setRunCounts(const SplitCoord &mrra, const unsigned int pathCount[], const unsigned int rankCount[]) const {
  PredictorT predIdx = mrra.predIdx;
  const NodePath* pathPos = &nodePath[mrra.backScale(del)];
  for (unsigned int path = 0; path < backScale(1); path++) {
    IndexRange idxRange;
    SplitCoord coord;
    if (pathPos[path].getCoords(predIdx, coord, idxRange)) {
      defMap->setRunCount(coord, pathCount[path] != idxRange.getExtent(), rankCount[path]);
    }
  }
}


void DefLayer::indexRestage(ObsPart *obsPart,
		    const PreCand &mrra,
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
			    const PreCand& mrra,
			    const DefLayer *levelFront,
			    const unsigned int reachBase[],
			    unsigned int reachOffset[],
			    unsigned int splitOffset[]) {
  obsPart->indexRestage(nodeRel ? getFrontPath() :  defMap->getSubtreePath(),
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
			IndexT implicit,
			IndexT margin) {
  if (implicit > 0 || margin > 0) {
    def[splitCoord.strideOffset(nPred)].setDense();
    denseCoord[defMap->denseOffset(splitCoord)].init(implicit, margin);
  }
}

void DefLayer::adjustRange(const PreCand& cand,
			   IndexRange& idxRange) const {
  if (isDense(cand)) {
    denseCoord[defMap->denseOffset(cand)].adjustRange(idxRange);
  }
}

  
IndexT DefLayer::getImplicit(const PreCand& cand) const {
  return isDense(cand) ? denseCoord[defMap->denseOffset(cand)].getImplicit() : 0;
}


