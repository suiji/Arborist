// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file level.cc

   @brief Methods involving individual tree levels during training.

   @author Mark Seligman
 */

#include "level.h"
#include "path.h"
#include "bottom.h"
#include "frontier.h"
#include "callback.h"
#include "rankedframe.h"
#include "runset.h"
#include "obspart.h"
#include "splitfrontier.h"
#include "splitnux.h"


Level::Level(IndexT nSplit_,
             PredictorT nPred_,
             const RankedFrame* rankedFrame,
             IndexT bagCount,
             IndexT idxLive_,
             bool _nodeRel,
             Bottom *_bottom) :
  nPred(nPred_),
  denseIdx(rankedFrame->getDenseIdx()),
  nPredDense(rankedFrame->getNPredDense()),
  nSplit(nSplit_),
  noIndex(bagCount),
  idxLive(idxLive_),
  defCount(0), del(0),
  indexAnc(vector<IndexRange>(nSplit)),
  def(vector<MRRA>(nSplit * nPred)),
  denseCoord(vector<DenseCoord>(nSplit * nPredDense)),
  relPath(make_unique<IdxPath>(idxLive)),
  nodeRel(_nodeRel),
  bottom(_bottom)
{
  MRRA df;

  // Coprocessor only.
  fill(def.begin(), def.end(), df);
}

Level::~Level() {
}


bool Level::nonreachPurge() {
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


void Level::flush(bool forward) {
  for (IndexT mrraIdx = 0; mrraIdx < nSplit; mrraIdx++) {
    for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
      SplitCoord splitCoord(mrraIdx, predIdx);
      if (!isDefined(splitCoord))
        continue;
      if (forward) {
        flushDef(splitCoord);
      }
      else {
        undefine(splitCoord);
      }
    }
  }
}


void Level::flushDef(const SplitCoord& splitCoord) {
  if (del == 0) // Already flushed to front level.
    return;

  unsigned int bufIdx;
  bool singleton;
  consume(splitCoord, bufIdx, singleton);
  frontDef(splitCoord, bufIdx, singleton);
  if (!singleton)
    bottom->scheduleRestage(del, splitCoord, bufIdx);
}


void Level::frontDef(const SplitCoord& splitCoord, unsigned int bufIdx, bool singleton) {
  unsigned int pathStart = splitCoord.backScale(del);
  for (unsigned int path = 0; path < backScale(1); path++) {
    bottom->addDef(SplitCoord(nodePath[pathStart + path].getSplitIdx(), splitCoord.predIdx), 1 - bufIdx, singleton);
  }
}


IndexRange Level::getRange(const SplitCoord& mrra) const {
  IndexRange idxRange = indexAnc[mrra.nodeIdx];
  adjustRange(mrra, idxRange);
  return idxRange;
}


void Level::adjustRange(const SplitCoord& splitCoord,
                        IndexRange& idxRange) const {
  if (isDense(splitCoord)) {
    (void) denseCoord[denseOffset(splitCoord)].adjustRange(idxRange);
  }
}



void Level::setSingleton(const SplitCoord& splitCoord) {
  def[splitCoord.strideOffset(nPred)].setSingleton();
}



bool Level::backdate(const IdxPath *one2Front) {
  if (!nodeRel)
    return false;

  relPath->backdate(one2Front);
  return true;
}


void Level::reachingPaths() {
  del++;
  vector<unsigned int> live(nSplit);
  vector<NodePath> path(backScale(nSplit));
  NodePath np;
  IndexRange bufRange;
  bufRange.set(0,0);
  np.init(noIndex, bufRange, 0);
  fill(path.begin(), path.end(), np);
  fill(live.begin(), live.end(), 0);
  
  nodePath = move(path);
  liveCount = move(live);
}


void Level::setExtinct(unsigned int idx) {
  relPath->setExtinct(idx);
}


void
Level::pathInit(IndexT splitIdx, unsigned int path, const IndexRange& bufRange, IndexT relBase) {
  IndexT mrraIdx = bottom->getHistory(this, splitIdx);
  unsigned int pathOff = backScale(mrraIdx);
  unsigned int pathBits = path & pathMask();
  nodePath[pathOff + pathBits].init(splitIdx, bufRange, relBase);
  liveCount[mrraIdx]++;
}


void
Level::setLive(IndexT idx, unsigned int path, IndexT targIdx, IndexT ndBase) {
  relPath->setLive(idx, path, targIdx, targIdx - ndBase);
}


IndexRange
Level::adjustRange(const SplitNux& cand,
		   const SplitFrontier* splitFrontier) const {
  IndexRange idxRange = splitFrontier->getBufRange(cand);
  if (isDense(cand)) {
    denseCoord[denseOffset(cand)].adjustRange(idxRange);
  }
  return idxRange;
}


IndexT
Level::getImplicit(const SplitNux& cand) const {
  return isDense(cand) ? denseCoord[denseOffset(cand)].getImplicit() : 0;
}


IndexT
Level::denseOffset(const SplitNux& cand) const {
  return denseOffset(cand.getSplitCoord());
}


bool
Level::isDense(const SplitNux& cand) const {
  return isDense(cand.getSplitCoord());
}


void
Level::rankRestage(ObsPart* obsPart,
		   const SplitCoord& mrra,
		   Level* levelFront,
		   unsigned int bufIdx) {
  unsigned int reachOffset[NodePath::pathMax()];
  if (nodeRel) { // Both levels employ node-relative indexing.
    unsigned int reachBase[NodePath::pathMax()];
    offsetClone(mrra, reachOffset, reachBase);
    rankRestage(obsPart, mrra, levelFront, bufIdx, reachOffset, reachBase);
  }
  else { // Source level employs subtree indexing.  Target may or may not.
    offsetClone(mrra, reachOffset);
    rankRestage(obsPart, mrra, levelFront, bufIdx, reachOffset);
  }
}


void
Level::offsetClone(const SplitCoord &mrra,
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


void
Level::rankRestage(ObsPart* obsPart,
		   const SplitCoord& mrra,
		   Level* levelFront,
		   unsigned int bufIdx,
		   unsigned int reachOffset[],
		   const unsigned int reachBase[]) {
  IndexRange idxRange = getRange(mrra);
  unsigned int pathCount[NodePath::pathMax()];
  fill(pathCount, pathCount + backScale(1), 0);

  PredictorT predIdx = mrra.predIdx;
  obsPart->prepath(nodeRel ?  getFrontPath() : bottom->getSubtreePath(), reachBase, predIdx, bufIdx, idxRange, pathMask(), reachBase == nullptr ? levelFront->isNodeRel() : true, pathCount);

  // Successors may or may not themselves be dense.
  packDense(idxRange.getStart(), pathCount, levelFront, mrra, reachOffset);

  IndexT rankPrev[NodePath::pathMax()];
  IndexT rankCount[NodePath::pathMax()];
  fill(rankPrev, rankPrev + backScale(1), bottom->getNoRank());
  fill(rankCount, rankCount + backScale(1), 0);

  obsPart->rankRestage(predIdx, bufIdx, idxRange, reachOffset, rankPrev, rankCount);
  setRunCounts(mrra, pathCount, rankCount);
}


void
Level::packDense(IndexT idxStart,
                      const unsigned int pathCount[],
                      Level* levelFront,
                      const SplitCoord& mrra,
                      unsigned int reachOffset[]) const {
  if (!isDense(mrra)) {
    return;
  }
  const NodePath* pathPos = &nodePath[mrra.backScale(del)];
  for (unsigned int path = 0; path < backScale(1); path++) {
    IndexRange idxRange;
    IndexT splitIdx = pathPos[path].getCoords(idxRange);
    if (splitIdx != noIndex) {
      IndexT margin = idxRange.getStart() - idxStart;
      IndexT extentDense = pathCount[path];
      levelFront->setDense(SplitCoord(splitIdx, mrra.predIdx), idxRange.getExtent() - extentDense, margin);
      reachOffset[path] -= margin;
      idxStart += extentDense;
    }
  }
}


void
Level::setRunCounts(const SplitCoord &mrra, const unsigned int pathCount[], const unsigned int rankCount[]) const {
  PredictorT predIdx = mrra.predIdx;
  const NodePath* pathPos = &nodePath[mrra.backScale(del)];
  for (unsigned int path = 0; path < backScale(1); path++) {
    IndexRange idxRange;
    IndexT splitIdx = pathPos[path].getCoords(idxRange);
    if (splitIdx != noIndex) {
      bottom->setRunCount(SplitCoord(splitIdx, predIdx), pathCount[path] != idxRange.getExtent(), rankCount[path]);
    }
  }
}


void
Level::indexRestage(ObsPart *obsPart,
		    const SplitCoord &mrra,
		    const Level *levelFront,
		    unsigned int bufIdx,
		    const vector<IndexT>& offCand) {
  unsigned int reachOffset[NodePath::pathMax()];
  unsigned int splitOffset[NodePath::pathMax()];
  if (nodeRel) { // Both levels employ node-relative indexing.
    IndexT reachBase[NodePath::pathMax()];
    offsetClone(mrra, offCand, reachOffset, splitOffset, reachBase);
    indexRestage(obsPart, mrra, levelFront, bufIdx, reachBase, reachOffset, splitOffset);
  }
  else { // Source level employs subtree indexing.  Target may or may not.
    offsetClone(mrra, offCand, reachOffset, splitOffset);
    indexRestage(obsPart, mrra, levelFront, bufIdx, nullptr, reachOffset, splitOffset);
  }
}


// COPROC:
/**
   @brief Clones offsets along path reaching from ancestor node.

   @param mrra is an MRRA coordinate.

   @param reachOffset holds the starting offset positions along the path.
 */
void
Level::offsetClone(const SplitCoord& mrra,
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


void
Level::indexRestage(ObsPart* obsPart,
		    const SplitCoord& mrra,
		    const Level *levelFront,
		    unsigned int bufIdx,
		    const unsigned int reachBase[],
		    unsigned int reachOffset[],
		    unsigned int splitOffset[]) {
  obsPart->indexRestage(nodeRel ? getFrontPath() : bottom->getSubtreePath(),
                        reachBase, mrra, bufIdx, getRange(mrra), 
                        pathMask(),
                        reachBase == nullptr ? levelFront->isNodeRel() : true,
                        reachOffset,
                        splitOffset);
}
