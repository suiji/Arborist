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


unsigned int Level::predFixed = 0;
vector<double> Level::predProb;

Level::Level(IndexType nSplit_,
             unsigned int _nPred,
             const RankedFrame* rankedFrame,
             IndexType bagCount,
             IndexType idxLive_,
             bool _nodeRel,
             Bottom *_bottom) :
  nPred(_nPred),
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
  offCand(vector<IndexType>(nSplit * nPred)),
  nodeRel(_nodeRel),
  bottom(_bottom)
{
  MRRA df;

  // Coprocessor only.
  fill(def.begin(), def.end(), df);
  fill(offCand.begin(), offCand.end(), bagCount);
}

Level::~Level() {
}

void Level::immutables(unsigned int feFixed, const vector<double> &feProb) {
  predFixed = feFixed;
  for (auto prob : feProb) {
    predProb.push_back(prob);
  }
}


void Level::deImmutables() {
  predFixed = 0;
  predProb.clear();
}


bool Level::nonreachPurge() {
  bool purged = false;
  for (unsigned int mrraIdx = 0; mrraIdx < nSplit; mrraIdx++) {
    if (liveCount[mrraIdx] == 0) {
      for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
        undefine(SplitCoord(mrraIdx, predIdx)); // Harmless if already undefined.
        purged = true;
      }
    }
  }

  return purged;
}


void Level::flush(bool forward) {
  for (unsigned int mrraIdx = 0; mrraIdx < nSplit; mrraIdx++) {
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
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


IndexRange Level::getRange(const SplitCoord &mrra) {
  IndexRange idxRange = indexAnc[mrra.nodeIdx];
  adjustRange(mrra, idxRange);
  return idxRange;
}


IndexRange Level::adjustRange(const SplitCoord& splitCoord,
                              const Frontier* frontier,
                              unsigned int& implicit) const {
  IndexSet iSet(frontier->getISet(splitCoord));
  IndexRange idxRange;
  idxRange.set(iSet.getStart(), iSet.getExtent());
  implicit = isDense(splitCoord) ? denseCoord[denseOffset(splitCoord)].adjustRange(idxRange) : 0;

  return idxRange;
}



void Level::adjustRange(const SplitCoord& splitCoord,
                        IndexRange& idxRange) const {
  if (isDense(splitCoord)) {
    (void) denseCoord[denseOffset(splitCoord)].adjustRange(idxRange);
  }
}



void Level::offsetClone(const SplitCoord &mrra, unsigned int reachOffset[], unsigned int reachBase[]) {
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


void Level::pathInit(const Bottom *bottom, unsigned int splitIdx, unsigned int path, const IndexRange& bufRange, unsigned int relBase) {
  unsigned int mrraIdx = bottom->getHistory(this, splitIdx);
  unsigned int pathOff = backScale(mrraIdx);
  unsigned int pathBits = path & pathMask();
  nodePath[pathOff + pathBits].init(splitIdx, bufRange, relBase);
  liveCount[mrraIdx]++;
}


void Level::setLive(unsigned int idx, unsigned int path, unsigned int targIdx, unsigned int ndBase) {
  relPath->setLive(idx, path, targIdx, targIdx - ndBase);
}


bool Level::scheduleSplit(const SplitCoord& splitCoord, unsigned int &rCount) const {
  rCount = bottom->getRunCount(splitCoord);
  return !isSingleton(splitCoord);
}


// TODO:  Preempt overflow by walking wide subtrees depth-nodeIdx.
void Level::candidates(const Frontier* frontier, SplitFrontier *splitFrontier) {
  IndexType cellCount = nSplit * nPred;

  auto ruPred = CallBack::rUnif(cellCount);

  vector<BHPair> heap(predFixed == 0 ? 0 : cellCount);

  unsigned int spanCand = 0;
  for (IndexType splitIdx = 0; splitIdx < nSplit; splitIdx++) {
    IndexType splitOff = splitIdx * nPred;
    if (frontier->isUnsplitable(splitIdx)) { // Node cannot split.
      continue;
    }
    else if (predFixed == 0) { // Probability of predictor splitable.
      candidateProb(splitFrontier, splitIdx, &ruPred[splitOff], spanCand);
    }
    else { // Fixed number of predictors splitable.
      candidateFixed(splitFrontier, splitIdx, &ruPred[splitOff], &heap[splitOff], spanCand);
    }
  }
  setSpan(spanCand);
}


void Level::candidateProb(SplitFrontier *splitFrontier,
                          unsigned int splitIdx,
                          const double ruPred[],
                          unsigned int &spanCand) {
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    if (ruPred[predIdx] < predProb[predIdx]) {
      (void) preschedule(splitFrontier, SplitCoord(splitIdx, predIdx), spanCand);
    }
  }
}

 
void Level::candidateFixed(SplitFrontier *splitFrontier,
                           unsigned int splitIdx,
                           const double ruPred[],
                           BHPair heap[],
                           unsigned int &spanCand) {
  // Inserts negative, weighted probability value:  choose from lowest.
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    BHeap::insert(heap, predIdx, -ruPred[predIdx] * predProb[predIdx]);
  }

  // Pops 'predFixed' items in order of increasing value.
  unsigned int schedCount = 0;
  for (unsigned int heapSize = nPred; heapSize > 0; heapSize--) {
    unsigned int predIdx = BHeap::slotPop(heap, heapSize - 1);
    schedCount += preschedule(splitFrontier, SplitCoord(splitIdx, predIdx), spanCand) ? 1 : 0;
    if (schedCount == predFixed)
      break;
  }
}


bool Level::preschedule(SplitFrontier *splitFrontier,
                        const SplitCoord& splitCoord,
                        unsigned int &spanCand) {
  bottom->reachFlush(splitCoord.nodeIdx, splitCoord.predIdx);

  unsigned int bufIdx;
  if (!isSingleton(splitCoord, bufIdx)) {
    offCand[splitCoord.strideOffset(nPred)] = spanCand;
    spanCand += splitFrontier->preschedule(splitCoord, bufIdx);
    return true;
  }
  return false;
}


void Level::rankRestage(ObsPart* samplePred,
                        const SplitCoord& mrra,
                        Level* levelFront,
                        unsigned int bufIdx) {
  unsigned int reachOffset[NodePath::pathMax()];
  if (nodeRel) { // Both levels employ node-relative indexing.
    unsigned int reachBase[NodePath::pathMax()];
    offsetClone(mrra, reachOffset, reachBase);
    rankRestage(samplePred, mrra, levelFront, bufIdx, reachOffset, reachBase);
  }
  else { // Source level employs subtree indexing.  Target may or may not.
    offsetClone(mrra, reachOffset);
    rankRestage(samplePred, mrra, levelFront, bufIdx, reachOffset);
  }
}


void Level::rankRestage(ObsPart* samplePred,
                        const SplitCoord& mrra,
                        Level* levelFront,
                        unsigned int bufIdx,
                        unsigned int reachOffset[],
                        const unsigned int reachBase[]) {
  IndexRange idxRange = getRange(mrra);
  unsigned int pathCount[NodePath::pathMax()];
  fill(pathCount, pathCount + backScale(1), 0);

  unsigned int predIdx = mrra.predIdx;
  samplePred->prepath(nodeRel ?  getFrontPath() : bottom->getSubtreePath(), reachBase, predIdx, bufIdx, idxRange, pathMask(), reachBase == nullptr ? levelFront->isNodeRel() : true, pathCount);

  // Successors may or may not themselves be dense.
  packDense(idxRange.getStart(), pathCount, levelFront, mrra, reachOffset);

  unsigned int rankPrev[NodePath::pathMax()];
  unsigned int rankCount[NodePath::pathMax()];
  fill(rankPrev, rankPrev + backScale(1), bottom->getNoRank());
  fill(rankCount, rankCount + backScale(1), 0);

  samplePred->rankRestage(predIdx, bufIdx, idxRange, reachOffset, rankPrev, rankCount);
  setRunCounts(bottom, mrra, pathCount, rankCount);
}


void Level::packDense(IndexType idxStart,
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
    IndexType splitIdx = pathPos[path].getCoords(idxRange);
    if (splitIdx != noIndex) {
      IndexType margin = idxRange.getStart() - idxStart;
      IndexType extentDense = pathCount[path];
      levelFront->setDense(SplitCoord(splitIdx, mrra.predIdx), idxRange.getExtent() - extentDense, margin);
      reachOffset[path] -= margin;
      idxStart += extentDense;
    }
  }
}


void Level::setRunCounts(Bottom *bottom, const SplitCoord &mrra, const unsigned int pathCount[], const unsigned int rankCount[]) const {
  unsigned int predIdx = mrra.predIdx;
  const NodePath* pathPos = &nodePath[mrra.backScale(del)];
  for (unsigned int path = 0; path < backScale(1); path++) {
    IndexRange idxRange;
    IndexType splitIdx = pathPos[path].getCoords(idxRange);
    if (splitIdx != noIndex) {
      bottom->setRunCount(SplitCoord(splitIdx, predIdx), pathCount[path] != idxRange.getExtent(), rankCount[path]);
    }
  }
}


// COPROC:
/**
   @brief Clones offsets along path reaching from ancestor node.

   @param mrra is an MRRA coordinate.

   @param reachOffset holds the starting offset positions along the path.
 */
void Level::offsetClone(const SplitCoord& mrra,
                        unsigned int reachOffset[],
                        unsigned int splitOffset[],
                        unsigned int reachBase[]) {
  unsigned int nodeStart = mrra.backScale(del);
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


void Level::indexRestage(ObsPart *samplePred,
                         const SplitCoord &mrra,
                         const Level *levelFront,
                         unsigned int bufIdx) {
  unsigned int reachOffset[NodePath::pathMax()];
  unsigned int splitOffset[NodePath::pathMax()];
  if (nodeRel) { // Both levels employ node-relative indexing.
    unsigned int reachBase[NodePath::pathMax()];
    offsetClone(mrra, reachOffset, splitOffset, reachBase);
    indexRestage(samplePred, mrra, levelFront, bufIdx, reachBase, reachOffset, splitOffset);
  }
  else { // Source level employs subtree indexing.  Target may or may not.
    offsetClone(mrra, reachOffset, splitOffset);
    indexRestage(samplePred, mrra, levelFront, bufIdx, nullptr, reachOffset, splitOffset);
  }
}


void Level::indexRestage(ObsPart* obsPart,
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
