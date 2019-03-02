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
#include "index.h"
#include "callback.h"
#include "bv.h"
#include "runset.h"
#include "samplepred.h"
#include "splitnode.h"


unsigned int Level::predFixed = 0;
vector<double> Level::predProb;

Level::Level(unsigned int _nSplit,
             unsigned int _nPred,
             const vector<unsigned int> &_denseIdx,
             unsigned int _nPredDense,
             unsigned int bagCount,
             unsigned int _idxLive,
             bool _nodeRel,
             Bottom *_bottom) :
  nPred(_nPred),
  denseIdx(_denseIdx),
  nPredDense(_nPredDense),
  nSplit(_nSplit),
  noIndex(bagCount),
  idxLive(_idxLive),
  defCount(0), del(0),
  indexAnc(vector<IndexAnc>(nSplit)),
  def(vector<MRRA>(nSplit * nPred)),
  denseCoord(vector<DenseCoord>(nSplit * nPredDense)),
  relPath(new IdxPath(idxLive)),
  offCand(vector<unsigned int>(nSplit * nPred)),
  nodeRel(_nodeRel),
  bottom(_bottom)
{
  MRRA df;
  df.init();
  // Coprocessor only.
  fill(def.begin(), def.end(), df);
  fill(offCand.begin(), offCand.end(), bagCount);
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
        undefine(mrraIdx, predIdx); // Harmless if already undefined.
        purged = true;
      }
    }
  }

  return purged;
}


void Level::flush(bool forward) {
  for (unsigned int mrraIdx = 0; mrraIdx < nSplit; mrraIdx++) {
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      if (!isDefined(mrraIdx, predIdx))
        continue;
      if (forward) {
        flushDef(mrraIdx, predIdx);
      }
      else {
        undefine(mrraIdx, predIdx);
      }
    }
  }
}


void Level::flushDef(unsigned int mrraIdx, unsigned int predIdx) {
  if (del == 0) // Already flushed to front level.
    return;

  unsigned int bufIdx;
  bool singleton;
  consume(mrraIdx, predIdx, bufIdx, singleton);
  frontDef(mrraIdx, predIdx, bufIdx, singleton);
  if (!singleton)
    bottom->scheduleRestage(del, mrraIdx, predIdx, bufIdx);
}


void Level::frontDef(unsigned int mrraIdx, unsigned int predIdx, unsigned int bufIdx, bool singleton) {
  unsigned int pathStart = backScale(mrraIdx);
  for (unsigned int path = 0; path < backScale(1); path++) {
    bottom->addDef(nodePath[pathStart + path].Idx(), predIdx, 1 - bufIdx, singleton);
  }
}


void Level::getBounds(const SPPair &mrra, unsigned int &startIdx, unsigned int &extent) {
  indexAnc[mrra.first].Ref(startIdx, extent);
  (void) adjustDense(mrra.first, mrra.second, startIdx, extent);
}


Level::~Level() {
  delete relPath;
}


void Level::offsetClone(const SPPair &mrra, unsigned int reachOffset[], unsigned int reachBase[]) {
  unsigned int nodeStart = backScale(mrra.first);
  for (unsigned int i = 0; i < backScale(1); i++) {
    reachOffset[i] = nodePath[nodeStart + i].IdxStart();
  }
  if (reachBase != nullptr) {
    for (unsigned int i = 0; i < backScale(1); i++) {
      reachBase[i] = nodePath[nodeStart + i].RelBase();
    }
  }
}


void Level::setSingleton(unsigned int levelIdx, unsigned int predIdx) {
  def[pairOffset(levelIdx, predIdx)].setSingleton();
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
  np.init(noIndex, 0, 0, 0);
  fill(path.begin(), path.end(), np);
  fill(live.begin(), live.end(), 0);
  
  nodePath = move(path);
  liveCount = move(live);
}


void Level::setExtinct(unsigned int idx) {
  relPath->setExtinct(idx);
}


void Level::pathInit(const Bottom *bottom, unsigned int splitIdx, unsigned int path, unsigned int start, unsigned int extent, unsigned int relBase) {
  unsigned int mrraIdx = bottom->getHistory(this, splitIdx);
  unsigned int pathOff = backScale(mrraIdx);
  unsigned int pathBits = path & pathMask();
  nodePath[pathOff + pathBits].init(splitIdx, start, extent, relBase);
  liveCount[mrraIdx]++;
}


void Level::setLive(unsigned int idx, unsigned int path, unsigned int targIdx, unsigned int ndBase) {
  relPath->setLive(idx, path, targIdx, targIdx - ndBase);
}


bool Level::scheduleSplit(unsigned int splitIdx, unsigned int predIdx, unsigned int &rCount) const {
  rCount = bottom->getRunCount(splitIdx, predIdx);
  return !isSingleton(splitIdx, predIdx);
}


// TODO:  Preempt overflow by walking wide subtrees depth-first.
void Level::candidates(const IndexLevel *index, SplitNode *splitNode) {
  int cellCount = nSplit * nPred;

  auto ruPred = CallBack::rUnif(cellCount);

  vector<BHPair> heap(predFixed == 0 ? 0 : cellCount);

  unsigned int spanCand = 0;
  for (unsigned int splitIdx = 0; splitIdx < nSplit; splitIdx++) {
    unsigned int splitOff = splitIdx * nPred;
    if (index->isUnsplitable(splitIdx)) { // Node cannot split.
      continue;
    }
    else if (predFixed == 0) { // Probability of predictor splitable.
      candidateProb(splitNode, splitIdx, &ruPred[splitOff], index->getExtent(splitIdx), spanCand);
    }
    else { // Fixed number of predictors splitable.
      candidateFixed(splitNode, splitIdx, &ruPred[splitOff], &heap[splitOff], index->getExtent(splitIdx), spanCand);
    }
  }
  setSpan(spanCand);
}


void Level::candidateProb(SplitNode *splitNode,
                          unsigned int splitIdx,
                          const double ruPred[],
                          unsigned int extent,
                          unsigned int &spanCand) {
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    if (ruPred[predIdx] < predProb[predIdx]) {
      (void) preschedule(splitNode, splitIdx, predIdx, extent, spanCand);
    }
  }
}

 
void Level::candidateFixed(SplitNode *splitNode,
                           unsigned int splitIdx,
                           const double ruPred[],
                           BHPair heap[],
                           unsigned int extent,
                           unsigned int &spanCand) {
  // Inserts negative, weighted probability value:  choose from lowest.
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    BHeap::insert(heap, predIdx, -ruPred[predIdx] * predProb[predIdx]);
  }

  // Pops 'predFixed' items in order of increasing value.
  unsigned int schedCount = 0;
  for (unsigned int heapSize = nPred; heapSize > 0; heapSize--) {
    unsigned int predIdx = BHeap::slotPop(heap, heapSize - 1);
    schedCount += preschedule(splitNode, splitIdx, predIdx, extent, spanCand) ? 1 : 0;
    if (schedCount == predFixed)
      break;
  }
}


bool Level::preschedule(SplitNode *splitNode,
                        unsigned int splitIdx,
                        unsigned int predIdx,
                        unsigned int extent,
                        unsigned int &spanCand) {
  bottom->reachFlush(splitIdx, predIdx);

  unsigned int bufIdx;
  if (!isSingleton(splitIdx, predIdx, bufIdx)) {
    splitNode->preschedule(splitIdx, predIdx, bufIdx);
    offCand[pairOffset(splitIdx, predIdx)] = spanCand;
    spanCand += extent;
    return true;
  }
  return false;
}


void Level::rankRestage(SamplePred *samplePred,
                        const SPPair &mrra,
                        Level *levelFront,
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


void Level::rankRestage(SamplePred *samplePred,
                        const SPPair &mrra,
                        Level *levelFront,
                        unsigned int bufIdx,
                        unsigned int reachOffset[],
                        const unsigned int reachBase[]) {
  unsigned int startIdx, extent;
  getBounds(mrra, startIdx, extent);

  unsigned int pathCount[NodePath::pathMax()];
  fill(pathCount, pathCount + backScale(1), 0);

  unsigned int predIdx = mrra.second;
  samplePred->prepath(nodeRel ?  getFrontPath() : bottom->subtreePath(), reachBase, predIdx, bufIdx, startIdx, extent, pathMask(), reachBase == nullptr ? levelFront->isNodeRel() : true, pathCount);

  // Successors may or may not themselves be dense.
  packDense(startIdx, pathCount, levelFront, mrra, reachOffset);

  unsigned int rankPrev[NodePath::pathMax()];
  unsigned int rankCount[NodePath::pathMax()];
  fill(rankPrev, rankPrev + backScale(1), bottom->getNoRank());
  fill(rankCount, rankCount + backScale(1), 0);

  samplePred->rankRestage(predIdx, bufIdx, startIdx, extent, reachOffset, rankPrev, rankCount);
  setRunCounts(bottom, mrra, pathCount, rankCount);
}


void Level::packDense(unsigned int idxLeft,
                      const unsigned int pathCount[],
                      Level *levelFront,
                      const SPPair &mrra,
                      unsigned int reachOffset[]) const {
  if (!isDense(mrra.first, mrra.second)) {
    return;
  }
  const NodePath *pathPos = &nodePath[backScale(mrra.first)];
  for (unsigned int path = 0; path < backScale(1); path++) {
    unsigned int levelIdx, idxStart, extent;
    pathPos[path].getCoords(levelIdx, idxStart, extent);
    if (levelIdx != noIndex) {
      unsigned int margin = idxStart - idxLeft;
      unsigned int extentDense = pathCount[path];
      levelFront->setDense(levelIdx, mrra.second, extent - extentDense, margin);
      reachOffset[path] -= margin;
      idxLeft += extentDense;
    }
  }
}


void Level::setRunCounts(Bottom *bottom, const SPPair &mrra, const unsigned int pathCount[], const unsigned int rankCount[]) const {
  unsigned int predIdx = mrra.second;
  const NodePath *pathPos = &nodePath[backScale(mrra.first)];
  for (unsigned int path = 0; path < backScale(1); path++) {
    unsigned int levelIdx, idxStart, extent;
    pathPos[path].getCoords(levelIdx, idxStart, extent);
    if (levelIdx != noIndex) {
      bottom->setRunCount(levelIdx, predIdx, pathCount[path] != extent, rankCount[path]);
    }
  }
}


// COPROC:
/**
   @brief Clones offsets along path reaching from ancestor node.

   @param mrra is an MRRA coordinate.

   @param reachOffset holds the starting offset positions along the path.
 */
void Level::offsetClone(const SPPair &mrra,
                        unsigned int reachOffset[],
                        unsigned int splitOffset[],
                        unsigned int reachBase[]) {
  unsigned int nodeStart = backScale(mrra.first);
  for (unsigned int i = 0; i < backScale(1); i++) {
    reachOffset[i] = nodePath[nodeStart + i].IdxStart();
    splitOffset[i] = offCand[pairOffset(mrra.first, mrra.second)];
  }
  if (reachBase != nullptr) {
    for (unsigned int i = 0; i < backScale(1); i++) {
      reachBase[i] = nodePath[nodeStart + i].RelBase();
    }
  }
}


void Level::indexRestage(SamplePred *samplePred,
                         const SPPair &mrra,
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


void Level::indexRestage(SamplePred *samplePred,
                         const SPPair &mrra,
                         const Level *levelFront,
                         unsigned int bufIdx,
                         const unsigned int reachBase[],
                         unsigned int reachOffset[],
                         unsigned int splitOffset[]) {
  unsigned int startIdx, extent;
  getBounds(mrra, startIdx, extent);

  samplePred->indexRestage(nodeRel ? getFrontPath() : bottom->subtreePath(),
                           reachBase, mrra.second, bufIdx, startIdx, extent,
                           pathMask(),
                           reachBase == nullptr ? levelFront->isNodeRel() : true,
                           reachOffset,
                           splitOffset);
}
