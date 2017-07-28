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
#include "splitpred.h"

//#include <iostream>
//using namespace std;

unsigned int Level::predFixed = 0;
const double *Level::predProb = 0;

Level::Level(unsigned int _nSplit, unsigned int _nPred, const std::vector<unsigned int> &_denseIdx, unsigned int _nPredDense, unsigned int bagCount, unsigned int _idxLive, bool _nodeRel, Bottom *_bottom, SamplePred *_samplePred) : nPred(_nPred), denseIdx(_denseIdx), nPredDense(_nPredDense), nSplit(_nSplit), noIndex(bagCount), idxLive(_idxLive), nodeRel(_nodeRel), bottom(_bottom), samplePred(_samplePred), defCount(0), del(0), indexAnc(std::vector<IndexAnc>(nSplit)), def(std::vector<MRRA>(nSplit * nPred)), denseCoord(std::vector<DenseCoord>(nSplit * nPredDense)), relPath(new IdxPath(idxLive)), offCand(std::vector<unsigned int>(nSplit * nPred)) {
  MRRA df;
  df.Init();
  // Coprocessor only.
  std::fill(def.begin(), def.end(), df);
  std::fill(offCand.begin(), offCand.end(), bagCount);
}

void Level::Immutables(unsigned int _predFixed, const double _predProb[]) {
  predFixed = _predFixed;
  predProb = _predProb;
}


void Level::DeImmutables() {
  predFixed = 0;
}


/**
   @brief Walks the definitions, purging those which no longer reach.

   @return true iff a definition was purged at this level.
 */
bool Level::NonreachPurge() {
  bool purged = false;
  for (unsigned int mrraIdx = 0; mrraIdx < nSplit; mrraIdx++) {
    if (liveCount[mrraIdx] == 0) {
      for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
        Undefine(mrraIdx, predIdx); // Harmless if already undefined.
        purged = true;
      }
    }
  }

  return purged;
}

/**
   @brief Moves entire level's defnitions to restaging schedule.

   @param bottom is the active bottom state.

   @return void.
 */
void Level::Flush(bool forward) {
  for (unsigned int mrraIdx = 0; mrraIdx < nSplit; mrraIdx++) {
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      if (!Defined(mrraIdx, predIdx))
	continue;
      if (forward) {
	FlushDef(mrraIdx, predIdx);
      }
      else {
	Undefine(mrraIdx, predIdx);
      }
    }
  }
}


/**
   @brief Removes definition from a back level and builds definition
   for each descendant reached in current level.

   @param mrra is the coordinate pair of the ancestor to flush.

   @return void.
 */
void Level::FlushDef(unsigned int mrraIdx, unsigned int predIdx) {
  if (del == 0) // Already flushed to front level.
    return;

  unsigned int bufIdx;
  bool singleton;
  Consume(mrraIdx, predIdx, bufIdx, singleton);
  FrontDef(mrraIdx, predIdx, bufIdx, singleton);
  if (!singleton)
    bottom->ScheduleRestage(del, mrraIdx, predIdx, bufIdx);
}


void Level::FrontDef(unsigned int mrraIdx, unsigned int predIdx, unsigned int bufIdx, bool singleton) {
  unsigned int pathStart = BackScale(mrraIdx);
  for (unsigned int path = 0; path < BackScale(1); path++) {
    bottom->AddDef(nodePath[pathStart + path].Idx(), predIdx, 1 - bufIdx, singleton);
  }
}


/**
   @brief Looks up the ancestor cell built for the corresponding index
   node and adjusts start and extent values by corresponding dense parameters.
 */
void Level::Bounds(const SPPair &mrra, unsigned int &startIdx, unsigned int &extent) {
  indexAnc[mrra.first].Ref(startIdx, extent);
  (void) AdjustDense(mrra.first, mrra.second, startIdx, extent);
}


/**
   @brief Heh heh.
 */
Level::~Level() {
  delete relPath;
}


/**
   @brief Sets the packed offsets for each successor.  Relies on Swiss Cheese
   index numbering ut prevent cell boundaries from crossing.

   @param idxLeft is the left-most index of the predecessor.

   @param pathCount inputs the counts along each reaching path.

   @param reachOffset outputs the dense starting offsets.

   @return void.
 */
void Level::PackDense(unsigned int idxLeft, const unsigned int pathCount[], Level *levelFront, const SPPair &mrra, unsigned int reachOffset[]) const {
  const NodePath *pathPos = &nodePath[BackScale(mrra.first)];
  for (unsigned int path = 0; path < BackScale(1); path++) {
    unsigned int levelIdx, idxStart, extent;
    pathPos[path].Coords(levelIdx, idxStart, extent);
    if (levelIdx != noIndex) {
      unsigned int margin = idxStart - idxLeft;
      unsigned int extentDense = pathCount[path];
      levelFront->SetDense(levelIdx, mrra.second, extent - extentDense, margin);
      reachOffset[path] -= margin;
      idxLeft += extentDense;
    }
  }
}


/**
   @brief Clones offsets along path reaching from ancestor node.

   @param mrra is an MRRA coordinate.

   @param reachOffset holds the starting offset positions along the path.

   @return path origin at the index passed.
 */
void Level::OffsetClone(const SPPair &mrra, unsigned int reachOffset[], unsigned int reachBase[]) {
  unsigned int nodeStart = BackScale(mrra.first);
  for (unsigned int i = 0; i < BackScale(1); i++) {
    reachOffset[i] = nodePath[nodeStart + i].IdxStart();
  }
  if (reachBase != nullptr) {
    for (unsigned int i = 0; i < BackScale(1); i++) {
      reachBase[i] = nodePath[nodeStart + i].RelBase();
    }
  }
}


/**
   @brief Sets dense count on target MRRA and, if singleton, sets run count to
   unity.

   @return void.
 */
void Level::RunCounts(Bottom *bottom, const SPPair &mrra, const unsigned int pathCount[], const unsigned int rankCount[]) const {
  unsigned int predIdx = mrra.second;
  const NodePath *pathPos = &nodePath[BackScale(mrra.first)];
  for (unsigned int path = 0; path < BackScale(1); path++) {
    unsigned int levelIdx, idxStart, extent;
    pathPos[path].Coords(levelIdx, idxStart, extent);
    if (levelIdx != noIndex) {
      bottom->SetRunCount(levelIdx, predIdx, pathCount[path] != extent, rankCount[path]);
    }
  }
}


/**
   @brief Sets the definition's heritable singleton bit and clears the
   current level's splitable bit.

   @return void.
*/
void Level::SetSingleton(unsigned int levelIdx, unsigned int predIdx) {
  def[PairOffset(levelIdx, predIdx)].SetSingleton();
}



/**
   @brief Revises node-relative indices, as appropriae.  Irregular,
   but data locality improves with tree depth.

   @param one2Front maps first level to front indices.

   @return true iff level employs node-relative indexing.
 */
bool Level::Backdate(const IdxPath *one2Front) {
  if (!nodeRel)
    return false;

  relPath->Backdate(one2Front);
  return true;
}


/**
  @brief Initializes reaching paths:  back levels 1 and higher.

  @return void.
 */
void Level::Paths() {
  del++;
  std::vector<unsigned int> live(nSplit);
  std::vector<NodePath> path(BackScale(nSplit));
  NodePath np;
  np.Init(noIndex, 0, 0, 0);
  std::fill(path.begin(), path.end(), np);
  std::fill(live.begin(), live.end(), 0);
  
  nodePath = move(path);
  liveCount = move(live);
}


void Level::SetExtinct(unsigned int idx) {
  relPath->SetExtinct(idx);
}


void Level::PathInit(const Bottom *bottom, unsigned int splitIdx, unsigned int path, unsigned int start, unsigned int extent, unsigned int relBase) {
  unsigned int mrraIdx = bottom->History(this, splitIdx);
  unsigned int pathOff = BackScale(mrraIdx);
  unsigned int pathBits = path & PathMask();
  nodePath[pathOff + pathBits].Init(splitIdx, start, extent, relBase);
  liveCount[mrraIdx]++;
}


/**
   @brief Sets path, target and node-relative offse.

   @return void.
 */
void Level::SetLive(unsigned int idx, unsigned int path, unsigned int targIdx, unsigned int ndBase) {
  relPath->SetLive(idx, path, targIdx, targIdx - ndBase);
}


/**
   @brief Schedules a non-singleton for splitting.

   @param splitIdx

   @param predIdx

   @return true iff pair scheduled for splitting.
 */
bool Level::Preschedule(SplitPred *splitPred, unsigned int splitIdx, unsigned int predIdx, unsigned int extent, unsigned int &spanCand) {
  bottom->ReachFlush(splitIdx, predIdx);

  unsigned int bufIdx;
  if (!Singleton(splitIdx, predIdx, bufIdx)) {
    splitPred->Preschedule(splitIdx, predIdx, bufIdx);
    offCand[PairOffset(splitIdx, predIdx)] = spanCand;
    spanCand += extent;
    return true;
  }
  return false;
}


/**
   @brief Determines whether a cell is suitable for splitting.  It may,
   for example, have become unsplitiable as a result of restaging's
   precipitating a singleton instance.

   @param levelIdx is the split index.

   @param predIdx is the predictor index.

   @param runCount outputs the run count iff not singleton.

   @return true iff candidate remains splitable.
 */
bool Level::ScheduleSplit(unsigned int splitIdx, unsigned int predIdx, unsigned int &rCount) const {
  if (!Singleton(splitIdx, predIdx)) {
    rCount = bottom->RunCount(splitIdx, predIdx);
    return true;
  }
  else {
    return false;
  }
}


// COPROC:
/**
   @brief Clones offsets along path reaching from ancestor node.

   @param mrra is an MRRA coordinate.

   @param reachOffset holds the starting offset positions along the path.

   @return path origin at the index passed.
 */
void Level::OffsetClone(const SPPair &mrra, unsigned int reachOffset[], unsigned int splitOffset[], unsigned int reachBase[]) {
  unsigned int nodeStart = BackScale(mrra.first);
  for (unsigned int i = 0; i < BackScale(1); i++) {
    reachOffset[i] = nodePath[nodeStart + i].IdxStart();
    splitOffset[i] = offCand[PairOffset(mrra.first, mrra.second)];
  }
  if (reachBase != nullptr) {
    for (unsigned int i = 0; i < BackScale(1); i++) {
      reachBase[i] = nodePath[nodeStart + i].RelBase();
    }
  }
}


/**
   @brief Signals SplitPred to schedule splitable pairs.

   @param unsplitable lists unsplitable nodes.

   @return void.
*/
void Level::Candidates(const IndexLevel *index, SplitPred *splitPred) {
    // TODO:  Pre-empt overflow by walking wide subtrees depth-first.
  int cellCount = nSplit * nPred;

  double *ruPred = new double[cellCount];
  CallBack::RUnif(cellCount, ruPred);

  BHPair *heap;
  if (predFixed > 0)
    heap = new BHPair[cellCount];
  else
    heap = 0;

  unsigned int spanCand = 0;
  for (unsigned int splitIdx = 0; splitIdx < nSplit; splitIdx++) {
    unsigned int splitOff = splitIdx * nPred;
    if (index->Unsplitable(splitIdx)) { // Node cannot split.
      continue;
    }
    else if (predFixed == 0) { // Probability of predictor splitable.
      CandidateProb(splitPred, splitIdx, &ruPred[splitOff], index->Extent(splitIdx), spanCand);
    }
    else { // Fixed number of predictors splitable.
      CandidateFixed(splitPred, splitIdx, &ruPred[splitOff], &heap[splitOff], index->Extent(splitIdx), spanCand);
    }
  }
  SetSpan(spanCand);

  if (heap != 0)
    delete [] heap;
  delete [] ruPred;
}


/**
   @brief Set splitable flag by Bernoulli sampling.

   @param splitIdx is the level-relative node index.

   @param ruPred is a vector of uniformly-sampled variates.

   @return total number of splitting candidates for the node.
 */
void Level::CandidateProb(SplitPred *splitPred, unsigned int splitIdx, const double ruPred[], unsigned int extent, unsigned int &spanCand) {
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    if (ruPred[predIdx] < predProb[predIdx]) {
      (void) Preschedule(splitPred, splitIdx, predIdx, extent, spanCand);
    }
  }
}

 
/**
   @brief Sets splitable flag for a fixed number of predictors.

   @param ruPred is a vector of uniformly-sampled variates.

   @param heap orders probability-weighted variates.

   @param extent is the index count of the splitting node.

   @param offSplit accumulates offsets for splitable pairs.

   @return total number of candidates for the node.
 */
void Level::CandidateFixed(SplitPred *splitPred, unsigned int splitIdx, const double ruPred[], BHPair heap[], unsigned int extent, unsigned int &spanCand) {
  // Inserts negative, weighted probability value:  choose from lowest.
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    BHeap::Insert(heap, predIdx, -ruPred[predIdx] * predProb[predIdx]);
  }

  // Pops 'predFixed' items in order of increasing value.
  unsigned int schedCount = 0;
  for (unsigned int heapSize = nPred; heapSize > 0; heapSize--) {
    unsigned int predIdx = BHeap::SlotPop(heap, heapSize - 1);
    schedCount += Preschedule(splitPred, splitIdx, predIdx, extent, spanCand) ? 1 : 0;
    if (schedCount == predFixed)
      break;
  }
}

void Level::Restage(SPPair &mrra, Level *levelFront, unsigned int bufIdx) {
  unsigned int reachOffset[1 << NodePath::pathMax];
  if (nodeRel) { // Both levels employ node-relative indexing.
    unsigned int reachBase[1 << NodePath::pathMax];
    OffsetClone(mrra, reachOffset, reachBase);
    Restage(mrra, levelFront, bufIdx, reachBase, reachOffset);
  }
  else { // Source level employs subtree indexing.  Target may or may not.
    OffsetClone(mrra, reachOffset, nullptr);
    Restage(mrra, levelFront, bufIdx, nullptr, reachOffset);
  }
}


/**
   @brief Precomputes path vector prior to restaging.  This is necessary
   in the case of dense ranks, as cell sizes are not derivable directly
   from index nodes.

   Decomposition into two paths adds ~5% performance penalty, but
   appears necessary for dense packing or for coprocessor loading.
 */
void Level::Restage(const SPPair &mrra, Level *levelFront, unsigned int bufIdx, const unsigned int reachBase[], unsigned int reachOffset[]) {
  unsigned int startIdx, extent;
  Bounds(mrra, startIdx, extent);

  unsigned int pathCount[1 << NodePath::pathMax];
  for (unsigned int path = 0; path < BackScale(1); path++) {
    pathCount[path] = 0;
  }

  unsigned int predIdx = mrra.second;
  samplePred->Prepath(nodeRel ?  FrontPath() : bottom->STPath(), reachBase, predIdx, bufIdx, startIdx, extent, PathMask(), reachBase == nullptr ? levelFront->NodeRel() : true, pathCount);

  // Successors may or may not themselves be dense.
  if (Dense(mrra.first, mrra.second)) {
    PackDense(startIdx, pathCount, levelFront, mrra, reachOffset);
  }

  unsigned int rankPrev[1 << NodePath::pathMax];
  unsigned int rankCount[1 << NodePath::pathMax];
  for (unsigned int path = 0; path < BackScale(1); path++) {
    rankPrev[path] = bottom->NoRank();
    rankCount[path] = 0;
  }
  samplePred->RestageRank(predIdx, bufIdx, startIdx, extent, reachOffset, rankPrev, rankCount);
  RunCounts(bottom, mrra, pathCount, rankCount);
}


// Coprocessor variants:

// Restages destination and index vectors only.
//
void Level::IndexRestage(SPPair &mrra, const Level *levelFront, unsigned int bufIdx) {
  unsigned int reachOffset[1 << NodePath::pathMax];
  unsigned int splitOffset[1 << NodePath::pathMax];
  if (nodeRel) { // Both levels employ node-relative indexing.
    unsigned int reachBase[1 << NodePath::pathMax];
    OffsetClone(mrra, reachOffset, splitOffset, reachBase);
    IndexRestage(mrra, levelFront, bufIdx, reachBase, reachOffset, splitOffset);
  }
  else { // Source level employs subtree indexing.  Target may or may not.
    OffsetClone(mrra, reachOffset, splitOffset);
    IndexRestage(mrra, levelFront, bufIdx, nullptr, reachOffset, splitOffset);
  }
}


void Level::IndexRestage(const SPPair &mrra, const Level *levelFront, unsigned int bufIdx, const unsigned int reachBase[], unsigned int reachOffset[], unsigned int splitOffset[]) {
  unsigned int startIdx, extent;
  Bounds(mrra, startIdx, extent);

  samplePred->IndexRestage(nodeRel ? FrontPath() : bottom->STPath(), reachBase, mrra.second, bufIdx, startIdx, extent, PathMask(), reachBase == nullptr ? levelFront->NodeRel() : true, reachOffset, splitOffset);
}

