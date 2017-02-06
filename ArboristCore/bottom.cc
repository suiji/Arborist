// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bottom.cc

   @brief Methods involving the most recently trained tree levels.

   @author Mark Seligman
 */

#include "bottom.h"
#include "bv.h"
#include "index.h"
#include "pretree.h"
#include "splitpred.h"
#include "samplepred.h"
#include "sample.h"
#include "splitsig.h"
#include "predblock.h"
#include "runset.h"
#include "rowrank.h"

#include <numeric>
#include <algorithm>

// Testing only:
//#include <iostream>
//using namespace std;
//#include <time.h>
//clock_t clock(void);


RelPath::RelPath(unsigned int _idxLive) : idxLive(_idxLive), relFront(std::vector<unsigned int>(idxLive)), pathFront(std::vector<unsigned char>(idxLive)), offFront(std::vector<unsigned int16_t>(idxLive)) {
  std::iota(relFront.begin(), relFront.end(), 0);
}


/**
   @brief Static entry for regression.
 */
Bottom *Bottom::FactoryReg(const PMTrain *_pmTrain, const RowRank *_rowRank, SamplePred *_samplePred, unsigned int _bagCount) {
  return new Bottom(_pmTrain, _samplePred, new SPReg(_pmTrain, _rowRank, _samplePred, _bagCount), _bagCount, _rowRank->SafeSize(_bagCount));
}


/**
   @brief Static entry for classification.
 */
Bottom *Bottom::FactoryCtg(const PMTrain *_pmTrain, const RowRank *_rowRank, SamplePred *_samplePred, const std::vector<SampleNode> &_sampleCtg, unsigned int _bagCount) {
  return new Bottom(_pmTrain, _samplePred, new SPCtg(_pmTrain, _rowRank, _samplePred, _sampleCtg, _bagCount), _bagCount, _rowRank->SafeSize(_bagCount));
}


/**
   @brief Class constructor.

   @param bagCount enables sizing of predicate bit vectors.

   @param splitCount specifies the number of splits to map.
 */
Bottom::Bottom(const PMTrain *_pmTrain, SamplePred *_samplePred, SplitPred *_splitPred, unsigned int _bagCount, unsigned int _stageSize) : nPred(_pmTrain->NPred()), nPredFac(_pmTrain->NPredFac()), bagCount(_bagCount), indexRel(false), prePath(std::vector<unsigned int>(_stageSize)), samplePath(new RelPath(bagCount)), splitPrev(0), frontCount(1), pmTrain(_pmTrain), samplePred(_samplePred), splitPred(_splitPred), splitSig(new SplitSig(nPred)), run(splitPred->Runs()), idxLive(bagCount) {
  std::vector<unsigned int> _history(0);
  history = std::move(_history);

  std::vector<unsigned char> _levelDelta(nPred);
  std::fill(_levelDelta.begin(), _levelDelta.end(), 0);
  levelDelta = std::move(_levelDelta);
  
  levelFront = new Level(1, nPred, bagCount, bagCount, indexRel);
  level.push_front(levelFront);

  levelFront->Ancestor(0, 0, bagCount);

  splitPred->SetBottom(this);
}


/**
   @brief Adds a new definition at the root level.

   @param nPred is the number of predictors.

   @return void.
 */
  // This is the only time that the denseCount is assigned outside of
  // restaging:
void Bottom::RootDef(unsigned int predIdx, unsigned int denseCount) {
  levelFront->Define(0, predIdx, IsFactor(predIdx) ? (pmTrain->FacCard(predIdx) + (denseCount > 0 ? 1 : 0)) : 0, 0, denseCount);
}

  
Level::Level(unsigned int _splitCount, unsigned int _nPred, unsigned int bagCount, unsigned int _idxLive, bool _indexRel) : nPred(_nPred), splitCount(_splitCount), noIndex(bagCount), idxLive(_idxLive), indexRel(_indexRel), defCount(0), del(0), indexAnc(std::vector<IndexAnc>(splitCount)), def(std::vector<MRRA>(splitCount * nPred)), relPath(new RelPath(idxLive)) {
  MRRA df;
  df.Undefine();
  std::fill(def.begin(), def.end(), df);
}


/**
   @brief Entry to spltting and restaging.

   @return vector of splitting signatures, possibly empty, for each node passed.
 */
const std::vector<class SSNode*> Bottom::Split(class Index *index, std::vector<IndexNode> &indexNode) {
  unsigned int supUnFlush = FlushRear();
  splitPred->LevelInit(index, indexNode, frontCount);

  Restage();

  // Source levels must persist through restaging ut allow path lookup.
  //
  for (unsigned int off = level.size() -1 ; off > supUnFlush; off--) {
    delete level[off];
    level.pop_back();
  }

  splitPred->Split(indexNode);

  std::vector<SSNode*> ssNode(frontCount);
  for (unsigned int levelIdx = 0; levelIdx < frontCount; levelIdx++) {
    ssNode[levelIdx] = splitSig->ArgMax(levelIdx, indexNode[levelIdx].MinInfo());
  }

  return ssNode;
}


/**
   @brief Flushes non-reaching definitions as well as those about
   to fall off the level deque.

   @return highest level not flushed.
 */
unsigned int Bottom::FlushRear() {
  unsigned int supUnFlush = level.size() - 1;

  // Capacity:  1 front level + 'pathMax' back levels.
  // If at capacity, every reaching definition should be flushed
  // to current level ut avoid falling off the deque.
  // Flushing prior to split assignment, rather than during, should
  // also save lookup time, as all definitions reaching from rear are
  // now at current level.
  //
  if ((level.size() > PathNode::pathMax)) {
    level.back()->Flush(this);
    supUnFlush--;
  }

  // Walks backward from rear, purging non-reaching definitions.
  // Stops when a level with no non-reaching nodes is encountered.
  //
  for (unsigned int off = supUnFlush; off > 0; off--) {
    if (!level[off]->NonreachPurge())
      break;
  }

  unsigned int backDef = 0;
  for (unsigned int off = supUnFlush; off > 0; off--) {
    backDef += level[off]->DefCount();
  }
  unsigned int thresh = backDef * efficiency;

  for (unsigned int off = supUnFlush; off > 0; off--) {
    if (level[off]->DefCount() <= thresh) {
      thresh -= level[off]->DefCount();
      level[off]->Flush(this);
      supUnFlush--;
    }
    else {
      break;
    }
  }

  return supUnFlush;
}


/**
   @brief Walks the definitions, purging those which no longer reach.

   @return true iff a definition was purged at this level.
 */
bool Level::NonreachPurge() {
  bool purged = false;
  for (unsigned int mrraIdx = 0; mrraIdx < splitCount; mrraIdx++) {
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
void Level::Flush(Bottom *bottom, bool forward) {
  for (unsigned int mrraIdx = 0; mrraIdx < splitCount; mrraIdx++) {
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      if (!Defined(mrraIdx, predIdx))
	continue;
      if (forward) {
	FlushDef(bottom, mrraIdx, predIdx);
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
void Level::FlushDef(Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx) {
  if (del == 0) // Already flushed to front level.
    return;

  unsigned int runCount, bufIdx;
  Consume(mrraIdx, predIdx, runCount, bufIdx);
  FrontDef(bottom, mrraIdx, predIdx, runCount, bufIdx);
  if (runCount != 1) // Singletons need not restage.
    bottom->ScheduleRestage(del, mrraIdx, predIdx, runCount, bufIdx);
}


void Level::FrontDef(Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx, unsigned int defRC, unsigned int sourceBit) {
  PathNode *pathStart = &pathNode[BackScale(mrraIdx)];
  unsigned int extent = BackScale(1);
  for (unsigned int path = 0; path < extent; path++) {
    bottom->AddDef(pathStart[path].Idx(), predIdx, defRC, 1 - sourceBit);
  }
}


void Bottom::ScheduleRestage(unsigned int del, unsigned int mrraIdx, unsigned int predIdx, unsigned int runCount, unsigned bufIdx) {
  SPPair mrra = std::make_pair(mrraIdx, predIdx);
  RestageCoord coord;
  coord.Init(mrra, del, runCount, bufIdx);
  restageCoord.push_back(coord);
}


/**
   @brief Looks up the ancestor cell built for the corresponding index
   node and adjusts start and extent values by corresponding dense parameters.
 */
void Level::Bounds(const SPPair &mrra, unsigned int &startIdx, unsigned int &extent) {
  indexAnc[mrra.first].Ref(startIdx, extent);
  (void) AdjustDense(mrra, startIdx, extent);
}


/**
   @brief Class finalizer.
 */
Bottom::~Bottom() {
  for (auto *defLevel : level) {
    defLevel->Flush(this, false);
    delete defLevel;
  }
  level.clear();

  delete samplePath;
  delete splitPred;
  delete splitSig;
}


/**
   @brief Heh heh.
 */
Level::~Level() {
  delete relPath;
}


/**
   @brief Ensures a pair will be restaged for the front level.

   @param runCount outputs the (unrestaged) run count of front-level definition.

   @param bufIdx outputs the front-level buffer index of the pair.

   @return true iff the front-level definition is a singleton.
 */
bool Bottom::ScheduleSplit(unsigned int levelIdx, unsigned int predIdx, unsigned int &runCount, unsigned int &bufIdx) {
  DefForward(levelIdx, predIdx);

  return !levelFront->Singleton(levelIdx, predIdx, runCount, bufIdx);
}


/**
   @brief Finds definition reaching coordinate pair at current level,
   flushing ancestor if necessary.

   @param levelIdx is the node index within current level.

   @param predIdx is the predictor index.

   @return void.
 */
void Bottom::DefForward(unsigned int levelIdx, unsigned int predIdx) {
  unsigned int del = ReachLevel(levelIdx, predIdx);
  level[del]->FlushDef(this, History(levelIdx, del), predIdx);
}


/**
   @brief Restages predictors and splits as pairs with equal priority.

   @return void, with side-effected restaging buffers.
 */
void Bottom::Restage() {
  int nodeIdx;

#pragma omp parallel default(shared) private(nodeIdx)
 {
#pragma omp for schedule(dynamic, 1)
    for (nodeIdx = 0; nodeIdx < int(restageCoord.size()); nodeIdx++) {
      Restage(restageCoord[nodeIdx]);
    }
  }

  restageCoord.clear();
}


/**
   @brief General, multi-level restaging.
 */
void Bottom::Restage(RestageCoord &rsCoord) {
  unsigned int del, runCount, bufIdx;
  SPPair mrra;
  rsCoord.Ref(mrra, del, runCount, bufIdx);

  SPNode *targ = Restage(mrra, bufIdx, del);
  RunCounts(targ, mrra, del);
}


/**
   @brief Restaging dispatch mechanism.
 */
SPNode *Bottom::Restage(SPPair mrra, unsigned int bufIdx, unsigned int del) {
  SPNode *targ;
  if (level[del]->IndexRel()) { // Source, target employ relative indexing.
    unsigned int reachOffset[1 << PathNode::pathMax];
    unsigned int reachBase[1 << PathNode::pathMax];
    OffsetClone(mrra, del, reachOffset, reachBase);
    if (IsDense(mrra, del)) {
      targ = RestageRelDense(reachOffset, reachBase, mrra, bufIdx, del);
    }
    else if (del == 1) {
      targ = RestageRelOne(reachOffset, reachBase, mrra, bufIdx);
    }
    else {
      targ = RestageRelGen(reachOffset, reachBase, mrra, bufIdx, del);
    }
  }
  else { //  Source employs sample indexing.  Target may or may not.
    unsigned int reachOffset[1 << PathNode::pathMax];
    OffsetClone(mrra, del, reachOffset);
    if (IsDense(mrra, del)) {
      targ = RestageSdxDense(reachOffset, mrra, bufIdx, del);
    }
    else if (del == 1) {
      targ = RestageSdxOne(reachOffset, mrra, bufIdx);
    }
    else {
      targ = RestageSdxGen(reachOffset, mrra, bufIdx, del);
    }
  }

  return targ;
}


/**
   @brief Precomputes path vector prior to restaging.  This is necessary
   in the case of dense ranks, as cell sizes are not derivable directly
   from index nodes.
 */
SPNode *Bottom::RestageRelDense(unsigned int reachOffset[], const unsigned int reachBase[], const SPPair &mrra, unsigned int bufIdx, unsigned int del) {
  SPNode *source, *targ;
  unsigned int *relIdxSource, *relIdxTarg;
  Buffers(mrra, bufIdx, source, relIdxSource, targ, relIdxTarg);

  unsigned int *ppBlock = &prePath[samplePred->StageOffset(mrra.second)];
  unsigned int pathCount[1 << PathNode::pathMax];
  for (unsigned int path = 0; path < level[del]->BackScale(1); path++) {
    pathCount[path] = 0;
  }

  // Decomposition into two paths adds ~5% performance penalty, but
  // is necessary for dense packing or for coprocessor loading.
  //
  RelPath *frontPath = FrontPath(del);
  unsigned int pathMask = level[del]->BackScale(1) - 1;
  unsigned int startIdx, extent;
  Bounds(mrra, del, startIdx, extent);
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int relSource = relIdxSource[idx];
    unsigned int path, offRel;
    if (frontPath->RelLive(relSource, path, offRel)) {
      path &= pathMask;
      ppBlock[idx] = path;
      pathCount[path]++;
      relIdxSource[idx] = reachBase[path] + offRel; // O.k. to overwrite.
    }
    else {
      ppBlock[idx] = PathNode::noPath;
    }
  }

  // Successors may or may not themselves be dense.
  level[del]->PackDense(startIdx, pathCount, levelFront, mrra, reachOffset);
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int path = ppBlock[idx];
    if (path != PathNode::noPath) {
      unsigned int destIdx = reachOffset[path]++;
      targ[destIdx] = source[idx];
      relIdxTarg[destIdx] = relIdxSource[idx];
    }
  }

  return targ;
}


/**
   @brief Precomputes path vector prior to restaging.  This is necessary
   in the case of dense ranks, as cell sizes are not derivable directly
   from index nodes.
 */
SPNode *Bottom::RestageSdxDense(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx, unsigned int del) {
  SPNode *source, *targ;
  unsigned int *relIdxSource, *relIdxTarg;
  Buffers(mrra, bufIdx, source, relIdxSource, targ, relIdxTarg);

  unsigned int *ppBlock = &prePath[samplePred->StageOffset(mrra.second)];
  unsigned int pathCount[1 << PathNode::pathMax];
  for (unsigned int path = 0; path < level[del]->BackScale(1); path++) {
    pathCount[path] = 0;
  }

  // Decomposition into two paths adds ~5% performance penalty, but
  // is necessary for dense packing or for coprocessor loading.
  //
  unsigned int pathMask = level[del]->BackScale(1) - 1;
  unsigned int startIdx, extent;
  Bounds(mrra, del, startIdx, extent);
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int relSource = relIdxSource[idx];
    unsigned int path;
    if (samplePath->FrontLive(relSource, path)) {
      path &= pathMask;
      ppBlock[idx] = path;
      pathCount[path]++;
      // RelFront() performs (slow) sIdx-to-relIdx mapping:  transition only.
      relIdxSource[idx] = indexRel ? samplePath->RelFront(relSource) : relSource;
    }
    else {
      ppBlock[idx] = PathNode::noPath;
    }
  }

  // Successors may or may not themselves be dense.
  level[del]->PackDense(startIdx, pathCount, levelFront, mrra, reachOffset);
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int path = ppBlock[idx];
    if (path != PathNode::noPath) {
      unsigned int destIdx = reachOffset[path]++;
      targ[destIdx] = source[idx];
      relIdxTarg[destIdx] = relIdxSource[idx];
    }
  }

  return targ;
}


/**
   @brief Sets the packed offsets for each successor.  Relies on Swiss Cheese
   index numbering ut prevent cell boundaries from crossing.

   @param idxLeft is the left-most index of the predecessor.

   @param pathCount inputs the path counts.

   @param reachOffset outputs the dense starting offsets.

   @return Count of explicit indices written.
 */
void Level::PackDense(unsigned int idxLeft, const unsigned int pathCount[], Level *levelFront, const SPPair &mrra, unsigned int reachOffset[]) const {
  const PathNode *pathPos = &pathNode[BackScale(mrra.first)];
  for (unsigned int path = 0; path < BackScale(1); path++) {
    unsigned int levelIdx, idxStart, idxCount;
    pathPos[path].Coords(levelIdx, idxStart, idxCount);
    if (levelIdx != noIndex) {
      unsigned int margin = idxStart - idxLeft;
      unsigned int idxLocal = pathCount[path];
      levelFront->SetDense(levelIdx, mrra.second, margin, idxCount - idxLocal);
      reachOffset[path] -= margin;
      idxLeft += idxLocal;
    }
  }
}


/**
   @brief General restaging, using relative indexing.

   @return void.
 */
SPNode *Bottom::RestageRelGen(unsigned int reachOffset[], const unsigned int reachBase[], const SPPair &mrra, unsigned int bufIdx, unsigned int del) {
  SPNode *source, *targ;
  unsigned int *relIdxSource, *relIdxTarg;
  Buffers(mrra, bufIdx, source, relIdxSource, targ, relIdxTarg);

  unsigned int startIdx, extent;
  Bounds(mrra, del, startIdx, extent);
  RelPath *frontPath = FrontPath(del);
  unsigned int pathMask = level[del]->BackScale(1) - 1;
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int relSource = relIdxSource[idx]; // previous relTarg
    unsigned int path, offRel;
    if (frontPath->RelLive(relSource, path, offRel)) {
      path &= pathMask;
      unsigned int destIdx = reachOffset[path]++;
      targ[destIdx] = source[idx];
      relIdxTarg[destIdx] = reachBase[path] + offRel;//relTarg;
    }
  }

  return targ;
}


/**
   @brief General restaging, using relative indexing.

   @return void.
 */
SPNode *Bottom::RestageSdxGen(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx, unsigned int del) {
  SPNode *source, *targ;
  unsigned int *relIdxSource, *relIdxTarg;
  Buffers(mrra, bufIdx, source, relIdxSource, targ, relIdxTarg);

  unsigned int startIdx, extent;
  Bounds(mrra, del, startIdx, extent);
  unsigned int pathMask = level[del]->BackScale(1) - 1;
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int relSource = relIdxSource[idx];
    unsigned int path;
    if (samplePath->FrontLive(relSource, path)) {
      unsigned int destIdx = reachOffset[path & pathMask]++;
      targ[destIdx] = source[idx];
      // RelFront() performs (slow) sIdx-to-relIdx mapping:  transition only.
      relIdxTarg[destIdx] = indexRel ? samplePath->RelFront(relSource) : relSource;
    }
  }

  return targ;
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
    reachOffset[i] = pathNode[nodeStart + i].IdxStart();
  }
  if (reachBase != 0) {
    for (unsigned int i = 0; i < BackScale(1); i++) {
      reachBase[i] = pathNode[nodeStart + i].RelBase();
    }
  }
}


/**
   @brief Restaging using relative indexing, specialized for single-level
   case.
 */
SPNode *Bottom::RestageRelOne(unsigned int reachOffset[], const unsigned int reachBase[], const SPPair &mrra, unsigned int bufIdx) {
  SPNode *source, *targ;
  unsigned int *relIdxSource, *relIdxTarg;
  Buffers(mrra, bufIdx, source, relIdxSource, targ, relIdxTarg);

  unsigned int startIdx, extent;
  Bounds(mrra, 1, startIdx, extent);
  RelPath *frontPath = FrontPath(1);
  unsigned int pathMask = level[1]->BackScale(1) - 1;
  unsigned int leftOff = reachOffset[0];
  unsigned int rightOff = reachOffset[1];
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int relSource = relIdxSource[idx];
    unsigned int path, offRel;
    if (frontPath->RelLive(relSource, path, offRel)) {
      path &= pathMask;
      unsigned int destIdx = path == 0 ? leftOff++ : rightOff++;
      targ[destIdx] = source[idx];
      relIdxTarg[destIdx] = reachBase[path] + offRel;
    }
  }

  reachOffset[0] = leftOff;
  reachOffset[1] = rightOff;

  return targ;
}


/**
   @brief Restaging using relative indexing, specialized for single-level
   case.
 */
SPNode *Bottom::RestageSdxOne(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx) {
  SPNode *source, *targ;
  unsigned int *relIdxSource, *relIdxTarg;
  Buffers(mrra, bufIdx, source, relIdxSource, targ, relIdxTarg);

  unsigned int startIdx, extent;
  Bounds(mrra, 1, startIdx, extent);
  unsigned int pathMask = level[1]->BackScale(1) - 1;
  unsigned int leftOff = reachOffset[0];
  unsigned int rightOff = reachOffset[1];
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int relSource = relIdxSource[idx];
    unsigned int path;
    if (samplePath->FrontLive(relSource, path)) {
      unsigned int destIdx = (path & pathMask) == 0 ? leftOff++ : rightOff++;
      targ[destIdx] = source[idx];
      // RelFront() performs (slow) sIdx-to-relIdx mapping:  transition only.
      relIdxTarg[destIdx] = indexRel ? samplePath->RelFront(relSource) : relSource;
    }
  }

  reachOffset[0] = leftOff;
  reachOffset[1] = rightOff;

  return targ;
}


/**
   @brief Sets buffer addresses from source coordinates.

   @param mrra holds the ancestor's coordinates.

   @param bufIdx is the index of the source buffer.

   @return void.
 */
void Bottom::Buffers(const SPPair &mrra, unsigned int bufIdx, SPNode *&source, unsigned int *&relIdxSource, SPNode *&targ, unsigned int *&relIdxTarg) const {
  samplePred->Buffers(mrra.second, bufIdx, source, relIdxSource, targ, relIdxTarg);
}



/**
   @brief Sets dense count on target MRRA and, if singleton, sets run count to
   unity.

   @return void.
 */
void Level::RunCounts(const SPNode targ[], const SPPair &mrra, const Bottom *bottom) const {
  unsigned int predIdx = mrra.second;
  const PathNode *pathPos = &pathNode[BackScale(mrra.first)];
  for (unsigned int path = 0; path < BackScale(1); path++) {
    unsigned int levelIdx, idxStart, idxCount;
    pathPos[path].Coords(levelIdx, idxStart, idxCount);
    if (levelIdx != noIndex) {
      bottom->SetRuns(levelIdx, predIdx, idxStart, idxCount, targ);
    }
  }
}


/**
   @brief Sets dense count and conveys tied cell as single run.

   @return void.
 */
void Level::SetRuns(const Bottom *bottom, unsigned int levelIdx, unsigned int predIdx, unsigned int idxStart, unsigned int idxCount, const SPNode *targ) {
  MRRA &reach = def[PairOffset(levelIdx, predIdx)];
  unsigned int denseCount = reach.AdjustDense(idxStart, idxCount);
  if (idxCount == 0) { // all indices implicit.
    reach.SetRunCount(1);
  }
  else if (targ->IsRun(idxStart, idxStart + idxCount - 1)) {
    if (bottom->IsFactor(predIdx)) { // Factor:  singleton or doubleton.
      unsigned int runCount = 1 + (denseCount > 0 ? 1 : 0);
      reach.SetRunCount(runCount);
    }
    else if (denseCount == 0) { // Numeric:  only singletons tracked.
      reach.SetRunCount(1);
    }
  }
}


bool Bottom::IsFactor(unsigned int predIdx) const {
  return pmTrain->IsFactor(predIdx);
}


/**
   @brief Invoked from splitting methods to precipitate creation of signature
   for candidate split.

   @return void.
*/
void Bottom::SSWrite(unsigned int levelIdx, unsigned int predIdx, unsigned int setPos, unsigned int bufIdx, const NuxLH &nux) const {
  splitSig->Write(levelIdx, predIdx, setPos, bufIdx, nux);
}


/**
   @brief Sets level data structures within attendant objects.

   @return void.
 */
void Bottom::LevelInit() {
  splitSig->LevelInit(frontCount);
}


/**
   @brief Clears level data structures within attendant objects.

   @return void.
 */
void Bottom::LevelClear() {
  splitPred->LevelClear();
  splitSig->LevelClear();
}


/**
   @brief Allocates storage for upcoming level and ensures a safe interval
   during which the contents of the current level's nodes can be inherited
   by the next level.

   @param splitNext is the number of nodes in the upcoming level.

   @param idxLive is the number of live sample indices upcoming.

   @return cout of live indices in upcoming level.
 */
void Bottom::Overlap(unsigned int splitNext, unsigned int _idxLive, unsigned int idxMax) {
  idxLive = _idxLive;
  splitPrev = frontCount;
  frontCount = splitNext;

  if (!indexRel) { // Sticky.
    indexRel = RelPath::Relable(bagCount, idxMax);
  }

  levelFront = new Level(frontCount, nPred, bagCount, idxLive, indexRel);
  level.push_front(levelFront);

  historyPrev = std::move(history);
  std::vector<unsigned int> _history(frontCount * (level.size()-1));
  history = std::move(_history);

  deltaPrev = std::move(levelDelta);
  std::vector<unsigned char> _levelDelta(frontCount * nPred);
  levelDelta = std::move(_levelDelta);

  // Recomputes paths reaching from non-front levels.
  //
  for(unsigned int i = 1; i < level.size(); i++) {
    level[i]->Paths();
  }
}


/**
  @brief Initializes reaching paths:  back levels 1 and higher.

  @return void.
 */
void Level::Paths() {
  del++;
  std::vector<unsigned int> live(splitCount);
  std::vector<PathNode> path(BackScale(splitCount));
  PathNode node;
  node.Init(noIndex, 0, 0, 0);
  std::fill(path.begin(), path.end(), node);
  std::fill(live.begin(), live.end(), 0);
  
  pathNode = move(path);
  liveCount = move(live);
}


void Level::PathInit(const Bottom *bottom, unsigned int levelIdx, unsigned int path, unsigned int start, unsigned int extent, unsigned int relBase) {
  unsigned int mrraIdx = bottom->History(levelIdx, del);
  unsigned int pathOff = BackScale(mrraIdx);
  unsigned int pathBits = path & (BackScale(1) - 1);
  pathNode[pathOff + pathBits].Init(levelIdx, start, extent, relBase);
  liveCount[mrraIdx]++;
}


/**
     @brief 

     @return void.
*/
void Bottom::PathUpdate(const std::vector<IndexNode> &indexNode, const PreTree *preTree, unsigned int levelWidth) {
  std::vector<unsigned int> relIdx(levelWidth + 1);
  std::fill(relIdx.begin(), relIdx.end(), idxLive);
  unsigned int idxTot = 0;
  unsigned int idx = 0;
  for (auto node : indexNode) {
    unsigned int parIdx, path, lhStart, idxCount, ptId;
    node.PathFields(parIdx, path, lhStart, idxCount, ptId);
    ReachingPath(parIdx, path, idx++, lhStart, idxCount, idxTot);
    unsigned int levelOff = preTree->LevelOffset(ptId);
    relIdx[levelOff] = idxTot;
    idxTot += idxCount;
  }

  const std::vector<unsigned int> relBase(relIdx);
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    bool isLeft;
    unsigned int levelOff = preTree->SampleOffset(sIdx, isLeft);
    FrontUpdate(sIdx, isLeft, relBase[levelOff], relIdx[levelOff]);
  }

  BackUpdate();
}


/**
   @brief Consumes all fields in current NodeCache item relevant to restaging.

   @param par is the index of the parent.

   @param path is a unique path identifier.

   @param levelIdx is the index of the heir.

   @param start is the cell starting index.

   @param extent is the index count.

   @return void.
*/
void Bottom::ReachingPath(unsigned int par, unsigned int path, unsigned int levelIdx, unsigned int start, unsigned int extent, unsigned int relBase) {
  for (unsigned int backLevel = 0; backLevel < level.size() - 1; backLevel++) {
    history[levelIdx + frontCount * backLevel] = backLevel == 0 ? par : historyPrev[par + splitPrev * (backLevel - 1)];
  }

  Inherit(levelIdx, par);
  levelFront->Ancestor(levelIdx, start, extent);
  
  // Places <levelIdx, start> pair at appropriate position in every
  // reaching path.
  //
  for (unsigned int i = 1; i < level.size(); i++) {
    level[i]->PathInit(this, levelIdx, path, start, extent, relBase);
  }
}


  /**
     @brief Reassigns a single slot of the first level's relative path map.

     @param sIdx is the slot to reassign.

     @param relIdx is the new slot relative index.

     @param isLeft indicates whether the reaching path terminates in a
     left branch.

     @return void.
   */
void Bottom::FrontUpdate(unsigned int sIdx, bool isLeft, unsigned int relBase, unsigned int &relIdx) const {
  bool isLive = relIdx != idxLive;

  if (!indexRel) {
    samplePath->FrontifySdx(sIdx, relIdx, isLive, isLeft);
  }
  else {
    samplePath->FrontifyRel(FrontPath(1), sIdx, relIdx, relBase, isLive, isLeft);
    if (isLive) {
      samplePred->Rel2Sample(relIdx, sIdx);
    }
  }  

  relIdx += isLive ? 1 : 0;
}


/**
   @brief Passes level 1's relative path map back to earlier levels for
   remapping.

    @return void.
 */
void Bottom::BackUpdate() const {
  if (indexRel) {
    for (unsigned int i = 2; i < level.size(); i++) {
      level[i]->BackUpdate(FrontPath(1));
    }
  }
}


