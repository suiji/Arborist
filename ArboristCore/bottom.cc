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
#include "splitpred.h"
#include "samplepred.h"
#include "sample.h"
#include "splitsig.h"
#include "predblock.h"
#include "runset.h"
#include "rowrank.h"

// Testing only:
//#include <iostream>
//using namespace std;
//#include <time.h>
//clock_t clock(void);



/**
   @brief Static entry for regression.
 */
Bottom *Bottom::FactoryReg(const RowRank *_rowRank, SamplePred *_samplePred, unsigned int _bagCount) {
  return new Bottom(_samplePred, new SPReg(_rowRank, _samplePred, _bagCount), _bagCount, PBTrain::NPred(), PBTrain::NPredFac());
}


/**
   @brief Static entry for classification.
 */
Bottom *Bottom::FactoryCtg(const RowRank *_rowRank, SamplePred *_samplePred, const std::vector<SampleNode> &_sampleCtg, unsigned int _bagCount) {
  return new Bottom(_samplePred, new SPCtg(_rowRank, _samplePred, _sampleCtg, _bagCount), _bagCount, PBTrain::NPred(), PBTrain::NPredFac());
}


/**
   @brief Class constructor.

   @param bagCount enables sizing of predicate bit vectors.

   @param splitCount specifies the number of splits to map.
 */
Bottom::Bottom(SamplePred *_samplePred, SplitPred *_splitPred, unsigned int _bagCount, unsigned int _nPred, unsigned int _nPredFac) : nPred(_nPred), nPredFac(_nPredFac), bagCount(_bagCount), samplePath(new SamplePath[bagCount]), splitPrev(0), frontCount(1), bvLeft(new BV(bagCount)), bvDead(new BV(bagCount)), samplePred(_samplePred), splitPred(_splitPred), splitSig(new SplitSig()), run(splitPred->Runs()) {
  std::vector<unsigned int> _history(0);
  history = std::move(_history);

  std::vector<unsigned char> _levelDelta(nPred);
  std::fill(_levelDelta.begin(), _levelDelta.end(), 0);
  levelDelta = std::move(_levelDelta);
  
  levelFront = new Level(1, nPred, bagCount);
  level.push_front(levelFront);

  levelFront->Node(0, 0, bagCount, bagCount);

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
  levelFront->Define(0, predIdx, PBTrain::FacCard(predIdx) + (denseCount > 0 ? 1 : 0), 0, denseCount);
}

  
Level::Level(unsigned int _splitCount, unsigned int _nPred, unsigned int _noIndex) : nPred(_nPred), splitCount(_splitCount), noIndex(_noIndex), defCount(0), del(0) {
  MRRA df;
  df.Undefine();

  cell.reserve(splitCount);
  def.reserve(splitCount * nPred);
  for (unsigned int i = 0; i < splitCount * nPred; i++) {
    def[i] = df;
  }
}


/**
   @brief Entry to spltting and restaging.

   @return vector of splitting signatures, possibly empty, for each node passed.
 */
const std::vector<class SSNode*> Bottom::Split(class Index *index, class IndexNode indexNode[]) {
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
  if ((level.size() > pathMax)) {
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


void Level::CellBounds(const SPPair &mrra, unsigned int &startIdx, unsigned int &extent) {
  return cell[mrra.first].Ref(startIdx, extent);
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

  delete bvLeft;
  delete bvDead;
  delete [] samplePath;
  delete splitPred;
  delete splitSig;
}


Level::~Level() {
}


void Bottom::PathLeft(unsigned int sIdx) const {
   samplePath[sIdx].PathLeft();
   bvLeft->SetBit(sIdx, true);
}


void Bottom::PathRight(unsigned int sIdx) const {
    samplePath[sIdx].PathRight();
    bvLeft->SetBit(sIdx, false);
}


void Bottom::PathExtinct(unsigned int sIdx) const {
  samplePath[sIdx].PathExtinct();
  bvDead->SetBit(sIdx, true);
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
  unsigned int reachOffset[1 << pathMax];
  unsigned int del, runCount, bufIdx;
  SPPair mrra;
  rsCoord.Ref(mrra, del, runCount, bufIdx);
  OffsetClone(mrra, del, reachOffset);

  SPNode *targ;
  if (del == 1) {
    targ = RestageOne(reachOffset, mrra, bufIdx);
  }
  else {
    targ = RestageIrr(reachOffset, mrra, bufIdx, del);
  }

  RunCounts(reachOffset, targ, mrra, del);
}


/**
   @brief Irregular restaging, suitable for smaller sample sets.

   @return void.
 */
SPNode *Bottom::RestageIrr(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx, unsigned int del) {
  SPNode *source, *targ;
  unsigned int *sIdxSource, *sIdxTarg;
  Buffers(mrra, bufIdx, source, sIdxSource, targ, sIdxTarg);

  unsigned int startIdx, extent;
  CellBounds(del, mrra, startIdx, extent);
  extent -= DenseCount(mrra.first, mrra.second, del);
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int sIdx = sIdxSource[idx];
    if (!bvDead->TestBit(sIdx)) {  // Irregular access:  1 bit.
      unsigned path = Path(sIdx, del); // Irregular access:  8 bits.
      unsigned int destIdx = reachOffset[path]++;
      targ[destIdx] = source[idx];
      sIdxTarg[destIdx] = sIdx;
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
void Level::OffsetClone(const SPPair &mrra, unsigned int reachOffset[]) {
  unsigned int nodeStart = BackScale(mrra.first);
  for (unsigned int i = 0; i < BackScale(1); i++) {
    reachOffset[i] = pathNode[nodeStart + i].Offset();
  }
}


/**
   @brief Specialized for two-path case, bypasses stack array.

 */
SPNode *Bottom::RestageOne(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx) {
  SPNode *source, *targ;
  unsigned int *sIdxSource, *sIdxTarg;
  Buffers(mrra, bufIdx, source, sIdxSource, targ, sIdxTarg);
  unsigned int startIdx, idxCount;
  CellBounds(1, mrra, startIdx, idxCount);
  idxCount -= DenseCount(mrra.first, mrra.second, 1);
  unsigned int leftOff = reachOffset[0];
  unsigned int rightOff = reachOffset[1];
  for (unsigned int idx = startIdx; idx < startIdx + idxCount; idx++) {
    unsigned int sIdx = sIdxSource[idx];
    if (!bvDead->TestBit(sIdx)) {
      unsigned int destIdx = Path(sIdx, 1) == 0 ? leftOff++ : rightOff++;
      targ[destIdx] = source[idx];
      sIdxTarg[destIdx] = sIdx;
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
void Bottom::Buffers(const SPPair &mrra, unsigned int bufIdx, SPNode *&source, unsigned int *&sIdxSource, SPNode *&targ, unsigned int *&sIdxTarg) const {
  samplePred->Buffers(mrra.second, bufIdx, source, sIdxSource, targ, sIdxTarg);
}



/**
   @brief Sets dense count on target MRRA and, if singleton, sets run count to
   unity.

   @return void.
 */
void Level::RunCounts(const unsigned int reachOffset[], const SPNode targ[], const SPPair &mrra, Level *levelFront) {
  unsigned int predIdx = mrra.second;
  PathNode *pathPos = &pathNode[BackScale(mrra.first)];
  for (unsigned int path = 0; path < BackScale(1); path++) {
    unsigned int levelIdx, offset, idxCount;
    pathPos[path].Coords(levelIdx, offset, idxCount);
    if (levelIdx != noIndex) {
      levelFront->SetRuns(levelIdx, predIdx, idxCount, offset, reachOffset[path], targ);
    }
  }
}


/**
   @brief Sets dense count and conveys tied cell as single run.

   @param idxNext is one beyond the final index written.

   @return void.
 */
void Level::SetRuns(unsigned int levelIdx, unsigned int predIdx, unsigned int idxCount, unsigned int start, unsigned int idxNext, const SPNode *targ) {
  unsigned int explicitCount = idxNext - start;
  unsigned int denseCount = idxCount - explicitCount;
  if (explicitCount == 0) {
    def[PairOffset(levelIdx, predIdx)].SetRunCount(1);
  }
  else if (targ->IsRun(start, idxNext - 1)) {
    if (PBTrain::IsFactor(predIdx)) { // Factor:  singleton or doubleton.
      unsigned int runCount = 1 + (denseCount > 0 ? 1 : 0);
      def[PairOffset(levelIdx, predIdx)].SetRunCount(runCount);
    }
    else if (denseCount == 0) { // Numeric:  only singletons tracked.
      def[PairOffset(levelIdx, predIdx)].SetRunCount(1);
    }
  }

  def[PairOffset(levelIdx, predIdx)].SetDenseCount(denseCount);
}



/**
   @brief Invoked from splitting methods to precipitate creation of signature
   for candidate split.

   @return void.
*/
void Bottom::SSWrite(unsigned int levelIdx, unsigned int predIdx, unsigned int setPos, unsigned int bufIdx, const SplitNux &nux) const {
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

   @return void.
 */
void Bottom::Overlap(unsigned int splitCount) {
  splitPrev = frontCount;
  levelFront = new Level(splitCount, nPred, bagCount);
  level.push_front(levelFront);

  historyPrev = std::move(history);
  std::vector<unsigned int> _history(splitCount * (level.size()-1));
  history = std::move(_history);

  deltaPrev = std::move(levelDelta);
  std::vector<unsigned char> _levelDelta(splitCount * nPred);
  levelDelta = std::move(_levelDelta);


  // Recomputes paths reaching from non-front levels.
  //
  for(unsigned int i = 1; i < level.size(); i++) {
    level[i]->Paths();
  }

  frontCount = splitCount;
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
  node.Init(noIndex, 0, 0);
  std::fill(path.begin(), path.end(), node);
  std::fill(live.begin(), live.end(), 0);
  
  pathNode = move(path);
  liveCount = move(live);
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
void Bottom::ReachingPath(unsigned int par, unsigned int path, unsigned int levelIdx, unsigned int start, unsigned int extent) {
  for (unsigned int backLevel = 0; backLevel < level.size() - 1; backLevel++) {
    history[levelIdx + frontCount * backLevel] = backLevel == 0 ? par : historyPrev[par + splitPrev * (backLevel - 1)];
  }

  Inherit(levelIdx, par);
  levelFront->Node(levelIdx, start, extent, par);

  // Places <levelIdx, start> pair at appropriate position in every
  // reaching path.
  //
  for (unsigned int i = 1; i < level.size(); i++) {
    level[i]->PathInit(this, levelIdx, path, start, extent);
  }
}


/**
   @brief Initializes the cell fields for a node in the upcoming level.

   @return void.
 */
void Level::Node(unsigned int levelIdx, unsigned int start, unsigned int extent, unsigned int par) {
  Cell _cell;
  _cell.Init(start, extent);
  cell[levelIdx] = _cell;
}


void Level::PathInit(const Bottom *bottom, unsigned int levelIdx, unsigned int path, unsigned int start, unsigned int extent) {
  unsigned int mrraIdx = bottom->History(levelIdx, del);
  unsigned int pathOff = BackScale(mrraIdx);
  unsigned int pathBits = path & (BackScale(1) - 1);
  pathNode[pathOff + pathBits].Init(levelIdx, start, extent);
  liveCount[mrraIdx]++;
}


SamplePath::SamplePath() : extinct(0), path(0) {
}
