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

// Testing only:
//#include <iostream>
//using namespace std;
//#include <time.h>
//clock_t clock(void);

/**
   @brief Static entry for regression.
 */
Bottom *Bottom::FactoryReg(SamplePred *_samplePred, unsigned int _bagCount) {
  return new Bottom(_samplePred, new SPReg(_samplePred, _bagCount), _bagCount, PBTrain::NPred(), PBTrain::NPredFac());
}


/**
   @brief Static entry for classification.
 */
Bottom *Bottom::FactoryCtg(SamplePred *_samplePred, SampleNode *_sampleCtg, unsigned int _bagCount) {
  return new Bottom(_samplePred, new SPCtg(_samplePred, _sampleCtg, _bagCount), _bagCount, PBTrain::NPred(), PBTrain::NPredFac());
}


/**
   @brief Class constructor.

   @param bagCount enables sizing of predicate bit vectors.

   @param splitCount specifies the number of splits to map.
 */
Bottom::Bottom(SamplePred *_samplePred, SplitPred *_splitPred, unsigned int _bagCount, unsigned int _nPred, unsigned int _nPredFac) : nPred(_nPred), nPredFac(_nPredFac), bagCount(_bagCount), samplePath(new SamplePath[bagCount]), frontCount(1), bvLeft(new BV(bagCount)), bvDead(new BV(bagCount)), samplePred(_samplePred), splitPred(_splitPred), splitSig(new SplitSig()), run(splitPred->Runs()) {
  levelFront = new Level(1, nPred, bagCount);
  level.push_front(levelFront);

  levelFront->Node(0, 0, bagCount, bagCount);
  levelFront->RootDef(nPred);

  splitPred->SetBottom(this);
}


/**
   @brief Adds a new definition at the root level.

   @param nPred is the number of predictors.

   @return void.
 */
void Level::RootDef(unsigned int nPred) {
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    def[predIdx] = PBTrain::FacCard(predIdx);
    defined->SetBit(0, predIdx);
    defCount++;
  }
}

  
Level::Level(unsigned int _splitCount, unsigned int _nPred, unsigned int _noIndex) : nPred(_nPred), splitCount(_splitCount), noIndex(_noIndex), defCount(0), del(0) {
  std::vector<unsigned int> _parent(splitCount);
  std::vector<Cell> _cell(splitCount);
  //std::map<unsigned int, unsigned int> _def;
  std::vector<unsigned int> _def(splitCount * nPred);
  std::fill(_def.begin(), _def.end(), 0);
  defined = new BitMatrix(splitCount, nPred);
  buffer = new BitMatrix(splitCount, nPred);
  parent = std::move(_parent);
  cell = std::move(_cell);
  def = std::move(_def);
}


/**
   @brief Entry to spltting and restaging.

   @return vector of splitting signatures, possibly empty, for each node passed.
 */
const std::vector<class SSNode*> Bottom::Split(class Index *index, class IndexNode indexNode[]) {
  FlushRear();
  splitPred->LevelInit(index, indexNode, frontCount);

  if (!restageCoord.empty()) {
    Restage();
    restageCoord.clear();
  }
  if (level.size() > pathMax) {
    delete level.back();
    level.pop_back();
  }

  Split(indexNode);

  std::vector<SSNode*> ssNode(frontCount);
  for (unsigned int levelIdx = 0; levelIdx < frontCount; levelIdx++) {
    ssNode[levelIdx] = splitSig->ArgMax(levelIdx, indexNode[levelIdx].MinInfo());
  }

  return ssNode;
}


/**
   @brief Flushes non-reaching definitions as well as those about
   to fall off the level deque.
 */
void Bottom::FlushRear() {
  // Walks backward from rear, purging non-reaching definitions.
  // Stops when a level with no non-reaching nodes is encountered.
  //
  for (unsigned int off = level.size() - 1; off > 0; off--) {
    if (!level[off]->NonreachPurge())
      break;
  }

  // Capacity:  1 front level + 'pathMax' back levels.
  // If at capacity, every reaching definition should be flushed
  // to current level ut avoid falling off the deque.
  // Flushing prior to split assignment, rather than during, should
  // also save lookup time, as all definitions reaching from rear are
  // now at current level.
  //
  if ((level.size() > pathMax)) {
    level.back()->Flush(this);
  }
}


/**
   @brief Walks the definitions, purging those which no longer reach.

   @return true iff a definition was purged at this level.
 */
bool Level::NonreachPurge() {
  bool purged = false;
  for (unsigned int mrraIdx = 0; mrraIdx < splitCount; mrraIdx++) {
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      if (!defined->TestBit(mrraIdx, predIdx))
	continue;
      if (liveCount[mrraIdx] == 0) {
	PurgeDef(mrraIdx, predIdx);
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
      if (!defined->TestBit(mrraIdx, predIdx))
	continue;
      if (forward) {
	FlushDef(bottom, mrraIdx, predIdx);
      }
      else {
	PurgeDef(mrraIdx, predIdx);
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
  // N.B.:  Assumes definition reaches
  PathNode *pathStart = &pathNode[BackScale(mrraIdx)];
  unsigned int extent = BackScale(1);
  unsigned int sourceBit = buffer->TestBit(mrraIdx, predIdx) ? 1 : 0;
  unsigned int defRC = def[PairOffset(mrraIdx, predIdx)];
  for (unsigned int path = 0; path < extent; path++) {
    bottom->AddDef(pathStart, path, predIdx, defRC, 1 - sourceBit);
  }
  SplitPair mrra = std::make_pair(mrraIdx, predIdx);
  bottom->ScheduleRestage(mrra, del, sourceBit);

  PurgeDef(mrraIdx, predIdx);
}


// def[], sourceBit may be left dirty:  no reads after dedefinition.
void Level::PurgeDef(unsigned int mrraIdx, unsigned int predIdx) {
  defined->ClearBit(mrraIdx, predIdx);
  defCount--;
}


/**
   @brief Adds a new definition to front level at passed coordinates, provided
   that the node index is live.

   @return true iff path reaches to current level.
 */
void Level::AddDef(PathNode pathReach[], unsigned int path, unsigned int predIdx, unsigned int defRC, unsigned int destBit) {
  unsigned reachIdx = pathReach[path].Idx();
  if (reachIdx != noIndex) {
    def[PairOffset(reachIdx, predIdx)] = defRC;
    defined->SetBit(reachIdx, predIdx);
    buffer->SetBit(reachIdx, predIdx, destBit);
    defCount++;
  }
}


void Bottom::ScheduleRestage(const SplitPair &mrra, unsigned int del, unsigned bufIdx) {
  RestageCoord rsSet;
  rsSet.Init(mrra, del, bufIdx);
  restageCoord.push_back(rsSet);
}


void Level::CellBounds(const SplitPair &mrra, unsigned int &startIdx, unsigned int &extent) {
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
  delete defined;
  delete buffer;
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
   @brief Records coordinates and offsets for candidate splitting pairs.

   @param runTop is the current top of a dense vector of run values.

   @return run count of front-level definition.
 */
unsigned int Bottom::ScheduleSplit(unsigned int levelIdx, unsigned int predIdx, unsigned int runTop) {
  unsigned int rc;
  unsigned int bufIdx = DefForward(levelIdx, predIdx, rc);

  if (rc != 1) { // No reason to attempt splitting a singleton.
    SplitCoord sg;
    sg.Init(splitCoord.size(), levelIdx, predIdx, bufIdx, rc, runTop);
    splitCoord.push_back(sg);
  }
  
  return rc;
}


/**
   @brief Finds definition reaching coordinate pair at current level,
   flushing ancestor if necessary.  Obtains node reached by coordinates
   within current level.

   @param levelIdx is the node index within current level.

   @param predIdx is the predictor index.

   @param runCount outputs the reached node's run count.

   @return buffer index of reached definition.
 */
unsigned int Bottom::DefForward(unsigned int levelIdx, unsigned int predIdx, unsigned int &runCount) {
  unsigned int mrraIdx = levelIdx; // Chains upward through parent indices.
  for (unsigned int del = 0; del < level.size(); del++) {
    if (level[del]->Defines(mrraIdx, predIdx)) {
      if (del > 0) { // Else already flushed to front level.
        level[del]->FlushDef(this, mrraIdx, predIdx);
      }
      break;
    }
  }
 
  return levelFront->Definition(levelIdx, predIdx, runCount);
}


/**
   @brief Tests whether this level defines split/predictor pair passed.

   @param mrraIdx inputs trial level index / outputs parent index if level
   does not define pair.

   @param predIdx is the predictor index of the pair.

   @return true iff pair defined at this level.
 */
bool Level::Defines(unsigned int &mrraIdx, unsigned int predIdx) {
  if (defined->TestBit(mrraIdx, predIdx)) {
    return true;
  }
  else {
    mrraIdx = parent[mrraIdx];
    return false;
  }
}


/**
   @brief Looks up definition associated with coordinates passed.

   @param levelIdx is the node index.

   @param predIdx is the predictor index.

   @return buffer index of definition.
 */
unsigned int Level::Definition(unsigned int levelIdx, unsigned int predIdx, unsigned int &runCount) {
  runCount = def[PairOffset(levelIdx, predIdx)];
  return buffer->TestBit(levelIdx, predIdx) ? 1 : 0;
}


/**
   @brief Dispatches splitting of staged pairs independently.

   @return void.
 */
void Bottom::Split(const IndexNode indexNode[]) {
  // Guards cast to int for OpenMP 2.0 back-compatibility.
  int splitPos;
#pragma omp parallel default(shared) private(splitPos)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitPos = 0; splitPos < int(splitCoord.size()); splitPos++) {
      splitCoord[splitPos].Split(samplePred, indexNode, splitPred);
    }
  }

  splitCoord.clear();
}


/**
   @brief Dispatches the staged node to the appropriate splitting family.

   @return void.
 */
void SplitCoord::Split(const SamplePred *samplePred, const IndexNode indexNode[], SplitPred *splitPred) {
  splitPred->Split(splitPos, &indexNode[levelIdx], samplePred->PredBase(predIdx, bufIdx));
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
  unsigned int del, bufIdx;
  SplitPair mrra;
  rsCoord.Ref(mrra, del, bufIdx);

  OffsetClone(mrra, del, reachOffset);

  SPNode *targ;
  if (del == 1) {
    targ = RestageOne(reachOffset, mrra, bufIdx);
  }
  else {
    targ = RestageIrr(reachOffset, mrra, del, bufIdx);
  }

  Singletons(reachOffset, targ, mrra, del);
}


/**
   @brief Irregular restaging, suitable for smaller sample sets.

   @return void.
 */
SPNode *Bottom::RestageIrr(unsigned int reachOffset[], const SplitPair &mrra, unsigned int del, unsigned int bufIdx) {
  SPNode *source, *targ;
  unsigned int *sIdxSource, *sIdxTarg;
  Buffers(mrra, bufIdx, source, sIdxSource, targ, sIdxTarg);

  unsigned int startIdx, extent;
  CellBounds(del, mrra, startIdx, extent);
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
void Level::OffsetClone(const SplitPair &mrra, unsigned int reachOffset[]) {
  unsigned int nodeStart = BackScale(mrra.first);
  for (unsigned int i = 0; i < BackScale(1); i++) {
    reachOffset[i] = pathNode[nodeStart + i].Offset();
  }
}


/**
   @brief Specialized for two-path case, bypasses stack array.

 */
SPNode *Bottom::RestageOne(unsigned int reachOffset[], const SplitPair &mrra, unsigned int bufIdx) {
  SPNode *source, *targ;
  unsigned int *sIdxSource, *sIdxTarg;
  Buffers(mrra, bufIdx, source, sIdxSource, targ, sIdxTarg);

  unsigned int startIdx, extent;
  CellBounds(1, mrra, startIdx, extent);
  unsigned int leftOff = reachOffset[0];
  unsigned int rightOff = reachOffset[1];
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
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
void Bottom::Buffers(const SplitPair &mrra, unsigned int bufIdx, SPNode *&source, unsigned int *&sIdxSource, SPNode *&targ, unsigned int *&sIdxTarg) const {
  samplePred->Buffers(mrra.second, bufIdx, source, sIdxSource, targ, sIdxTarg);
}


void Level::Singletons(const unsigned int reachOffset[], const SPNode targ[], const SplitPair &mrra, Level *levelFront) {
  unsigned int predIdx = mrra.second;
  if (PBTrain::FacCard(predIdx) == 0)
    return;
  PathNode *pathPos = &pathNode[BackScale(mrra.first)];
  for (unsigned int path = 0; path < BackScale(1); path++) {
    unsigned int levelIdx, offset;
    pathPos[path].Coords(levelIdx, offset);
    if (levelIdx != noIndex) {
      if (targ->IsRun(offset, reachOffset[path]-1)) {
	levelFront->def[PairOffset(levelIdx, predIdx)] = 1;//->SetSingleton();
      }
    }
  }
}


/**
   @brief Invoked from splitting methods to precipitate creation of signature
   for candidate split.

   @return void.
 */
void Bottom::SSWrite(unsigned int splitIdx, unsigned int lhSampCount, unsigned int lhIdxCount, double info) {
  unsigned int levelIdx, predIdx, bufIdx;
  int runsetPos;
  SplitRef(splitIdx, levelIdx, predIdx, runsetPos, bufIdx);
  splitSig->Write(levelIdx, predIdx, runsetPos, bufIdx, lhSampCount, lhIdxCount, info);
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
void Bottom::NewLevel(unsigned int splitCount) {
  levelFront = new Level(splitCount, nPred, bagCount);
  level.push_front(levelFront);
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
  node.Init(noIndex, 0);
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

   @param start is the cell starting position.

   @return void.
*/
void Bottom::ReachingPath(unsigned int par, unsigned int path, unsigned int levelIdx, unsigned int start, unsigned int extent) {
  levelFront->Node(levelIdx, start, extent, par);

  // Places <levelIdx, start> pair at appropriate position in every
  // reaching path.
  //
  unsigned int mrraIdx = levelFront->ParentIdx(levelIdx);
  for (unsigned int i = 1; i < level.size(); i++) {
    level[i]->PathInit(mrraIdx, path, levelIdx, start);
  }
}


/**
   @brief Initializes the cell and parent fields for a node in the upcoming level.

   @return void.
 */
void Level::Node(unsigned int levelIdx, unsigned int start, unsigned int extent, unsigned int par) {
  Cell _cell;
  _cell.Init(start, extent);
  cell[levelIdx] = _cell;
  parent[levelIdx] = par;
}


void Level::PathInit(unsigned int &mrraIdx, unsigned int path, unsigned int levelIdx, unsigned int start) {
  unsigned int pathOff = BackScale(mrraIdx);
  unsigned int pathBits = path & (BackScale(1) - 1);
  pathNode[pathOff + pathBits].Init(levelIdx, start);
  liveCount[mrraIdx]++;
  mrraIdx = parent[mrraIdx];
}


SamplePath::SamplePath() : extinct(0), path(0) {
}
