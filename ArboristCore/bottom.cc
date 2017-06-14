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
#include "path.h"

#include <numeric>
#include <algorithm>

// Testing only:
//#include <iostream>
//using namespace std;
//#include <time.h>
//clock_t clock(void);


/**
   @brief Static entry for regression.
 */
Bottom *Bottom::FactoryReg(const PMTrain *_pmTrain, const RowRank *_rowRank, SamplePred *_samplePred, unsigned int _bagCount) {
  return new Bottom(_pmTrain, _samplePred, _rowRank, new SPReg(_pmTrain, _rowRank, _samplePred, _bagCount), _bagCount);
}


/**
   @brief Static entry for classification.
 */
Bottom *Bottom::FactoryCtg(const PMTrain *_pmTrain, const RowRank *_rowRank, SamplePred *_samplePred, const std::vector<SampleNode> &_sampleCtg, unsigned int _bagCount) {
  return new Bottom(_pmTrain, _samplePred, _rowRank, new SPCtg(_pmTrain, _rowRank, _samplePred, _sampleCtg, _bagCount), _bagCount);
}


/**
   @brief Class constructor.

   @param bagCount enables sizing of predicate bit vectors.

   @param splitCount specifies the number of splits to map.
 */
Bottom::Bottom(const PMTrain *_pmTrain, SamplePred *_samplePred, const class RowRank *_rowRank, SplitPred *_splitPred, unsigned int _bagCount) : nPred(_pmTrain->NPred()), nPredFac(_pmTrain->NPredFac()), bagCount(_bagCount), termST(std::vector<unsigned int>(bagCount)), nodeRel(false), stPath(new IdxPath(bagCount)), splitPrev(0), splitCount(1), pmTrain(_pmTrain), samplePred(_samplePred), rowRank(_rowRank), splitPred(_splitPred), splitSig(new SplitSig(nPred)), run(splitPred->Runs()), replayExpl(new BV(bagCount)), history(std::vector<unsigned int>(0)), levelDelta(std::vector<unsigned char>(nPred)), levelFront(new Level(1, nPred, rowRank->DenseIdx(), rowRank->NPredDense(), bagCount, bagCount, nodeRel)), runCount(std::vector<unsigned int>(nPredFac)) {
  level.push_front(levelFront);
  levelFront->Ancestor(0, 0, bagCount);
  std::fill(levelDelta.begin(), levelDelta.end(), 0);
  std::fill(runCount.begin(), runCount.end(), 0);

  splitPred->SetBottom(this);
}


/**
   @brief Adds a new definition at the root level.

   @param predIdx is the predictor index.

   @param singleton is true iff column consists of indentically-ranked samples.

   @param implicit is the number of implicitly-sampled indices.

   @return void.
 */
void Bottom::RootDef(unsigned int predIdx, bool singleton, unsigned int implicit) {
  const unsigned int bufIdx = 0; // Initial staging buffer index.
  const unsigned int levelIdx = 0;
  (void) levelFront->Define(levelIdx, predIdx, bufIdx, singleton, implicit);
  SetRunCount(levelIdx, predIdx, false, singleton ? 1 : pmTrain->FacCard(predIdx));
}

  
Level::Level(unsigned int _splitCount, unsigned int _nPred, const std::vector<unsigned int> &_denseIdx, unsigned int _nPredDense, unsigned int bagCount, unsigned int _idxLive, bool _nodeRel) : nPred(_nPred), denseIdx(_denseIdx), nPredDense(_nPredDense), splitCount(_splitCount), noIndex(bagCount), idxLive(_idxLive), nodeRel(_nodeRel), defCount(0), del(0), indexAnc(std::vector<IndexAnc>(splitCount)), def(std::vector<MRRA>(splitCount * nPred)), denseCoord(std::vector<DenseCoord>(splitCount * nPredDense)), relPath(new IdxPath(idxLive)) {
  MRRA df;
  df.Init();
  std::fill(def.begin(), def.end(), df);
}


/**
   @param sumExpl outputs response sum over explicit hand of the split.

   @return true iff left hand of the split is explicit.
 */
bool Bottom::NonTerminal(PreTree *preTree, SSNode *ssNode, unsigned int extent, unsigned int ptId, double &sumExpl) {
  return ssNode->NonTerminal(this, preTree, Runs(), extent, ptId, sumExpl);
}


/**
  @brief Absorbs subtree sample-to-pt map.  Implemented as move, until
  such time as independent subtrees supported.

  @param stMap is the subtree's sample-to-frontier map.

  @return pretree index.
*/
void Bottom::SubtreeFrontier(PreTree *preTree) const {
  preTree->SubtreeFrontier(termKey, termST);
}


/**
   @brief Prepares crescent successor level.

   @param _idxLive is the number of nonextinct indices in this level.

   @param levelTerminal is true iff the subtree terminates at this level.
 */
void Bottom::Overlap(PreTree *preTree, unsigned int splitNext, unsigned int leafNext) {
  preTree->Level(splitNext, leafNext);
  replayExpl->Clear();
}


/**
   @brief Maps a block of sample indices from a splitting pair to the pretree node in whose sample set the indices now, as a result of splitting, reside.

   @param predIdx is the splitting predictor.

   @param sourceBit (0/1) indicates which buffer holds the current values.

   @param start is the block starting index.

   @param end is the block ending index.

   @return sum of response values associated with each replayed index.
*/
double Bottom::BlockReplay(unsigned int predIdx, unsigned int sourceBit, unsigned int start, unsigned int extent) {
  return samplePred->BlockReplay(predIdx, sourceBit, start, extent, replayExpl);
}


/**
   @brief Selets reindexing method based on current indexing mode.

   @return void.
 */
void Bottom::Reindex(IndexLevel *indexLevel) {
  if (nodeRel) {
    indexLevel->Reindex(this, replayExpl);
  }
  else {
    indexLevel->Reindex(this, replayExpl, stPath);
  }
}


/**
   @brief Entry to spltting and restaging.

   @return vector of splitting signatures, possibly empty, for each node passed.
 */
void Bottom::Split(IndexLevel &index, std::vector<SSNode*> &argMax) {
  LevelInit();
  unsigned int supUnFlush = FlushRear();
  splitPred->LevelInit(index);

  Backdate();
  Restage();

  // Source levels must persist through restaging ut allow path lookup.
  //
  for (unsigned int off = level.size() -1 ; off > supUnFlush; off--) {
    delete level[off];
    level.pop_back();
  }

  splitPred->Split(index);

  return ArgMax(index, argMax);
}


void Bottom::ArgMax(const IndexLevel &index, std::vector<SSNode*> &argMax) {
  unsigned int levelCount = argMax.size();

  unsigned int levelIdx;
#pragma omp parallel default(shared) private(levelIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (levelIdx = 0; levelIdx < levelCount; levelIdx++) {
      argMax[levelIdx] = splitSig->ArgMax(levelIdx, index.MinInfo(levelIdx));
    }
  }
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
  if ((level.size() > NodePath::pathMax)) {
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

  // TODO:  Exit if def-bit unset.  Eliminate run-count check.
  unsigned int bufIdx;
  bool singleton;
  Consume(mrraIdx, predIdx, bufIdx, singleton);
  FrontDef(bottom, mrraIdx, predIdx, bufIdx, singleton);
  if (!singleton)
    bottom->ScheduleRestage(del, mrraIdx, predIdx, bufIdx);
}


void Level::FrontDef(Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx, unsigned int bufIdx, bool singleton) {
  unsigned int pathStart = BackScale(mrraIdx);
  for (unsigned int path = 0; path < BackScale(1); path++) {
    bottom->AddDef(nodePath[pathStart + path].Idx(), predIdx, 1 - bufIdx, singleton);
  }
}


void Bottom::ScheduleRestage(unsigned int del, unsigned int mrraIdx, unsigned int predIdx, unsigned bufIdx) {
  SPPair mrra = std::make_pair(mrraIdx, predIdx);
  RestageCoord rsCoord;
  rsCoord.Init(mrra, del, bufIdx);
  restageCoord.push_back(rsCoord);
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
   @brief Class finalizer.
 */
Bottom::~Bottom() {
  for (auto *defLevel : level) {
    defLevel->Flush(this, false);
    delete defLevel;
  }
  level.clear();

  delete stPath;
  delete splitPred;
  delete splitSig;
  delete replayExpl;
}


/**
   @brief Heh heh.
 */
Level::~Level() {
  delete relPath;
}


/**
   @brief Ensures a pair will be restaged for the front level.

   @param bufIdx outputs the front-level buffer index of the pair.

   @return true iff the front-level definition is not a singleton.
 */
bool Bottom::Preschedule(unsigned int levelIdx, unsigned int predIdx, unsigned int &bufIdx) {
  unsigned int del = ReachLevel(levelIdx, predIdx);
  level[del]->FlushDef(this, History(levelIdx, del), predIdx);

  return !levelFront->Singleton(levelIdx, predIdx, bufIdx);
}


/**
   @brief Determines whether a cell is suitable for splitting.

   @param levelIdx is the split index.

   @param predIdx is the predictor index.

   @param runCount outputs the run count.

   @return true iff split candidate is not singleton.
 */
bool Bottom::ScheduleSplit(unsigned int levelIdx, unsigned int predIdx, unsigned int &rCount) const {
  if (levelFront->Singleton(levelIdx, predIdx)) {
    rCount = 1;
    return false;
  }
  else {
    bool isFactor;
    unsigned int facIdx = FacIdx(predIdx, isFactor);
    rCount = isFactor ? runCount[levelIdx * nPredFac + facIdx] : 0;
    return true;
  }
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
  unsigned int del, bufIdx;
  SPPair mrra;
  rsCoord.Ref(mrra, del, bufIdx);

  unsigned int reachOffset[1 << NodePath::pathMax];
  if (level[del]->NodeRel()) { // Both levels employ node-relative indexing.
    unsigned int reachBase[1 << NodePath::pathMax];
    OffsetClone(mrra, del, reachOffset, reachBase);
    Restage(mrra, bufIdx, del, reachBase, reachOffset);
  }
  else { // Source level employs subtree indexing.  Target may or may not.
    OffsetClone(mrra, del, reachOffset);
    Restage(mrra, bufIdx, del, nullptr, reachOffset);
  }
}


/**
   @brief  Diagnositc test for restaging.  Checks that all target paths
   advance by the expected number of indices.

   @return count of mismatches.
 */
unsigned int Level::DiagRestage(const SPPair &mrra, unsigned int reachOffset[]) {
  unsigned int mismatch = 0;
  unsigned int nullPath = 0;
  unsigned int nodeStart = BackScale(mrra.first);
  for (unsigned int path = 0; path < BackScale(1); path++) {
    NodePath &np = nodePath[nodeStart + path];
    if (reachOffset[path] - np.IdxStart() != np.Extent()) {
      //cout << path << ":  " << reachOffset[path] << " != " << np.IdxStart() << " + " << np.Extent() << endl;
      mismatch++;
    }
    if (np.Extent() == 0)
      nullPath++;
  }

  return mismatch;
}


/**
   @brief Precomputes path vector prior to restaging.  This is necessary
   in the case of dense ranks, as cell sizes are not derivable directly
   from index nodes.

   Decomposition into two paths adds ~5% performance penalty, but
   appears necessary for dense packing or for coprocessor loading.
 */
void Bottom::Restage(const SPPair &mrra, unsigned int bufIdx, unsigned int del, const unsigned int reachBase[], unsigned int reachOffset[]) {
  unsigned int startIdx, extent;
  Bounds(mrra, del, startIdx, extent);

  unsigned int pathCount[1 << NodePath::pathMax];
  for (unsigned int path = 0; path < level[del]->BackScale(1); path++) {
    pathCount[path] = 0;
  }

  unsigned int predIdx = mrra.second;
  samplePred->Prepath(level[del]->NodeRel() ?  FrontPath(del) : stPath, reachBase, predIdx, bufIdx, startIdx, extent, PathMask(del), reachBase == nullptr ? nodeRel : true, pathCount);

  // Successors may or may not themselves be dense.
  if (DensePlacement(mrra, del)) {
    level[del]->PackDense(startIdx, pathCount, levelFront, mrra, reachOffset);
  }

  unsigned int rankPrev[1 << NodePath::pathMax];
  unsigned int rankCount[1 << NodePath::pathMax];
  for (unsigned int path = 0; path < level[del]->BackScale(1); path++) {
    rankPrev[path] = rowRank->NoRank();
    rankCount[path] = 0;
  }
  samplePred->RestageRank(predIdx, bufIdx, startIdx, extent, reachOffset, rankPrev, rankCount);
  level[del]->RunCounts(this, mrra, pathCount, rankCount);
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


bool Bottom::IsFactor(unsigned int predIdx) const {
  return pmTrain->IsFactor(predIdx);
}



unsigned int Bottom::FacIdx(unsigned int predIdx, bool &isFactor) const {
  return pmTrain->BlockIdx(predIdx, isFactor);
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
  splitSig->LevelInit(splitCount);
}


/**
   @brief Updates level-based data structures within attendant objects.

   @return void.
 */
void Bottom::LevelClear() {
  splitPred->LevelClear();
  splitSig->LevelClear();
}


/**
   @brief Updates subtree and pretree mappings from temporaries constructed
   during the overlap.  Initializes data structures for restaging and
   splitting the current level of the subtree.

   @param splitNext is the number of splitable nodes in the current
   subtree level.

   @param idxMax is the maximum index width among live nodes.

   @return void.
 */
void Bottom::LevelPrepare(unsigned int splitNext, unsigned int idxLive, unsigned int idxMax) {
  splitPrev = splitCount;
  splitCount = splitNext;
  if (splitCount == 0) // No further splitting or restaging.
    return;

  if (!nodeRel) { // Sticky.
    nodeRel = IdxPath::Localizes(bagCount, idxMax);
  }
  levelFront = new Level(splitCount, nPred, rowRank->DenseIdx(), rowRank->NPredDense(), bagCount, idxLive, nodeRel);
  level.push_front(levelFront);

  historyPrev = std::move(history);
  history = std::move(std::vector<unsigned int>(splitCount * (level.size()-1)));

  deltaPrev = std::move(levelDelta);
  levelDelta = std::move(std::vector<unsigned char>(splitCount * nPred));

  runCount = std::move(std::vector<unsigned int>(splitCount * nPredFac));
  std::fill(runCount.begin(), runCount.end(), 0);

  // Recomputes paths reaching from non-front levels.
  //
  for (unsigned int i = 1; i < level.size(); i++) {
    level[i]->Paths();
  }
}


/**
   @brief Pushes first level's path maps back to all back levels
   employing node-relative indexing.

   @return void.
 */
void Bottom::Backdate() const {
  if (level.size() > 2 && level[1]->NodeRel()) {
    for (auto lv = level.begin() + 2; lv != level.end(); lv++) {
      if (!(*lv)->Backdate(FrontPath(1))) {
	break;
      }
    }
  }
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
  std::vector<unsigned int> live(splitCount);
  std::vector<NodePath> path(BackScale(splitCount));
  NodePath np;
  np.Init(noIndex, 0, 0, 0);
  std::fill(path.begin(), path.end(), np);
  std::fill(live.begin(), live.end(), 0);
  
  nodePath = move(path);
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
void Bottom::ReachingPath(unsigned int levelIdx, unsigned int parIdx, unsigned int start, unsigned int extent, unsigned int relBase/*ptId*/, unsigned int path) {
  for (unsigned int backLevel = 0; backLevel < level.size() - 1; backLevel++) {
    history[levelIdx + splitCount * backLevel] = backLevel == 0 ? parIdx : historyPrev[parIdx + splitPrev * (backLevel - 1)];
  }

  Inherit(levelIdx, parIdx);
  levelFront->Ancestor(levelIdx, start, extent);
  
  // Places <levelIdx, start> pair at appropriate position in every
  // reaching path.
  //
  for (unsigned int i = 1; i < level.size(); i++) {
    level[i]->PathInit(this, levelIdx, path, start, extent, relBase);//RelBase(ptId));
  }
}


/**
     @brief Updates both node-relative path for a live index, as
     well as subtree-relative if back levels warrant.

     @param ndx is a node-relative index from the previous level.

     @param stx is the associated subtree-relative index.

     @param path is the path reaching the target node.

     @param targIdx is the updated node-relative index:  current level.

     @param ndBase is the base index of the target node:  current level.

     @return void.
   */
void Bottom::SetLive(unsigned int ndx, unsigned int targIdx, unsigned int stx, unsigned int path, unsigned int ndBase) {
  levelFront->SetLive(ndx, path, targIdx, ndBase);

  if (!level.back()->NodeRel()) {
    stPath->SetLive(stx, path, targIdx);  // Irregular write.
  }
}


/**
   @brief Copies a node's subtree indices onto the terminal vector.  Marks
   index paths as extinct.
 */
void Bottom::Terminal(unsigned int termBase, unsigned int extent, unsigned int ptId) {
  TermKey key;
  key.Init(termBase, extent, ptId);
  termKey.push_back(key);
}


/**
   @brief Sends subtree-relative index to terminal vector.  Marks subtree-
   relative path as extinct if still required by back levels.
 */
void Bottom::SetExtinct(unsigned int termIdx, unsigned int stIdx) {  
  termST[termIdx] = stIdx;
  if (!level.back()->NodeRel()) {
    stPath->SetExtinct(stIdx);
  }
}


void Level::SetExtinct(unsigned int idx) {
  relPath->SetExtinct(idx);
}


void Level::PathInit(const Bottom *bottom, unsigned int levelIdx, unsigned int path, unsigned int start, unsigned int extent, unsigned int relBase) {
  unsigned int mrraIdx = bottom->History(levelIdx, del);
  unsigned int pathOff = BackScale(mrraIdx);
  unsigned int pathBits = path & PathMask();
  nodePath[pathOff + pathBits].Init(levelIdx, start, extent, relBase);
  liveCount[mrraIdx]++;
}


/**
   @brief Sets path, target and node-relative offse.

   @return void.
 */
void Level::SetLive(unsigned int idx, unsigned int path, unsigned int targIdx, unsigned int ndBase) {
  relPath->SetLive(idx, path, targIdx, targIdx - ndBase);
}

  
