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
Bottom::Bottom(const PMTrain *_pmTrain, SamplePred *_samplePred, SplitPred *_splitPred, unsigned int _bagCount, unsigned int _stageSize) : nPred(_pmTrain->NPred()), nPredFac(_pmTrain->NPredFac()), bagCount(_bagCount), termST(std::vector<unsigned int>(bagCount)), termTop(0), nodeRel(false), prePath(std::vector<unsigned int>(_stageSize)), stPath(new IdxPath(bagCount)), splitPrev(0), frontCount(1), levelBase(0), ptHeight(1), pmTrain(_pmTrain), samplePred(_samplePred), splitPred(_splitPred), splitSig(new SplitSig(nPred)), run(splitPred->Runs()), idxLive(bagCount), rel2ST(std::vector<unsigned int>(bagCount)), relBase(std::vector<unsigned int>(1)), replayExpl(new BV(bagCount)), history(std::vector<unsigned int>(0)), levelDelta(std::vector<unsigned char>(nPred)), levelFront(new Level(1, nPred, bagCount, bagCount, nodeRel)) {
  std::iota(rel2ST.begin(), rel2ST.end(), 0);
  relBase[0] = 0;
  std::fill(levelDelta.begin(), levelDelta.end(), 0);
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

  
Level::Level(unsigned int _splitCount, unsigned int _nPred, unsigned int bagCount, unsigned int _idxLive, bool _nodeRel) : nPred(_nPred), splitCount(_splitCount), noIndex(bagCount), idxLive(_idxLive), nodeRel(_nodeRel), defCount(0), del(0), indexAnc(std::vector<IndexAnc>(splitCount)), def(std::vector<MRRA>(splitCount * nPred)), relPath(new IdxPath(idxLive)) {
  MRRA df;
  df.Undefine();
  std::fill(def.begin(), def.end(), df);
}


double Bottom::NonTerminal(PreTree *preTree, SSNode *ssNode, unsigned int extent, unsigned int lhExtent, double sum, unsigned int &ptId) {
  double lhSum = ssNode->NonTerminal(this, preTree, Runs(), extent, sum, ptId);
  Successor(preTree->LHId(ptId), lhExtent);
  Successor(preTree->RHId(ptId), extent - lhExtent);

  return lhSum;
}


/**
   @brief Copies a node's subtree indices onto the terminal vector.  Marks
   index paths as extinct.
 */
void Bottom::Terminal(unsigned int extent, unsigned int ptId) {
  unsigned int nodeBase = RelBase(ptId);
  for (unsigned int relIdx = nodeBase; relIdx < nodeBase + extent; relIdx++) {
    SetExtinct(relIdx);
  }

  TermKey key;
  key.Init(extent, ptId);
  termKey.push_back(key);
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
void Bottom::LevelSucc(PreTree *preTree, unsigned int splitNext, unsigned int leafNext, unsigned int idxExtent, unsigned int _idxLive, bool levelTerminal) {
  preTree->Level(splitNext, leafNext);
  replayExpl->Clear();

  succBase = std::move(std::vector<unsigned int>(splitNext + leafNext));
  std::fill(succBase.begin(), succBase.end(), idxExtent); // Inattainable base.

  succST = std::move(std::vector<unsigned int>(idxExtent));
  std::fill(succST.begin(), succST.end(), idxExtent); // Inattainable st index.

  liveBase = 0;
  extinctBase = _idxLive;
  idxLive = levelTerminal ? 0 : _idxLive; // Value for upcoming level.
}


/**
   @brief Builds index base offsets to mirror crescent pretree level.

   @return void.
 */
void Bottom::Successor(unsigned int ptId, unsigned int extent) {
  unsigned int succOff = OffsetSucc(ptId);
  if (IndexSet::Splitable(extent)) {
    succBase[succOff] = liveBase;
    liveBase += extent;
  }
  else {
    succBase[succOff] = extinctBase;
    extinctBase += extent;
  }
}


/**
   @brief Maps a block of sample indices from a splitting pair to the pretree node in whose sample set the indices now, as a result of splitting, reside.

   @param predIdx is the splitting predictor.

   @param sourceBit (0/1) indicates which buffer holds the current values.

   @param start is the block starting index.

   @param end is the block ending index.

   @return sum of response values associated with each replayed index.
*/
double Bottom::BlockPreplay(unsigned int predIdx, unsigned int sourceBit, unsigned int start, unsigned int extent) {
  return samplePred->BlockPreplay(predIdx, sourceBit, start, extent, replayExpl);
}


/**
   @brief Updates the successor level's node- and subtree-relative index maps
   via a stable partition.

   @return void.
 */
void Bottom::Replay(const PreTree *preTree, unsigned int ptId, unsigned int path, bool leftExpl, unsigned int extent, unsigned int lhExtent) {
  unsigned int baseImpl = SuccBase(leftExpl ? preTree->RHId(ptId) : preTree->LHId(ptId));
  unsigned int baseExpl = SuccBase(leftExpl ? preTree->LHId(ptId) : preTree->RHId(ptId));

  unsigned int idxSource = RelBase(ptId);
  unsigned int offExpl = baseExpl;
  unsigned int offImpl = baseImpl;
  unsigned int pathL = IdxPath::PathNext(path, true);
  unsigned int pathR = IdxPath::PathNext(path, false);
  for (unsigned int relIdx = idxSource; relIdx < idxSource + extent; relIdx++) {
    unsigned int stIdx = rel2ST[relIdx];
    bool expl = replayExpl->TestBit(nodeRel ? relIdx : stIdx);
    unsigned int targIdx = expl ? offExpl++ : offImpl++;
    succST[targIdx] = stIdx;

    // Live index updates could be deferred to production of the next level,
    // but the relIdx-to-targIdx mapping is conveniently available here.
    //
    // Extinct subtree indices are best updated on a per-node basis, however,
    // so their treatment is deferred until production of the next level.
    // Extinct node-relative indices, however, can be treated on the fly.
    //
    if (targIdx >= idxLive) {
      levelFront->SetExtinct(relIdx);
    }
    else {
      SetLive(relIdx, stIdx, (expl && leftExpl) || !(expl || leftExpl) ? pathL : pathR, targIdx, expl ? baseExpl : baseImpl);
    }
  }
}


/**
   @brief Diagnostic test for replay.  Checks that left and right successors
   receive the expected index counts.

   @return count of mismatched expectations.
 */
unsigned int Bottom::DiagReplay(const PreTree *preTree, unsigned int offExpl, unsigned int offImpl, bool leftExpl, unsigned int ptId, unsigned int lhExtent, unsigned int rhExtent) {
  unsigned int mismatch = 0;
  unsigned int extentImpl = leftExpl ? rhExtent : lhExtent;
  unsigned int extentExpl = leftExpl ? lhExtent : rhExtent;
  unsigned int ptImpl = leftExpl ? preTree->RHId(ptId) : preTree->LHId(ptId);
  unsigned int ptExpl = leftExpl ? preTree->LHId(ptId) : preTree->RHId(ptId);
  if (offExpl != SuccBase(ptExpl) + extentExpl) {
    mismatch++;
  }
  if (offImpl != SuccBase(ptImpl) + extentImpl) {
    mismatch++;
  }

  return mismatch;
}


/**
   @brief Entry to spltting and restaging.

   @return vector of splitting signatures, possibly empty, for each node passed.
 */
void Bottom::Split(IndexLevel &index, std::vector<SSNode*> &argMax) {
  LevelInit();
  unsigned int supUnFlush = FlushRear();
  splitPred->LevelInit(index);

  BackUpdate(); // Deferred until all level-based paths updated.
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

  unsigned int runCount, bufIdx;
  Consume(mrraIdx, predIdx, runCount, bufIdx);
  FrontDef(bottom, mrraIdx, predIdx, runCount, bufIdx);
  if (runCount != 1) // Singletons need not restage.
    bottom->ScheduleRestage(del, mrraIdx, predIdx, runCount, bufIdx);
}


void Level::FrontDef(Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx, unsigned int defRC, unsigned int sourceBit) {
  NodePath *pathStart = &nodePath[BackScale(mrraIdx)];
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
  unsigned int startIdx, extent;
  Bounds(mrra, del, startIdx, extent);
  
  unsigned int reachOffset[1 << NodePath::pathMax];
  if (level[del]->NodeRel()) { // Both levels employ node-relative indexing.
    unsigned int reachBase[1 << NodePath::pathMax];
    OffsetClone(mrra, del, reachOffset, reachBase);
    if (IsDense(mrra, del)) {
      targ = RestageNdxDense(reachOffset, reachBase, mrra, bufIdx, del);
    }
    else if (del == 1) {
      targ = samplePred->RestageNdxOne(reachOffset, reachBase, mrra.second, bufIdx, FrontPath(1), PathMask(1), startIdx, extent);
    }
    else {
      targ = samplePred->RestageNdxGen(reachOffset, reachBase, mrra.second, bufIdx, FrontPath(del), PathMask(del), startIdx, extent);
    }
  }
  else { // Source level employs subtree indexing.  Target may or may not.
    OffsetClone(mrra, del, reachOffset);
    if (IsDense(mrra, del)) {
      targ = RestageStxDense(reachOffset, mrra, bufIdx, del);
    }
    else if (del == 1) {
      targ = samplePred->RestageStxOne(reachOffset, mrra.second, bufIdx, stPath, PathMask(1), startIdx, extent, nodeRel);

    }
    else {
      targ = samplePred->RestageStxGen(reachOffset, mrra.second, bufIdx, stPath, PathMask(del), startIdx, extent, nodeRel);
    }
  }
  //  if (!IsDense(mrra, del) && level[del]->DiagRestage(mrra, reachOffset) > 0)
  //cout << "Bad restage" << endl;

  return targ;
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
 */
SPNode *Bottom::RestageNdxDense(unsigned int reachOffset[], const unsigned int reachBase[], const SPPair &mrra, unsigned int bufIdx, unsigned int del) {
  IdxPath *frontPath = FrontPath(del);
  unsigned int pathMask = PathMask(del);
  unsigned int startIdx, extent;
  Bounds(mrra, del, startIdx, extent);
  
  unsigned int *ppBlock = &prePath[samplePred->StageOffset(mrra.second)];
  unsigned int pathCount[1 << NodePath::pathMax];
  for (unsigned int path = 0; path < level[del]->BackScale(1); path++) {
    pathCount[path] = 0;
  }

  SPNode *source, *targ;
  unsigned int *idxSource, *idxTarg;
  Buffers(mrra, bufIdx, source, idxSource, targ, idxTarg);
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int relSource = idxSource[idx];
    unsigned int path, offRel;
    if (frontPath->RefLive(relSource, pathMask, path, offRel)) {
      ppBlock[idx] = path;
      pathCount[path]++;
      idxSource[idx] = reachBase[path] + offRel; // O.k. to overwrite.
    }
    else {
      ppBlock[idx] = NodePath::noPath;
    }
  }

  // Decomposition into two paths adds ~5% performance penalty, but
  // is necessary for dense packing or for coprocessor loading.
  //
  // Successors may or may not themselves be dense.
  level[del]->PackDense(startIdx, pathCount, levelFront, mrra, reachOffset);
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int path = ppBlock[idx];
    if (path != NodePath::noPath) {
      unsigned int destIdx = reachOffset[path]++;
      targ[destIdx] = source[idx];
      idxTarg[destIdx] = idxSource[idx];
    }
  }

  return targ;
}


/**
   @brief Precomputes path vector prior to restaging.  This is necessary
   in the case of dense ranks, as cell sizes are not derivable directly
   from index nodes.
 */
SPNode *Bottom::RestageStxDense(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx, unsigned int del) {

  // Decomposition into two paths adds ~5% performance penalty, but
  // is necessary for dense packing or for coprocessor loading.
  //
  unsigned int pathMask = PathMask(del);
  unsigned int startIdx, extent;
  Bounds(mrra, del, startIdx, extent);

  unsigned int *ppBlock = &prePath[samplePred->StageOffset(mrra.second)];
  unsigned int pathCount[1 << NodePath::pathMax];
  for (unsigned int path = 0; path < level[del]->BackScale(1); path++) {
    pathCount[path] = 0;
  }

  SPNode *source, *targ;
  unsigned int *idxSource, *idxTarg;
  Buffers(mrra, bufIdx, source, idxSource, targ, idxTarg);
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int relSource = idxSource[idx];
    unsigned int path;
    if (stPath->PathLive(relSource, pathMask, path)) {
      ppBlock[idx] = path;
      pathCount[path]++;
      // RelFront() performs (slow) sIdx-to-relIdx mapping:  transition only.
      idxSource[idx] = nodeRel ? stPath->RelFront(relSource) : relSource;
    }
    else {
      ppBlock[idx] = NodePath::noPath;
    }
  }

  // Successors may or may not themselves be dense.
  level[del]->PackDense(startIdx, pathCount, levelFront, mrra, reachOffset);
  for (unsigned int idx = startIdx; idx < startIdx + extent; idx++) {
    unsigned int path = ppBlock[idx];
    if (path != NodePath::noPath) {
      unsigned int destIdx = reachOffset[path]++;
      targ[destIdx] = source[idx];
      idxTarg[destIdx] = idxSource[idx];
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
  const NodePath *pathPos = &nodePath[BackScale(mrra.first)];
  for (unsigned int path = 0; path < BackScale(1); path++) {
    unsigned int levelIdx, idxStart, extent;
    pathPos[path].Coords(levelIdx, idxStart, extent);
    if (levelIdx != noIndex) {
      unsigned int margin = idxStart - idxLeft;
      unsigned int idxLocal = pathCount[path];
      levelFront->SetDense(levelIdx, mrra.second, margin, extent - idxLocal);
      reachOffset[path] -= margin;
      idxLeft += idxLocal;
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
  if (reachBase != 0) {
    for (unsigned int i = 0; i < BackScale(1); i++) {
      reachBase[i] = nodePath[nodeStart + i].RelBase();
    }
  }
}


/**
   @brief Sets buffer addresses from source coordinates.

   @param mrra holds the ancestor's coordinates.

   @param bufIdx is the index of the source buffer.

   @return void.
 */
void Bottom::Buffers(const SPPair &mrra, unsigned int bufIdx, SPNode *&source, unsigned int *&idxSource, SPNode *&targ, unsigned int *&idxTarg) const {
  samplePred->Buffers(mrra.second, bufIdx, source, idxSource, targ, idxTarg);
}


/**
   @brief Sets dense count on target MRRA and, if singleton, sets run count to
   unity.

   @return void.
 */
void Level::RunCounts(const SPNode targ[], const SPPair &mrra, const Bottom *bottom) const {
  unsigned int predIdx = mrra.second;
  const NodePath *pathPos = &nodePath[BackScale(mrra.first)];
  for (unsigned int path = 0; path < BackScale(1); path++) {
    unsigned int levelIdx, idxStart, extent;
    pathPos[path].Coords(levelIdx, idxStart, extent);
    if (levelIdx != noIndex) {
      bottom->SetRuns(levelIdx, predIdx, idxStart, extent, targ);
    }
  }
}


/**
   @brief Sets dense count and conveys tied cell as single run.

   @return void.
 */
void Level::SetRuns(const Bottom *bottom, unsigned int levelIdx, unsigned int predIdx, unsigned int idxStart, unsigned int extent, const SPNode *targ) {
  MRRA &reach = def[PairOffset(levelIdx, predIdx)];
  unsigned int denseCount = reach.AdjustDense(idxStart, extent);
  if (extent == 0) { // all indices implicit.
    reach.SetRunCount(1);
  }
  else if (targ[idxStart].Rank() == targ[idxStart + extent - 1].Rank()) {
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
   @brief Updates level-based data structures within attendant objects.

   @return void.
 */
void Bottom::LevelClear() {
  splitPred->LevelClear();
  splitSig->LevelClear();
}


/**
   @brief Updates subtree and pretree mappings from temporaries constructed
   during the overlap.

   @param splitNext is the number of splitable nodes in the current
   pretree level.

   @return void.
 */
void Bottom::Overlap(unsigned int splitNext) {
  levelBase = ptHeight;
  ptHeight += succBase.size();
  relBase = std::move(succBase);
  rel2ST = std::move(succST);

  SplitPrepare(splitNext);
}


/**
   @brief Initializes data structures for restaging and splitting
   the current level of the subtree.

   @param idxMax is the maximum index width among live nodes.

   @return void.
 */
void Bottom::SplitPrepare(unsigned int splitNext) {
  if (splitNext == 0) // No further splitting or restaging.
    return;
  
  splitPrev = frontCount;
  frontCount = splitNext;
  if (!nodeRel) { // Sticky.
    nodeRel = IdxPath::Localizes(bagCount, IdxMax());
  }

  levelFront = new Level(frontCount, nPred, bagCount, idxLive, nodeRel);
  level.push_front(levelFront);

  historyPrev = std::move(history);
  history = std::move(std::vector<unsigned int>(frontCount * (level.size()-1)));

  deltaPrev = std::move(levelDelta);
  levelDelta = std::move(std::vector<unsigned char>(frontCount * nPred));

  // Recomputes paths reaching from non-front levels.
  //
  for (unsigned int i = 1; i < level.size(); i++) {
    level[i]->Paths();
  }
}


/**
   @brief Computes the extent of the widest splitable node.  Splitable nodes
   are characterized by base values between zero and 'idxLive'.

   @return maximal node extent.
 */
unsigned int Bottom::IdxMax() const {
  unsigned int idxMax = 0;
  unsigned int top = idxLive;
  for (auto base = relBase.end() - 1; base != relBase.begin(); base--) {
    if (*base < idxLive) { // Otherwise unsplitable.
      unsigned int extent = top - *base;
      idxMax = extent > idxMax ? extent : idxMax;
      top = *base;
    }
  }

  return idxMax;
}


/**
   @brief Pushes first level's path maps back to all back levels
   employing node-relative indexing.

   @return void.
 */
void Bottom::BackUpdate() const {
  if (level.size() > 2 && level[1]->NodeRel()) {
    for (auto lv = level.begin() + 2; lv != level.end(); lv++) {
      if (!(*lv)->BackUpdate(FrontPath(1))) {
	break;
      }
    }
  }
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
 void Bottom::ReachingPath(unsigned int levelIdx, unsigned int parIdx, unsigned int start, unsigned int extent, unsigned int ptId, unsigned int path) {
  for (unsigned int backLevel = 0; backLevel < level.size() - 1; backLevel++) {
    history[levelIdx + frontCount * backLevel] = backLevel == 0 ? parIdx : historyPrev[parIdx + splitPrev * (backLevel - 1)];
  }

  Inherit(levelIdx, parIdx);
  levelFront->Ancestor(levelIdx, start, extent);
  
  // Places <levelIdx, start> pair at appropriate position in every
  // reaching path.
  //
  for (unsigned int i = 1; i < level.size(); i++) {
    level[i]->PathInit(this, levelIdx, path, start, extent, RelBase(ptId));
  }
}
  /**
     @brief Updates both subtree- and node-relative paths for a live index.

     @param ndx is a node-relative index from the previous level.

     @param stx is the associated subtree-relative index.

     @param path is the path reaching the target node.

     @param targIdx is the updated node-relative index:  current level.

     @param ndBase is the base index of the target node:  current level.

     @return void.
   */
void Bottom::SetLive(unsigned int ndx, unsigned int stx, unsigned int path, unsigned int targIdx, unsigned int ndBase) {
  // Can omit if node-relative indexing is precluded.
  //
  levelFront->SetLive(ndx, path, targIdx, ndBase);

  // Subtree-relative paths no longer needed when all levels employ
  // node-relative restaging.
  //
  if (!level.back()->NodeRel()) {
    stPath->SetLive(stx, path, targIdx);  // Irregular write.
  }
}


/**
   @brief Terminates subtree-relative path for an extinct index.  Also
   terminates node-relative path if currently live.

   @param relIdx is a node-relative index.

   @return void.
 */
void Bottom::SetExtinct(unsigned int idx) {
  levelFront->SetExtinct(idx);

  unsigned int stIdx = rel2ST[idx];
  termST[termTop++] = stIdx;
  if (!level.back()->NodeRel()) {
    stPath->SetExtinct(stIdx);
  }
}


void Level::PathInit(const Bottom *bottom, unsigned int levelIdx, unsigned int path, unsigned int start, unsigned int extent, unsigned int relBase) {
  unsigned int mrraIdx = bottom->History(levelIdx, del);
  unsigned int pathOff = BackScale(mrraIdx);
  unsigned int pathBits = path & PathMask();
  nodePath[pathOff + pathBits].Init(levelIdx, start, extent, relBase);
  liveCount[mrraIdx]++;
}


void Level::SetExtinct(unsigned int idx) {
  if (idx < idxLive) { // May have been set during Replay().
    relPath->SetExtinct(idx);
  }
}


/**
   @brief Sets path, target and node-relative offse.

   @return void.
 */
void Level::SetLive(unsigned int idx, unsigned int path, unsigned int targIdx, unsigned int ndBase) {
  relPath->SetLive(idx, path, targIdx, nodeRel ? targIdx - ndBase : 0);
}

  
/**
   @brief Revises node-relative indices, as appropriae.  Irregular,
   but data locality improves with tree depth.

   @param one2Front maps first level to front indices.

   @return true iff level employs node-relative indexing.
 */
bool Level::BackUpdate(const IdxPath *one2Front) {
  if (!nodeRel)
    return false;

  relPath->BackUpdate(one2Front);
  return true;
}

  
