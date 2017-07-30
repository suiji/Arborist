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
#include "level.h"
#include "bv.h"
#include "index.h"
#include "splitpred.h"
#include "samplepred.h"
#include "sample.h"
#include "predblock.h"
#include "runset.h"
#include "rowrank.h"
#include "path.h"
#include "splitsig.h"

#include <numeric>
#include <algorithm>

// Testing only:
//#include <iostream>
//using namespace std;
//#include <time.h>


/**
   @brief Class constructor.

   @param bagCount enables sizing of predicate bit vectors.

   @param splitCount specifies the number of splits to map.
 */
Bottom::Bottom(const PMTrain *_pmTrain, const RowRank *_rowRank, SplitPred *_splitPred, SamplePred *samplePred, unsigned int _bagCount) : nPred(_pmTrain->NPred()), nPredFac(_pmTrain->NPredFac()), bagCount(_bagCount), stPath(new IdxPath(bagCount)), splitPrev(0), splitCount(1), pmTrain(_pmTrain), rowRank(_rowRank), noRank(rowRank->NoRank()), splitPred(_splitPred), run(splitPred->Runs()), history(std::vector<unsigned int>(0)), levelDelta(std::vector<unsigned char>(nPred)), levelFront(new Level(1, nPred, rowRank->DenseIdx(), rowRank->NPredDense(), bagCount, bagCount, false, this, samplePred)), runCount(std::vector<unsigned int>(nPredFac)) {
  level.push_front(levelFront);
  levelFront->Ancestor(0, 0, bagCount);
  std::fill(levelDelta.begin(), levelDelta.end(), 0);
  std::fill(runCount.begin(), runCount.end(), 0);
}


/**
   @brief Adds a new definition at the root level.

   @param predIdx is the predictor index.

   @param expl is the number of explicitly-staged indices.

   @param singleton is true iff column consists of indentically-ranked samples.

   @return void.
 */
void Bottom::RootDef(unsigned int predIdx, unsigned int expl, bool singleton) {
  const unsigned int bufIdx = 0; // Initial staging buffer index.
  const unsigned int levelIdx = 0;
  (void) levelFront->Define(levelIdx, predIdx, bufIdx, singleton, bagCount - expl);
  SetRunCount(levelIdx, predIdx, false, singleton ? 1 : pmTrain->FacCard(predIdx));
}

  
/**
   @brief Entry to spltting and restaging.

   @return void, with output vector of splitting signatures.
 */
void Bottom::Split(const SamplePred *samplePred, IndexLevel *index, std::vector<SSNode> &argMax) {
  unsigned int supUnFlush = FlushRear();
  levelFront->Candidates(index, splitPred);

  Backdate();
  Restage();

  // Reaching levels must persist through restaging ut allow path lookup.
  //
  for (unsigned int off = level.size() -1 ; off > supUnFlush; off--) {
    delete level[off];
    level.pop_back();
  }
  splitPred->ScheduleSplits(index, levelFront);
  splitPred->Split(samplePred, argMax);
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
    level.back()->Flush();
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
      level[off]->Flush();
      supUnFlush--;
    }
    else {
      break;
    }
  }

  return supUnFlush;
}


void Bottom::ScheduleRestage(unsigned int del, unsigned int mrraIdx, unsigned int predIdx, unsigned bufIdx) {
  SPPair mrra = std::make_pair(mrraIdx, predIdx);
  RestageCoord rsCoord;
  rsCoord.Init(mrra, del, bufIdx);
  restageCoord.push_back(rsCoord);
}


/**
   @brief Class finalizer.
 */
Bottom::~Bottom() {
  for (auto *defLevel : level) {
    defLevel->Flush(false);
    delete defLevel;
  }
  level.clear();

  delete stPath;
  delete splitPred;
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
  level[del]->Restage(mrra, levelFront, bufIdx);
}


void Bottom::IndexRestage() {
  int nodeIdx;

#pragma omp parallel default(shared) private(nodeIdx)
  {
#pragma omp for schedule(dynamic, 1)    
    for (nodeIdx = 0; nodeIdx < int(restageCoord.size()); nodeIdx++) {
      IndexRestage(restageCoord[nodeIdx]);
    }
  }
}


void Bottom::IndexRestage(RestageCoord &rsCoord) {
  unsigned int del, bufIdx;
  SPPair mrra;
  rsCoord.Ref(mrra, del, bufIdx);
  level[del]->IndexRestage(mrra, levelFront, bufIdx);
}


bool Bottom::IsFactor(unsigned int predIdx) const {
  return pmTrain->IsFactor(predIdx);
}



unsigned int Bottom::FacIdx(unsigned int predIdx, bool &isFactor) const {
  return pmTrain->BlockIdx(predIdx, isFactor);
}


/**
   @brief Passes through to SplitPred's level initializer.

   @return void.
 */
void Bottom::LevelInit(IndexLevel *index) {
  splitPred->LevelInit(index);
}


/**
   @brief Passes through to SplitPred's level clearer.

   @return void.
 */
void Bottom::LevelClear() {
  splitPred->LevelClear();
}


/**
   @brief Updates subtree and pretree mappings from temporaries constructed
   during the overlap.  Initializes data structures for restaging and
   splitting the current level of the subtree.

   @param splitNext is the number of splitable nodes in the current
   subtree level.

   @param idxMax is the maximum index width among live nodes.

   @return true iff front level employs node-relative indexing.
 */
void Bottom::Overlap(SamplePred *samplePred, unsigned int splitNext, unsigned int idxLive, bool nodeRel) {
  splitPrev = splitCount;
  splitCount = splitNext;
  if (splitCount == 0) // No further splitting or restaging.
    return;

  levelFront = new Level(splitCount, nPred, rowRank->DenseIdx(), rowRank->NPredDense(), bagCount, idxLive, nodeRel, this, samplePred);
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
   @brief Consumes all fields in current NodeCache item relevant to restaging.

   @param par is the index of the parent.

   @param path is a unique path identifier.

   @param levelIdx is the index of the heir.

   @param start is the cell starting index.

   @param extent is the index count.

   @return void.
*/
void Bottom::ReachingPath(unsigned int levelIdx, unsigned int parIdx, unsigned int start, unsigned int extent, unsigned int relBase, unsigned int path) {
  for (unsigned int backLevel = 0; backLevel < level.size() - 1; backLevel++) {
    history[levelIdx + splitCount * backLevel] = backLevel == 0 ? parIdx : historyPrev[parIdx + splitPrev * (backLevel - 1)];
  }

  Inherit(levelIdx, parIdx);
  levelFront->Ancestor(levelIdx, start, extent);
  
  // Places <levelIdx, start> pair at appropriate position in every
  // reaching path.
  //
  for (unsigned int i = 1; i < level.size(); i++) {
    level[i]->PathInit(this, levelIdx, path, start, extent, relBase);
  }
}


/**
     @brief Updates both node-relative path for a live index, as
     well as subtree-relative if back levels warrant.

     @param ndx is a node-relative index from the previous level.

     @param targIdx is the updated node-relative index:  current level.

     @param stx is the associated subtree-relative index.

     @param path is the path reaching the target node.

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
    @brief Terminates node-relative path an extinct index.  Also
    terminates subtree-relative path if currently live.

    @param nodeIdx is a node-relative index.

    @return void.
*/
void Bottom::SetExtinct(unsigned int nodeIdx, unsigned int stIdx) {
  levelFront->SetExtinct(nodeIdx);
  SetExtinct(stIdx);
}


/**
   @brief Marks subtree-relative path as extinct, as required by back levels.
 */
void Bottom::SetExtinct(unsigned int stIdx) {
  if (!level.back()->NodeRel()) {
    stPath->SetExtinct(stIdx);
  }
}


unsigned int Bottom::SplitCount(unsigned int del) const {
  return level[del]->SplitCount();
}


/**
    @brief Flips source bit if a definition reaches to current level.

   @return void
*/
void Bottom::AddDef(unsigned int reachIdx, unsigned int predIdx, unsigned int bufIdx, bool singleton) {
  if (levelFront->Define(reachIdx, predIdx, bufIdx, singleton)) {
    levelDelta[reachIdx * nPred + predIdx] = 0;
  }
}
  

  /**
     @brief Locates index of ancestor several levels back.

     @param splitIdx is descendant index.

     @param del is the number of levels back.

     @return index of ancestor node.
   */
unsigned int Bottom::History(const Level *reachLevel, unsigned int splitIdx) const {
  return reachLevel == levelFront ? splitIdx : history[splitIdx + (reachLevel->Del() - 1) * splitCount];
}


unsigned int Bottom::AdjustDense(unsigned int levelIdx, unsigned int predIdx, unsigned int &startIdx, unsigned int &extent) const {
    return levelFront->AdjustDense(levelIdx, predIdx, startIdx, extent);
  }


IdxPath *Bottom::FrontPath(unsigned int del) const {
  return level[del]->FrontPath();
}


/**
     @brief Determines whether front-level pair is a singleton.

     @return true iff the pair is a singleton.
   */

bool Bottom::Singleton(unsigned int levelIdx, unsigned int predIdx) const {
  return levelFront->Singleton(levelIdx, predIdx);
}


void Bottom::SetSingleton(unsigned int splitIdx, unsigned int predIdx) const {
  levelFront->SetSingleton(splitIdx, predIdx);
}


/**
   @brief Flushes MRRA for a pair and instantiates definition at front level.
 */
void Bottom::ReachFlush(unsigned int splitIdx, unsigned int predIdx) const {
  Level *reachLevel = ReachLevel(splitIdx, predIdx);
  reachLevel->FlushDef(History(reachLevel, splitIdx), predIdx);
}


/**
   @brief Passes pre-bias computation for an index set through to SplitPred.

   @param sum is the sum of responses over an index set.

   @param sCount is the sum of sampled indices over an index set.

   @return pre-bias value derived from SplitPred.
 */
double Bottom::Prebias(unsigned int splitIdx, double sum, unsigned int sCount) const {
  return splitPred->Prebias(splitIdx, sum, sCount);
}
