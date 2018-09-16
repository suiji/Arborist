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
#include "splitnode.h"
#include "splitcand.h"
#include "samplepred.h"
#include "sample.h"
#include "framemap.h"
#include "runset.h"
#include "rowrank.h"
#include "path.h"

#include <numeric>
#include <algorithm>


Bottom::Bottom(const FrameTrain *_frameTrain,
               const RowRank *_rowRank,
               SplitNode *_splitNode,
               vector<StageCount> &stageCount,
               unsigned int _bagCount) :
  nPred(_frameTrain->getNPred()),
  nPredFac(_frameTrain->getNPredFac()),
  bagCount(_bagCount),
  stPath(new IdxPath(bagCount)),
  splitPrev(0), splitCount(1),
  frameTrain(_frameTrain),
  rowRank(_rowRank),
  noRank(rowRank->NoRank()),
  splitNode(_splitNode),
  run(splitNode->getRuns()),
  history(vector<unsigned int>(0)),
  levelDelta(vector<unsigned char>(nPred)),
  levelFront(new Level(1, nPred,rowRank->DenseIdx(), rowRank->NPredDense(), bagCount, bagCount, false, this)),
  runCount(vector<unsigned int>(nPredFac))
{
  level.push_front(levelFront);
  levelFront->Ancestor(0, 0, bagCount);
  fill(levelDelta.begin(), levelDelta.end(), 0);
  fill(runCount.begin(), runCount.end(), 0);
  RootDef(stageCount);
}


void Bottom::RootDef(const vector<StageCount> &stageCount) {
  const unsigned int bufIdx = 0; // Initial staging buffer index.
  const unsigned int splitIdx = 0;
  for (unsigned int predIdx = 0; predIdx < stageCount.size(); predIdx++) {
    bool singleton = stageCount[predIdx].singleton;
    unsigned int expl = stageCount[predIdx].expl;
    (void) levelFront->Define(splitIdx, predIdx, bufIdx, singleton, bagCount - expl);
    setRunCount(splitIdx, predIdx, false, singleton ? 1 : frameTrain->FacCard(predIdx));
  }
}


vector<SplitCand> Bottom::split(SamplePred *samplePred,
                                 IndexLevel *index) {
  unsigned int supUnFlush = flushRear();
  levelFront->candidates(index, splitNode);

  backdate();
  Restage(samplePred);

  // Reaching levels must persist through restaging ut allow path lookup.
  //
  for (unsigned int off = level.size() -1 ; off > supUnFlush; off--) {
    delete level[off];
    level.pop_back();
  }
  splitNode->scheduleSplits(index, levelFront);

  return move(splitNode->split(samplePred));
}



unsigned int Bottom::flushRear() {
  unsigned int supUnFlush = level.size() - 1;

  // Capacity:  1 front level + 'pathMax' back levels.
  // If at capacity, every reaching definition should be flushed
  // to current level ut avoid falling off the deque.
  // Flushing prior to split assignment, rather than during, should
  // also save lookup time, as all definitions reaching from rear are
  // now at current level.
  //
  if (!NodePath::isRepresentable(level.size())) {
    level.back()->flush();
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
    backDef += level[off]->getDefCount();
  }

  unsigned int thresh = backDef * efficiency;
  for (unsigned int off = supUnFlush; off > 0; off--) {
    if (level[off]->getDefCount() <= thresh) {
      thresh -= level[off]->getDefCount();
      level[off]->flush();
      supUnFlush--;
    }
    else {
      break;
    }
  }

  return supUnFlush;
}


void Bottom::ScheduleRestage(unsigned int del,
                             unsigned int mrraIdx,
                             unsigned int predIdx,
                             unsigned bufIdx) {
  SPPair mrra = make_pair(mrraIdx, predIdx);
  RestageCoord rsCoord;
  rsCoord.Init(mrra, del, bufIdx);
  restageCoord.push_back(rsCoord);
}


Bottom::~Bottom() {
  for (auto *defLevel : level) {
    defLevel->flush(false);
    delete defLevel;
  }
  level.clear();

  delete stPath;
}


void Bottom::Restage(SamplePred *samplePred) {
  int nodeIdx;

#pragma omp parallel default(shared) private(nodeIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (nodeIdx = 0; nodeIdx < int(restageCoord.size()); nodeIdx++) {
      Restage(samplePred, restageCoord[nodeIdx]);
    }
  }

  restageCoord.clear();
}


void Bottom::Restage(SamplePred *samplePred, RestageCoord &rsCoord) {
  unsigned int del, bufIdx;
  SPPair mrra;
  rsCoord.Ref(mrra, del, bufIdx);
  samplePred->Restage(level[del], levelFront, mrra, bufIdx);
}


bool Bottom::factorStride(unsigned int predIdx,
                          unsigned int nStride,
                          unsigned int &facStride) const {
  bool isFactor;
  facStride = frameTrain->FacStride(predIdx, nStride, isFactor);
  return isFactor;
}


void Bottom::levelInit(IndexLevel *index) {
  splitNode->levelInit(index);
}


void Bottom::levelClear() {
  splitNode->levelClear();
}


void Bottom::Overlap(unsigned int splitNext,
                     unsigned int idxLive,
                     bool nodeRel) {
  splitPrev = splitCount;
  splitCount = splitNext;
  if (splitCount == 0) // No further splitting or restaging.
    return;

  levelFront = new Level(splitCount, nPred, rowRank->DenseIdx(), rowRank->NPredDense(), bagCount, idxLive, nodeRel, this);
  level.push_front(levelFront);

  historyPrev = move(history);
  history = move(vector<unsigned int>(splitCount * (level.size()-1)));

  deltaPrev = move(levelDelta);
  levelDelta = move(vector<unsigned char>(splitCount * nPred));

  runCount = move(vector<unsigned int>(splitCount * nPredFac));
  fill(runCount.begin(), runCount.end(), 0);

  // Recomputes paths reaching from non-front levels.
  //
  for (unsigned int i = 1; i < level.size(); i++) {
    level[i]->Paths();
  }
}


void Bottom::backdate() const {
  if (level.size() > 2 && level[1]->isNodeRel()) {
    for (auto lv = level.begin() + 2; lv != level.end(); lv++) {
      if (!(*lv)->backdate(FrontPath(1))) {
        break;
      }
    }
  }
}

  
void Bottom::reachingPath(unsigned int levelIdx,
                          unsigned int parIdx,
                          unsigned int start,
                          unsigned int extent,
                          unsigned int relBase,
                          unsigned int path) {
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


void Bottom::setLive(unsigned int ndx,
                     unsigned int targIdx,
                     unsigned int stx,
                     unsigned int path,
                     unsigned int ndBase) {
  levelFront->setLive(ndx, path, targIdx, ndBase);

  if (!level.back()->isNodeRel()) {
    stPath->setLive(stx, path, targIdx);  // Irregular write.
  }
}


void Bottom::setExtinct(unsigned int nodeIdx,
                        unsigned int stIdx) {
  levelFront->setExtinct(nodeIdx);
  setExtinct(stIdx);
}


void Bottom::setExtinct(unsigned int stIdx) {
  if (!level.back()->isNodeRel()) {
    stPath->setExtinct(stIdx);
  }
}


unsigned int Bottom::getSplitCount(unsigned int del) const {
  return level[del]->getSplitCount();
}


void Bottom::AddDef(unsigned int reachIdx,
                    unsigned int predIdx,
                    unsigned int bufIdx,
                    bool singleton) {
  if (levelFront->Define(reachIdx, predIdx, bufIdx, singleton)) {
    levelDelta[reachIdx * nPred + predIdx] = 0;
  }
}
  

unsigned int Bottom::History(const Level *reachLevel,
                             unsigned int splitIdx) const {
  return reachLevel == levelFront ? splitIdx : history[splitIdx + (reachLevel->Del() - 1) * splitCount];
}


/**
   Passes through to front level.
 */
unsigned int Bottom::adjustDense(unsigned int levelIdx,
                                 unsigned int predIdx,
                                 unsigned int &startIdx,
                                 unsigned int &extent) const {
    return levelFront->adjustDense(levelIdx, predIdx, startIdx, extent);
}


IdxPath *Bottom::FrontPath(unsigned int del) const {
  return level[del]->FrontPath();
}


/**
   Passes through to front level.
 */
bool Bottom::isSingleton(unsigned int levelIdx,
                       unsigned int predIdx) const {
  return levelFront->isSingleton(levelIdx, predIdx);
}


void Bottom::setSingleton(unsigned int splitIdx,
                          unsigned int predIdx) const {
  levelFront->setSingleton(splitIdx, predIdx);
}


void Bottom::reachFlush(unsigned int splitIdx,
                        unsigned int predIdx) const {
  Level *reachingLevel = reachLevel(splitIdx, predIdx);
  reachingLevel->flushDef(History(reachingLevel, splitIdx), predIdx);
}
