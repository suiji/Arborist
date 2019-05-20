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
#include "summaryframe.h"
#include "runset.h"
#include "rankedframe.h"
#include "path.h"
#include "ompthread.h"

#include <numeric>
#include <algorithm>


Bottom::Bottom(const SummaryFrame* frame_,
               unsigned int bagCount) :
  frame(frame_),
  nPred(frame->getNPred()),
  nPredFac(frame->getNPredFac()),
  stPath(make_unique<IdxPath>(bagCount)),
  splitPrev(0), splitCount(1),
  rankedFrame(frame->getRankedFrame()),
  noRank(rankedFrame->NoRank()),
  history(vector<unsigned int>(0)),
  levelDelta(vector<unsigned char>(nPred)),
  runCount(vector<unsigned int>(nPredFac))
{

  level.push_front(make_unique<Level>(1, nPred, rankedFrame, bagCount, bagCount, false, this));
  level[0]->initAncestor(0, 0, bagCount);
  fill(levelDelta.begin(), levelDelta.end(), 0);
  fill(runCount.begin(), runCount.end(), 0);
}


void Bottom::rootDef(const vector<StageCount>& stageCount, unsigned int bagCount) {
  const unsigned int bufIdx = 0; // Initial staging buffer index.
  const unsigned int splitIdx = 0; // Root split index.
  unsigned int predIdx = 0;
  for (auto sc : stageCount) {
    (void) level[0]->define(splitIdx, predIdx, bufIdx, sc.singleton, bagCount - sc.expl);
    setRunCount(splitIdx, predIdx, false, sc.singleton ? 1 : frame->getCardinality(predIdx));
    predIdx++;
  }
}


void Bottom::scheduleSplits(SamplePred *samplePred,
                            SplitNode* splitNode,
                            IndexLevel *index) {
  splitNode->levelInit(index);
  unsigned int unflushTop = flushRear();
  level[0]->candidates(index, splitNode);

  backdate();
  restage(samplePred);

  // Reaching levels must persist through restaging ut allow path lookup.
  //
  for (unsigned int off = level.size() - 1; off > unflushTop; off--) {
    level.erase(level.end());
  }
  splitNode->scheduleSplits(index, level[0].get());
}



unsigned int Bottom::flushRear() {
  unsigned int unflushTop = level.size() - 1;

  // Capacity:  1 front level + 'pathMax' back levels.
  // If at capacity, every reaching definition should be flushed
  // to current level ut avoid falling off the deque.
  // Flushing prior to split assignment, rather than during, should
  // also save lookup time, as all definitions reaching from rear are
  // now at current level.
  //
  if (!NodePath::isRepresentable(level.size())) {
    level.back()->flush();
    unflushTop--;
  }

  // Walks backward from rear, purging non-reaching definitions.
  // Stops when a level with no non-reaching nodes is encountered.
  //
  for (unsigned int off = unflushTop; off > 0; off--) {
    if (!level[off]->nonreachPurge())
      break;
  }

  unsigned int backDef = 0;
  for (auto lv = level.begin() + unflushTop; lv != level.begin(); lv--) {
    backDef += (*lv)->getDefCount();
  }

  unsigned int thresh = backDef * efficiency;
  for (auto lv = level.begin() + unflushTop; lv != level.begin(); lv--) {
    if ((*lv)->getDefCount() <= thresh) {
      thresh -= (*lv)->getDefCount();
      (*lv)->flush();
      unflushTop--;
    }
    else {
      break;
    }
  }

  return unflushTop;
}


void Bottom::scheduleRestage(unsigned int del,
                             unsigned int mrraIdx,
                             unsigned int predIdx,
                             unsigned bufIdx) {
  SPPair mrra = make_pair(mrraIdx, predIdx);
  RestageCoord rsCoord;
  rsCoord.init(mrra, del, bufIdx);
  restageCoord.push_back(rsCoord);
}


Bottom::~Bottom() {
  for (auto & defLevel : level) {
    defLevel->flush(false);
  }
  level.clear();
}


void Bottom::restage(SamplePred *samplePred) {
  OMPBound nodeIdx;
  OMPBound idxTop = restageCoord.size();
  
#pragma omp parallel default(shared) private(nodeIdx) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (nodeIdx = 0; nodeIdx < idxTop; nodeIdx++) {
      restage(samplePred, restageCoord[nodeIdx]);
    }
  }

  restageCoord.clear();
}


void Bottom::restage(SamplePred *samplePred, RestageCoord &rsCoord) {
  unsigned int del, bufIdx;
  SPPair mrra;
  rsCoord.Ref(mrra, del, bufIdx);
  samplePred->restage(level[del].get(), level[0].get(), mrra, bufIdx);
}


bool Bottom::factorStride(unsigned int predIdx,
                          unsigned int nStride,
                          unsigned int &facStride) const {
  bool isFactor;
  facStride = frame->getFacStride(predIdx, nStride, isFactor);
  return isFactor;
}


void Bottom::overlap(unsigned int splitNext,
                     unsigned int bagCount,
                     unsigned int idxLive,
                     bool nodeRel) {
  splitPrev = splitCount;
  splitCount = splitNext;
  if (splitCount == 0) // No further splitting or restaging.
    return;

  level.push_front(make_unique<Level>(splitCount, nPred, rankedFrame, bagCount, idxLive, nodeRel, this));

  historyPrev = move(history);
  history = vector<unsigned int>(splitCount * (level.size()-1));

  deltaPrev = move(levelDelta);
  levelDelta = vector<unsigned char>(splitCount * nPred);

  runCount = vector<unsigned int>(splitCount * nPredFac);
  fill(runCount.begin(), runCount.end(), 0);

  for (auto lv = level.begin() + 1; lv != level.end(); lv++) {
    (*lv)->reachingPaths();
  }
}


void Bottom::backdate() const {
  if (level.size() > 2 && level[1]->isNodeRel()) {
    for (auto lv = level.begin() + 2; lv != level.end(); lv++) {
      if (!(*lv)->backdate(getFrontPath(1))) {
        break;
      }
    }
  }
}

  
void Bottom::reachingPath(unsigned int splitIdx,
                          unsigned int parIdx,
                          unsigned int start,
                          unsigned int extent,
                          unsigned int relBase,
                          unsigned int path) {
  for (unsigned int backLevel = 0; backLevel < level.size() - 1; backLevel++) {
    history[splitIdx + splitCount * backLevel] = backLevel == 0 ? parIdx : historyPrev[parIdx + splitPrev * (backLevel - 1)];
  }

  inherit(splitIdx, parIdx);
  level[0]->initAncestor(splitIdx, start, extent);
  
  // Places <splitIdx, start> pair at appropriate position in every
  // reaching path.
  //
  for (auto lv = level.begin() + 1; lv != level.end(); lv++) {
    (*lv)->pathInit(this, splitIdx, path, start, extent, relBase);
  }
}


void Bottom::setLive(unsigned int ndx,
                     unsigned int targIdx,
                     unsigned int stx,
                     unsigned int path,
                     unsigned int ndBase) {
  level[0]->setLive(ndx, path, targIdx, ndBase);

  if (!level.back()->isNodeRel()) {
    stPath->setLive(stx, path, targIdx);  // Irregular write.
  }
}


void Bottom::setExtinct(unsigned int nodeIdx,
                        unsigned int stIdx) {
  level[0]->setExtinct(nodeIdx);
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


void Bottom::addDef(unsigned int reachIdx,
                    unsigned int predIdx,
                    unsigned int bufIdx,
                    bool singleton) {
  if (level[0]->define(reachIdx, predIdx, bufIdx, singleton)) {
    levelDelta[reachIdx * nPred + predIdx] = 0;
  }
}
  

unsigned int Bottom::getHistory(const Level *reachLevel,
                                unsigned int splitIdx) const {
  return reachLevel == level[0].get() ? splitIdx : history[splitIdx + (reachLevel->getDel() - 1) * splitCount];
}


/**
   Passes through to front level.
 */
unsigned int Bottom::adjustDense(unsigned int splitIdx,
                                 unsigned int predIdx,
                                 unsigned int &startIdx,
                                 unsigned int &extent) const {
    return level[0]->adjustDense(splitIdx, predIdx, startIdx, extent);
}


const IdxPath *Bottom::getFrontPath(unsigned int del) const {
  return level[del]->getFrontPath();
}


/**
   Passes through to front level.
 */
bool Bottom::isSingleton(unsigned int splitIdx,
                         unsigned int predIdx) const {
  return level[0]->isSingleton(splitIdx, predIdx);
}


void Bottom::setSingleton(unsigned int splitIdx,
                          unsigned int predIdx) const {
  level[0]->setSingleton(splitIdx, predIdx);
}


void Bottom::reachFlush(unsigned int splitIdx,
                        unsigned int predIdx) const {
  Level *reachingLevel = reachLevel(splitIdx, predIdx);
  reachingLevel->flushDef(getHistory(reachingLevel, splitIdx), predIdx);
}
