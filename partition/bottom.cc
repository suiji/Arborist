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
#include "frontier.h"
#include "splitfrontier.h"
#include "summaryframe.h"
#include "rankedframe.h"
#include "path.h"
#include "stagecount.h"
#include "ompthread.h"

#include <numeric>
#include <algorithm>


Bottom::Bottom(const SummaryFrame* frame_,
               IndexT bagCount) :
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
  IndexRange bufRange;
  bufRange.set(0, bagCount);
  level[0]->initAncestor(0, bufRange);
  fill(levelDelta.begin(), levelDelta.end(), 0);
  fill(runCount.begin(), runCount.end(), 0);
}


void
Bottom::rootDef(const vector<StageCount>& stageCount,
		IndexT bagCount) {
  const unsigned int bufIdx = 0; // Initial staging buffer index.
  const IndexT splitIdx = 0; // Root split index.
  PredictorT predIdx = 0;
  for (auto sc : stageCount) {
    SplitCoord splitCoord(splitIdx, predIdx);
    (void) level[0]->define(splitCoord, bufIdx, sc.singleton, bagCount - sc.expl);
    setRunCount(splitCoord, false, sc.singleton ? 1 : frame->getCardinality(predIdx));
    predIdx++;
  }
}


void
Bottom::scheduleSplits(SplitFrontier* splitFrontier,
		       Frontier *frontier) {
  splitFrontier->init();
  unsigned int flushCount = flushRear();
  splitFrontier->candidates(frontier, this);
    //level[0]->candidates(frontier, splitFrontier);

  backdate();
  restage(splitFrontier);

  // Reaching levels must persist through restaging ut allow path lookup.
  //
  if (flushCount > 0) {
    level.erase(level.end() - flushCount, level.end());
  }
  splitFrontier->scheduleSplits(this);
}


bool
Bottom::isSingleton(const SplitCoord& splitCoord) const {
  return level[0]->isSingleton(splitCoord);
}


bool
Bottom::isSingleton(const SplitCoord& splitCoord,
		    unsigned int& bufIdx) const {
  return level[0]->isSingleton(splitCoord, bufIdx);
}


IndexRange
Bottom::adjustRange(const SplitNux* nux,
		    const Frontier* frontier) const {
  return level[0]->adjustRange(nux, frontier);
}


IndexT
Bottom::getImplicitCount(const SplitNux* nux) const {
  return level[0]->getImplicit(nux);
}



unsigned int
Bottom::flushRear() {
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

  // assert(unflushTop < level.size();
  return level.size() - 1 - unflushTop;
}


void
Bottom::scheduleRestage(unsigned int del,
			const SplitCoord& splitCoord,
			unsigned bufIdx) {
  restageCoord.emplace_back(RestageCoord(splitCoord, del, bufIdx));
}


Bottom::~Bottom() {
  for (auto & defLevel : level) {
    defLevel->flush(false);
  }
  level.clear();
}


void
Bottom::restage(const SplitFrontier* splitFrontier) {
  OMPBound idxTop = restageCoord.size();
  
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound nodeIdx = 0; nodeIdx < idxTop; nodeIdx++) {
      restage(splitFrontier, restageCoord[nodeIdx]);
    }
  }

  restageCoord.clear();
}


void
Bottom::restage(const SplitFrontier* splitFrontier,
		RestageCoord &rsCoord) {
  unsigned int del, bufIdx;
  SplitCoord mrra = rsCoord.Ref(del, bufIdx);
  splitFrontier->restage(level[del].get(), level[0].get(), mrra, bufIdx);
}


bool
Bottom::factorStride(PredictorT predIdx,
		     unsigned int nStride,
		     unsigned int& facStride) const {
  bool isFactor;
  facStride = frame->getFacStride(predIdx, nStride, isFactor);
  return isFactor;
}


void
Bottom::overlap(IndexT splitNext,
                IndexT bagCount,
                IndexT idxLive,
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

  runCount = vector<PredictorT>(splitCount * nPredFac);
  fill(runCount.begin(), runCount.end(), 0);

  for (auto lv = level.begin() + 1; lv != level.end(); lv++) {
    (*lv)->reachingPaths();
  }
}


void
Bottom::backdate() const {
  if (level.size() > 2 && level[1]->isNodeRel()) {
    for (auto lv = level.begin() + 2; lv != level.end(); lv++) {
      if (!(*lv)->backdate(getFrontPath(1))) {
        break;
      }
    }
  }
}

  
void
Bottom::reachingPath(IndexT splitIdx,
		     IndexT parIdx,
		     const IndexRange& bufRange,
		     IndexT relBase,
		     unsigned int path) {
  for (unsigned int backLevel = 0; backLevel < level.size() - 1; backLevel++) {
    history[splitIdx + splitCount * backLevel] = backLevel == 0 ? parIdx : historyPrev[parIdx + splitPrev * (backLevel - 1)];
  }

  inherit(splitIdx, parIdx);
  level[0]->initAncestor(splitIdx, bufRange);
  
  // Places <splitIdx, start> pair at appropriate position in every
  // reaching path.
  //
  for (auto lv = level.begin() + 1; lv != level.end(); lv++) {
    (*lv)->pathInit(splitIdx, path, bufRange, relBase);
  }
}


void
Bottom::setLive(IndexT ndx,
		IndexT targIdx,
		IndexT stx,
		unsigned int path,
		IndexT ndBase) {
  level[0]->setLive(ndx, path, targIdx, ndBase);

  if (!level.back()->isNodeRel()) {
    stPath->setLive(stx, path, targIdx);  // Irregular write.
  }
}


void
Bottom::setExtinct(IndexT nodeIdx,
		   IndexT stIdx) {
  level[0]->setExtinct(nodeIdx);
  setExtinct(stIdx);
}


void
Bottom::setExtinct(IndexT stIdx) {
  if (!level.back()->isNodeRel()) {
    stPath->setExtinct(stIdx);
  }
}


IndexT
Bottom::getSplitCount(unsigned int del) const {
  return level[del]->getSplitCount();
}


void
Bottom::addDef(const SplitCoord& splitCoord,
	       unsigned int bufIdx,
	       bool singleton) {
  if (level[0]->define(splitCoord, bufIdx, singleton)) {
    levelDelta[splitCoord.strideOffset(nPred)] = 0;
  }
}
  

IndexT
Bottom::getHistory(const Level *reachLevel,
                   IndexT splitIdx) const {
  return reachLevel == level[0].get() ? splitIdx : history[splitIdx + (reachLevel->getDel() - 1) * splitCount];
}


SplitCoord
Bottom::getHistory(const Level* reachLevel,
		   const SplitCoord& coord) const {
  return reachLevel == level[0].get() ? coord :
    SplitCoord(history[coord.nodeIdx + splitCount * (reachLevel->getDel() - 1)], coord.predIdx);
}

const IdxPath*
Bottom::getFrontPath(unsigned int del) const {
  return level[del]->getFrontPath();
}


void
Bottom::setSingleton(const SplitCoord& splitCoord) const {
  level[0]->setSingleton(splitCoord);
}


void
Bottom::reachFlush(const SplitCoord& splitCoord) const {
  Level *reachingLevel = reachLevel(splitCoord);
  reachingLevel->flushDef(getHistory(reachingLevel, splitCoord));
}
