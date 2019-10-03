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
#include "splitfrontier.h"
#include "splitnux.h"
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
  const unsigned int bufRoot = 0; // Initial staging buffer index.
  const IndexT splitIdx = 0; // Root split index.
  PredictorT predIdx = 0;
  for (auto sc : stageCount) {
    SplitCoord splitCoord(splitIdx, predIdx);
    (void) level[0]->define(DefCoord(splitCoord, bufRoot), sc.singleton, bagCount - sc.expl);
    setRunCount(splitCoord, false, sc.singleton ? 1 : frame->getCardinality(predIdx));
    predIdx++;
  }
}


void
Bottom::scheduleSplits(SplitFrontier* splitFrontier) {
  splitFrontier->init();
  unsigned int flushCount = flushRear();
  vector<DefCoord> preCand = splitFrontier->precandidates(this);

  backdate();
  restage(splitFrontier);

  // Reaching levels must persist through restaging ut allow path lookup.
  //
  if (flushCount > 0) {
    level.erase(level.end() - flushCount, level.end());
  }
  vector<SplitNux> postCand = postSchedule(splitFrontier, preCand);
  splitFrontier->split(postCand);
}


vector<SplitNux>
Bottom::postSchedule(SplitFrontier* splitFrontier, vector<DefCoord>& preCand) {
  vector<PredictorT> runCount;
  vector<SplitNux> postCand;
  vector<PredictorT> nCand(splitFrontier->getNSplit());
  fill(nCand.begin(), nCand.end(), 0);
  for (auto & pc : preCand) {
    postSchedule(splitFrontier, pc, runCount, nCand, postCand);
  }

  splitFrontier->setCandOff(nCand);
  splitFrontier->setRunOffsets(runCount);

  return postCand;
}


void
Bottom::postSchedule(const SplitFrontier* splitFrontier,
		     const DefCoord& preCand,
		     vector<PredictorT>& runCount,
		     vector<PredictorT>& nCand,
		     vector<SplitNux>& postCand) const {
  SplitCoord splitCoord = preCand.splitCoord;
  if (!isSingleton(splitCoord)) {
    PredictorT setIdx = getSetIdx(splitFrontier, splitCoord, runCount);
    postCand.emplace_back(preCand, splitFrontier, setIdx, adjustRange(preCand, splitFrontier), getImplicitCount(preCand));
    nCand[splitCoord.nodeIdx]++;
  }
}


PredictorT
Bottom::getSetIdx(const SplitFrontier* splitFrontier,
		  const SplitCoord& splitCoord,
		  vector<PredictorT>& outCount) const {
  IndexT facStride;
  PredictorT rCount = factorStride(splitCoord.predIdx, splitCoord.nodeIdx, facStride) ? runCount[facStride] : 0;
  PredictorT setIdx;
  if (rCount > 1) {
    setIdx = outCount.size();
    outCount.push_back(rCount);
  }
  else {
    setIdx = splitFrontier->getNoSet();
  }
  return setIdx;
}


unsigned int
Bottom::preschedule(SplitFrontier* splitFrontier,
		    const SplitCoord& splitCoord,
		    vector<DefCoord>& preCand) const {
  reachFlush(splitCoord);
  DefCoord defCoord(splitCoord, 0); // Dummy initialization.
  if (!isSingleton(splitCoord, defCoord)) {
    splitFrontier->preschedule(defCoord, preCand);
    return 1;
  }
  else {
    return 0;
  }
}


bool
Bottom::isSingleton(const SplitCoord& splitCoord) const {
  return level[0]->isSingleton(splitCoord);
}


bool
Bottom::isSingleton(const SplitCoord& splitCoord,
		    DefCoord& defCoord) const {
  return level[0]->isSingleton(splitCoord, defCoord);
}


IndexT
Bottom::getImplicitCount(const DefCoord& preCand) const {
  return level[0]->getImplicit(preCand);
}


IndexRange
Bottom::adjustRange(const DefCoord& preCand,
		    const SplitFrontier* splitFrontier) const {
  return level[0]->adjustRange(preCand, splitFrontier);
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

  IndexT backDef = 0;
  for (auto lv = level.begin() + unflushTop; lv != level.begin(); lv--) {
    backDef += (*lv)->getDefCount();
  }

  IndexT thresh = backDef * efficiency;
  for (auto lv = level.begin() + unflushTop; lv != level.begin(); lv--) {
    if ((*lv)->flush(thresh)) {
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
			const DefCoord& defCoord) {
  restageCoord.emplace_back(RestageCoord(defCoord, del));
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
		RestageCoord& rsCoord) {
  unsigned int del;
  DefCoord mrra = rsCoord.Ref(del);
  splitFrontier->restage(level[del].get(), level[0].get(), mrra);
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
Bottom::addDef(const DefCoord& defCoord,
	       bool singleton) {
  if (level[0]->define(defCoord, singleton)) {
    levelDelta[defCoord.splitCoord.strideOffset(nPred)] = 0;
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
