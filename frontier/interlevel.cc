// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file interlevel.cc

   @brief Tracks repartitioned cells on a per-level basis.

   @author Mark Seligman
 */

#include "ompthread.h"
#include "frontier.h"
#include "sampleobs.h"
#include "splitfrontier.h"
#include "interlevel.h"
#include "obsfrontier.h"
#include "splitnux.h"
#include "trainframe.h"
#include "layout.h"
#include "indexset.h"

#include <algorithm>


InterLevel::InterLevel(const TrainFrame* frame,
	       Frontier* frontier_) :
  nPred(frame->getNPred()),
  frontier(frontier_),
  positionMask(getPositionMask(nPred)),
  levelShift(getLevelShift(nPred)),
  bagCount(frontier->getBagCount()),
  rootPath(make_unique<IdxPath>(bagCount)),
  level(0),
  splitCount(1),
  layout(frame->getLayout()),
  obsPart(make_unique<ObsPart>(layout, bagCount)),
  stageMap(vector<vector<PredictorT>>(1)),
  ofFront(make_unique<ObsFrontier>(frontier, 1, nPred, bagCount, this)) {
  stageMap[0] = vector<PredictorT>(nPred);
}

bool InterLevel::isStaged(const SplitCoord& coord, StagedCell*& cell) {
  IndexT dummy;
  PredictorT stagePos;
  if (isStaged(coord, dummy, stagePos)) {
    cell = ofFront->getCellAddr(coord.nodeIdx, stagePos);
    return true;
  }
  else
    return false;
}


StagedCell*  InterLevel::getFrontCellAddr(const SplitCoord& coord) {
  unsigned int stageLevel;
  PredictorT predPos;
  if (isStaged(coord, stageLevel, predPos)) {
    if (stageLevel != level) {
      return nullptr;
    }
    else {
      return ofFront->getCellAddr(coord.nodeIdx, predPos);
    }
  }
  else {
    return nullptr;
  }
}


ObsPart* InterLevel::getObsPart() const {
  return obsPart.get();
}


IndexT* InterLevel::getIdxBuffer(const SplitNux* nux) const {
  return obsPart->getIdxBuffer(nux);
}


Obs* InterLevel::getPredBase(const SplitNux* nux) const {
  return obsPart->getPredBase(nux);
}


ObsFrontier* InterLevel::getFront() {
  return ofFront.get();
}


void InterLevel::repartition(Frontier* frontier,
			     const SampleObs* sampleObs) {
  // Precandidates precipitate restaging ancestors at this level,
  // as do all history flushes.
  vector<unsigned int> nExtinct;
  if (level == 0) {
    nExtinct = stage(sampleObs);
  }
  else {
    nExtinct = restage();
  }
  ofFront->prune(nExtinct);
}


bool InterLevel::preschedule(const SplitCoord& coord) {
  unsigned int stageLevel;
  PredictorT stagePos;
  if (isStaged(coord, stageLevel, stagePos)) {
    if (stageLevel != level) {
      history[level - stageLevel - 1]->prestageAncestor(ofFront.get(), coord.nodeIdx, stagePos);
    }
    return true;
  }
  return false;
}


void InterLevel::appendAncestor(StagedCell& scAnc, unsigned int historyIdx) {
  history[historyIdx]->delist(scAnc);
  ancestor.emplace_back(scAnc, historyIdx);
}


vector<unsigned int> InterLevel::stage(const SampleObs* sampleObs) {
  ofFront->prestageRoot();
  ofFront->setRankTarget();

  OMPBound predTop = nPred;
  vector<unsigned int> nExtinct(predTop);

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound predIdx = 0; predIdx < predTop; predIdx++) {
      nExtinct[predIdx] = ofFront->stage(predIdx, obsPart.get(), layout, sampleObs);
    }
  }
  return nExtinct;
}


vector<unsigned int> InterLevel::restage() {
  unsigned int backPop = prestageRear(); // Popable layers persist.
  ofFront->setRankTarget();

  OMPBound idxTop = ancestor.size();
  vector<unsigned int> nExtinct(idxTop);
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound idx = 0; idx < idxTop; idx++) {
      nExtinct[idx] = restage(ancestor[idx]);
    }
  }

  ancestor.clear();
  while (backPop--) { // Rear layers may now pop.
    history.pop_back();
  }

  return nExtinct;
}


unsigned int InterLevel::prestageRear() {
  // TODO:  replace constant.
  // 8-bit paths cannot represent beyond a 7-layer history.
  unsigned int backPop = 0;
  if (history.size() == 7) {//!NodePath::isRepresentable(history.size()))
    history.back()->prestageLayer(ofFront.get());
    backPop++;
  }

  for (int backLayer = history.size() - backPop - 1; backLayer >= 0; backLayer--) {
    if ((history[backLayer])->stageOccupancy() < stageEfficiency) {
      history[backLayer]->prestageLayer(ofFront.get());
      backPop++;
    }
    else {
      break;
    }
  }

  return backPop;
}


unsigned int InterLevel::restage(Ancestor& ancestor) {
  return history[ancestor.historyIdx]->restage(obsPart.get(), ancestor.cell, ofFront.get());
}


vector<IndexSet> InterLevel::overlap(const Frontier* frontier,
				     const SampleMap& smNext,
				     const vector<IndexSet>& frontierNodes) {
  vector<IndexSet> frontierNext = frontier->produce();
  splitCount = frontierNext.size();
  if (splitCount != 0) { // Otherwise no further splitting or repartitioning.
    // ofFront is assigned its front range by reviseStageMap().  This
    // front range is then applied to all layers on deque, following
    // which ofFront is itself placed on the deque.
    //
    reviseStageMap(frontierNodes);

    ofFront->setFrontRange(frontierNodes, frontierNext);
    for (auto lv = history.begin(); lv != history.end(); lv++) {
      (*lv)->applyFront(ofFront.get(), frontierNext);
    }
    history.push_front(move(ofFront));

    ofFront = make_unique<ObsFrontier>(frontier, splitCount, nPred, smNext.getEndIdx(), this);
  }
  level++;

  return frontierNext;
}


void InterLevel::reviseStageMap(const vector<IndexSet>& frontierNodes) {
  vector<vector<PredictorT>> stageMapNext(splitCount);
  IndexT terminalCount = 0;
  for (IndexT parIdx = 0; parIdx < frontierNodes.size(); parIdx++) {
    if (frontierNodes[parIdx].isTerminal()) {
      terminalCount++;
    }
    else {
      IndexT splitIdx = 2 * (parIdx - terminalCount);
      stageMapNext[splitIdx] = stageMap[parIdx];
      stageMapNext[splitIdx+1] = stageMap[parIdx];
    }
  }

  stageMap = move(stageMapNext);
}


void InterLevel::rootSuccessor(IndexT rootIdx,
			       PathT path,
			       IndexT smIdx) {
  rootPath->setSuccessor(rootIdx, path);
}


void InterLevel::rootExtinct(IndexT rootIdx) {
  rootPath->setExtinct(rootIdx);
}