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
#include "sampledobs.h"
#include "splitfrontier.h"
#include "interlevel.h"
#include "obsfrontier.h"
#include "splitnux.h"
#include "predictorframe.h"
#include "indexset.h"

#include <algorithm>


InterLevel::InterLevel(const PredictorFrame* frame_,
		       const SampledObs* sampledObs_,
		       const Frontier* frontier) :
  frame(frame_),
  nPred(frame->getNPred()),
  positionMask(getPositionMask(nPred)),
  levelShift(getLevelShift(nPred)),
  bagCount(frontier->getBagCount()),
  noRank(frame->getNoRank()),
  sampledObs(sampledObs_),
  rootPath(make_unique<IdxPath>(bagCount)),
  pathIdx(vector<PathT>(frame->getSafeSize(bagCount))),
  level(0),
  splitCount(1),
  obsPart(make_unique<ObsPart>(frame, bagCount)),
  stageMap(vector<vector<PredictorT>>(1)) {
  stageMap[0] = vector<PredictorT>(nPred);
}


bool InterLevel::isFactor(PredictorT predIdx) const {
  return frame->isFactor(predIdx);
}


bool InterLevel::isStaged(const SplitCoord& coord, StagedCell*& cell) const {
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

PathT* InterLevel::getPathBlock(PredictorT predIdx) {
  return &pathIdx[obsPart->getStageOffset(predIdx)];
}


IndexT* InterLevel::getIdxBuffer(const SplitNux& nux) const {
  return obsPart->getIdxBuffer(nux);
}


Obs* InterLevel::getPredBase(const SplitNux& nux) const {
  return obsPart->getPredBase(nux);
}


ObsFrontier* InterLevel::getFront() {
  return ofFront.get();
}


CandType InterLevel::repartition(const Frontier* frontier) {
  ofFront = make_unique<ObsFrontier>(frontier, this);
  CandType cand(this);
  cand.precandidates(frontier, this);
  // Precandidates precipitate restaging ancestors at this level,
  // as do all history flushes.
  vector<unsigned int> nExtinct;
  if (level == 0) {
    nExtinct = stage();
  }
  else {
    nExtinct = restage();
  }
  ofFront->prune(nExtinct);
  return cand;
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


vector<unsigned int> InterLevel::stage() {
  ofFront->prestageRoot(frame, sampledObs);

  OMPBound predTop = nPred;
  vector<unsigned int> nExtinct(predTop);

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound predIdx = 0; predIdx < predTop; predIdx++) {
      nExtinct[predIdx] = ofFront->stage(predIdx, obsPart.get(), frame, sampledObs);
    }
  }
  return nExtinct;
}


vector<unsigned int> InterLevel::restage() {
  unsigned int backPop = prestageRear(); // Popable layers persist.
  ofFront->runValues();

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


void InterLevel::overlap(const vector<IndexSet>& frontierNodes,
			 const vector<IndexSet>& frontierNext,
			 IndexT endIdx) {
  splitCount = frontierNext.size();
  if (splitCount != 0) { // Otherwise no further splitting or repartitioning.
    reviseStageMap(frontierNodes);

    // ofFront is assigned its front range by reviseStageMap().  This
    // front range is then applied to all layers on deque, following
    // which ofFront is itself placed on the deque.
    //
    ofFront->setFrontRange(frontierNodes, frontierNext);
    for (auto lv = history.begin(); lv != history.end(); lv++) {
      (*lv)->applyFront(ofFront.get(), frontierNext);
    }
    history.push_front(std::move(ofFront));
  }
  level++;
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

  stageMap = std::move(stageMapNext);
}


void InterLevel::rootSuccessor(IndexT rootIdx,
			       PathT path,
			       IndexT smIdx) {
  rootPath->setSuccessor(rootIdx, path);
}


void InterLevel::rootExtinct(IndexT rootIdx) {
  rootPath->setExtinct(rootIdx);
}


double InterLevel::interpolateRank(const SplitNux& cand,
				   IndexT obsLeft,
				   IndexT obsRight) const {
  IndexT sIdx = obsPart->getSampleIndex(cand, obsLeft); 
  IndexT rankLeft = sampledObs->getRank(cand.getPredIdx(), sIdx);
  sIdx = obsPart->getSampleIndex(cand, obsRight);
  IndexT rankRight = sampledObs->getRank(cand.getPredIdx(), sIdx);
  IndexRange rankRange(rankLeft, rankRight - rankLeft);

  return rankRange.interpolate(cand.getSplitQuant());
}


double InterLevel::interpolateRank(const SplitNux& cand,
				   IndexT obsIdx,
				   bool residualLeft) const {
  IndexT residualRank = frame->getImplicitRank(cand.getPredIdx());
  IndexT sIdx = obsPart->getSampleIndex(cand, obsIdx);
  IndexT rank = sampledObs->getRank(cand.getPredIdx(), sIdx);
  IndexT rankLeft = residualLeft ? residualRank : rank;
  IndexT rankRight = residualLeft ? rank : residualRank;
  IndexRange rankRange(rankLeft, rankRight - rankLeft);

  return rankRange.interpolate(cand.getSplitQuant());
}


IndexT InterLevel::getCode(const SplitNux& cand,
			   IndexT obsIdx,
			   bool isImplicit) const {
  if (isImplicit) {
    return frame->getImplicitRank(cand.getPredIdx());
  }
  IndexT sIdx = obsPart->getSampleIndex(cand, obsIdx);
  return sampledObs->getRank(cand.getPredIdx(), sIdx);
}
