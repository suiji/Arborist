// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file deffrontier.cc

   @brief Maintains refrerences to repartioned cells reaching frontier.

   @author Mark Seligman
 */

#include "ompthread.h"
#include "frontier.h"
#include "splitfrontier.h"
#include "defmap.h"
#include "deffrontier.h"
#include "partition.h"
#include "splitnux.h"
#include "sampleobs.h"
#include "trainframe.h"
#include "layout.h"
#include "path.h"
#include "indexset.h"

#include "algparam.h"

#include <numeric>
#include <algorithm>


DefMap::DefMap(const TrainFrame* frame,
			 Frontier* frontier_) :
  nPred(frame->getNPred()),
  frontier(frontier_),
  bagCount(frontier->getBagCount()),
  rootPath(make_unique<IdxPath>(bagCount)),
  splitPrev(0),
  splitCount(1),
  layout(frame->getLayout()),
  nPredDense(layout->getNPredDense()),
  denseIdx(layout->getDenseIdx()),
  obsPart(make_unique<ObsPart>(layout, bagCount)),
  history(vector<unsigned int>(0)),
  layerDelta(vector<unsigned char>(nPred))
{
  layer.push_front(make_unique<DefFrontier>(1, nPred, bagCount, bagCount, this));
  layer[0]->initAncestor(0, IndexRange(0, bagCount));
}


void DefMap::clearDefs(unsigned int flushCount) {
  ancestor.clear();
  if (flushCount > 0) {
    layer.erase(layer.end() - flushCount, layer.end());
  }
}


bool DefMap::isSingleton(const MRRA& mrra) const {
  return layer[0]->isSingleton(mrra.splitCoord);
}


vector<SplitNux> DefMap::getCandidates(const SplitFrontier* sf) const {
  vector<SplitNux> postCand;
  for (auto pcVec : preCand) {
    for (auto pc : pcVec) {
      if (!pc.isSingleton()) {
	postCand.emplace_back(pc, sf);
      }
    }
  }

  return postCand;
}


const ObsPart* DefMap::getObsPart() const {
  return obsPart.get();
}


IndexT* DefMap::getIdxBuffer(const SplitNux* nux) const {
  return obsPart->getIdxBuffer(nux);
}


ObsCell* DefMap::getPredBase(const SplitNux* nux) const {
  return obsPart->getPredBase(nux);
}


IndexT DefMap::getImplicitCount(const MRRA& mrra) const {
  return layer[0]->getImplicit(mrra);
}


void DefMap::adjustRange(const MRRA& mrra,
			      IndexRange& idxRange) const {
  layer[0]->adjustRange(mrra, idxRange);
}


unsigned int DefMap::flushRear() {
  unsigned int unflushTop = layer.size() - 1;

  // Capacity:  1 front layer + 'pathMax' back layers.
  // If at capacity, every reaching definition should be flushed
  // to current layer ut avoid falling off the deque.
  // Flushing prior to split assignment, rather than during, should
  // also save lookup time, as all definitions reaching from rear are
  // now at current layer.
  //
  if (!NodePath::isRepresentable(layer.size())) {
    layer.back()->flush(this);
    unflushTop--;
  }

  // Walks backward from rear, purging non-reaching definitions.
  // Stops when a layer with no non-reaching nodes is encountered.
  //
  for (unsigned int off = unflushTop; off > 0; off--) {
    if (!layer[off]->nonreachPurge())
      break;
  }

  IndexT backDef = 0;
  for (auto lv = layer.begin() + unflushTop; lv != layer.begin(); lv--) {
    backDef += (*lv)->getDefCount();
  }

  IndexT thresh = backDef * efficiency;
  for (auto lv = layer.begin() + unflushTop; lv != layer.begin(); lv--) {
    if ((*lv)->flush(this, thresh)) {
      unflushTop--;
    }
    else {
      break;
    }
  }

  // assert(unflushTop < layer.size();
  return layer.size() - 1 - unflushTop;
}


void DefMap::setPrecandidates(const SampleObs* sample, unsigned int level) {
  preCand = vector<vector<PreCand>>(splitCount);
  // Precandidates precipitate restaging ancestors at this level,
  // as do all non-singleton definitions arising from flushes.
  CandType::precandidates(this);
  if (level == 0)
    stage(sample);
}


void DefMap::stage(const SampleObs* sample) {
  IndexT predIdx = 0;
  vector<StageCount> stageCount = layout->stage(sample, obsPart.get());
  for (auto sc : stageCount) {
    layer[0]->rootDefine(predIdx, sc);
    setStageCount(SplitCoord(0, predIdx), sc); // All root cells must define.
    predIdx++;
  }

  for (auto & pc : preCand[0]) { // Root:  single split.
    pc.setStageCount(stageCount[pc.mrra.splitCoord.predIdx]);
  }
}


void DefMap::setStageCount(const SplitCoord& splitCoord,
				IndexT idxImplicit,
				IndexT rankCount) {
  vector<PreCand>& pcSplit = preCand[splitCoord.nodeIdx];
  StageCount sc(idxImplicit, rankCount);
  setStageCount(splitCoord, sc); // def cell must be refreshed.
  for (auto & pc : pcSplit) { // Replace with binary search.
    if (pc.mrra.splitCoord.predIdx == splitCoord.predIdx) {
      pc.setStageCount(sc); // Sets precandidate, if any.
      return;
    }
  }
}


void DefMap::setStageCount(const SplitCoord& splitCoord,
				const StageCount& stageCount) const {
  layer[0]->setStageCount(splitCoord, stageCount);
}


void DefMap::appendAncestor(const MRRA& cand) {
  ancestor.push_back(cand);
}


void DefMap::restage() {
  unsigned int flushCount = flushRear();

  OMPBound idxTop = ancestor.size();
  
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound nodeIdx = 0; nodeIdx < idxTop; nodeIdx++) {
      restage(ancestor[nodeIdx]);
    }
  }

  clearDefs(flushCount); // Definitions currently must persist to this point.
}


void DefMap::restage(const MRRA& mrra) const {
  layer[mrra.del]->rankRestage(obsPart.get(), mrra, layer[0].get());
}


bool DefMap::preschedule(const SplitCoord& splitCoord, double dRand) {
  unsigned int bufIdx;
  reachFlush(splitCoord);
  if (preschedulable(splitCoord, bufIdx)) {
    preCand[splitCoord.nodeIdx].emplace_back(splitCoord, bufIdx, getRandLow(dRand));
    return true;
  }
  else {
    return false;
  }
}


bool DefMap::preschedulable(const SplitCoord& splitCoord, unsigned int& bufIdx) const {
  return !layer[0]->isSingleton(splitCoord, bufIdx);
}


void DefMap::reachFlush(const SplitCoord& splitCoord) {
  DefFrontier *reachingLayer = reachLayer(splitCoord);
  reachingLayer->flushDef(getHistory(reachingLayer, splitCoord), this);
}


bool DefMap::isUnsplitable(IndexT splitIdx) const {
  return frontier->isUnsplitable(splitIdx);
}


DefMap::~DefMap() {
  for (auto & defFrontier : layer) {
    defFrontier->flush();
  }
  layer.clear();
}


void DefMap::reachingPath(const IndexSet& iSet,
			  const IndexSet& par) {
  IndexT splitIdx = iSet.getSplitIdx();
  IndexT parIdx = par.getSplitIdx();
  for (unsigned int backLayer = 0; backLayer < layer.size() - 1; backLayer++) {
    history[splitIdx + splitCount * backLayer] = backLayer == 0 ? parIdx : historyPrev[parIdx + splitPrev * (backLayer - 1)];
  }

  inherit(splitIdx, parIdx);
  IndexRange bufRange = iSet.getBufRange();
  layer[0]->initAncestor(splitIdx, bufRange);
  
  // Places <splitIdx, start> pair at appropriate position in every
  // reaching path.
  //
  IndexT idxStart = frontier->idxStartUpcoming(iSet);
  PathT path = iSet.getPath();
  for (auto lv = layer.begin() + 1; lv != layer.end(); lv++) {
    (*lv)->pathInit(splitIdx, path, bufRange, idxStart);
  }
}


void DefMap::nextLevel(const BranchSense* branchSense,
		       const SampleMap& smNonterm,
		       SampleMap& smTerminal,
		       SampleMap& smNext) {
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitIdx = 0; splitIdx < frontier->getNSplit(); splitIdx++) {
      frontier->setScore(splitIdx);
      layer[0]->updateMap(frontier->getNode(splitIdx), branchSense, smNonterm, smTerminal, smNext);
    }
  }

  overlap(smNext);
}


void DefMap::rootSuccessor(IndexT rootIdx,
			   PathT path,
			   IndexT smIdx) {
  rootPath->setSuccessor(rootIdx, path);
}


void DefMap::rootExtinct(IndexT rootIdx) {
  rootPath->setExtinct(rootIdx);
}


void DefMap::overlap(const SampleMap& smNext) {
  splitPrev = exchange(splitCount, smNext.getNodeCount());
  if (splitCount == 0) // No further splitting or repartitioning.
    return;

  IndexT idxLive = smNext.getEndIdx();
  layer.push_front(make_unique<DefFrontier>(splitCount, nPred, bagCount, idxLive, this));

  historyPrev = move(history);
  history = vector<unsigned int>(splitCount * (layer.size()-1));

  deltaPrev = move(layerDelta);
  layerDelta = vector<unsigned char>(splitCount * nPred);

  for (auto lv = layer.begin() + 1; lv != layer.end(); lv++) {
    (*lv)->reachingPaths();
  }
}


IndexT DefMap::getSplitCount(unsigned int del) const {
  return layer[del]->getSplitCount();
}


void DefMap::addDef(const MRRA& defCoord,
		    bool singleton) {
  if (layer[0]->define(defCoord, singleton)) {
    layerDelta[defCoord.splitCoord.strideOffset(nPred)] = 0;
  }
}
  

IndexT DefMap::getHistory(const DefFrontier *reachLayer,
                   IndexT splitIdx) const {
  return reachLayer == layer[0].get() ? splitIdx : history[splitIdx + (reachLayer->getDel() - 1) * splitCount];
}


SplitCoord DefMap::getHistory(const DefFrontier* reachLayer,
		   const SplitCoord& coord) const {
  return reachLayer == layer[0].get() ? coord :
    SplitCoord(history[coord.nodeIdx + splitCount * (reachLayer->getDel() - 1)], coord.predIdx);
}
