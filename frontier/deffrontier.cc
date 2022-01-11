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

#include "frontier.h"
#include "splitfrontier.h"
#include "deffrontier.h"
#include "deflayer.h"
#include "partition.h"
#include "splitnux.h"
#include "sample.h"
#include "trainframe.h"
#include "layout.h"
#include "path.h"
#include "ompthread.h"
#include "indexset.h"

#include "algparam.h"

#include <numeric>
#include <algorithm>


DefFrontier::DefFrontier(const TrainFrame* frame,
			 Frontier* frontier_) :
  nPred(frame->getNPred()),
  frontier(frontier_),
  bagCount(frontier->getBagCount()),
  stPath(make_unique<IdxPath>(bagCount)),
  splitPrev(0),
  splitCount(1),
  layout(frame->getLayout()),
  nPredDense(layout->getNPredDense()),
  denseIdx(layout->getDenseIdx()),
  obsPart(make_unique<ObsPart>(layout, bagCount)),
  history(vector<unsigned int>(0)),
  layerDelta(vector<unsigned char>(nPred))
{
  layer.push_front(make_unique<DefLayer>(1, nPred, bagCount, bagCount, false, this));
  layer[0]->initAncestor(0, IndexRange(0, bagCount));
}


void DefFrontier::clearDefs(unsigned int flushCount) {
  ancestor.clear();
  if (flushCount > 0) {
    layer.erase(layer.end() - flushCount, layer.end());
  }
}


void DefFrontier::reachFlush(const SplitCoord& splitCoord) {
  DefLayer *reachingLayer = reachLayer(splitCoord);
  reachingLayer->flushDef(getHistory(reachingLayer, splitCoord), this);
}


bool DefFrontier::isSingleton(const MRRA& mrra) const {
  return layer[0]->isSingleton(mrra.splitCoord);
}


vector<SplitNux> DefFrontier::getCandidates(const SplitFrontier* sf) const {
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


const ObsPart* DefFrontier::getObsPart() const {
  return obsPart.get();
}


IndexT* DefFrontier::getBufferIndex(const SplitNux* nux) const {
  return obsPart->getBufferIndex(nux);
}


SampleRank* DefFrontier::getPredBase(const SplitNux* nux) const {
  return obsPart->getPredBase(nux);
}


IndexT DefFrontier::getImplicitCount(const MRRA& mrra) const {
  return layer[0]->getImplicit(mrra);
}


void DefFrontier::adjustRange(const MRRA& mrra,
			      IndexRange& idxRange) const {
  layer[0]->adjustRange(mrra, idxRange);
}


unsigned int DefFrontier::flushRear() {
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


void DefFrontier::stage(const Sample* sample) {
  IndexT predIdx = 0;
  vector<StageCount> stageCount = layout->stage(sample, obsPart.get());
  for (auto sc : stageCount) {
    layer[0]->rootDefine(predIdx, sc);
    setStageCount(SplitCoord(0, predIdx), sc); // All root cells must define.
    predIdx++;
  }

  setPrecandidates(0);
  for (auto & pc : preCand[0]) { // Root:  single split.
    pc.setStageCount(stageCount[pc.mrra.splitCoord.predIdx]);
  }
}


void DefFrontier::setPrecandidates(unsigned int level) {
  frontier->earlyExit(level);

  preCand = vector<vector<PreCand>>(splitCount);
  // Precandidates precipitate restaging ancestors at this level,
  // as do all non-singleton definitions arising from flushes.
  CandType::precandidates(this);
}


void DefFrontier::setStageCount(const SplitCoord& splitCoord,
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


void DefFrontier::setStageCount(const SplitCoord& splitCoord,
				const StageCount& stageCount) const {
  layer[0]->setStageCount(splitCoord, stageCount);
}


void DefFrontier::appendAncestor(const MRRA& cand) {
  ancestor.push_back(cand);
}


void DefFrontier::restage() {
  unsigned int flushCount = flushRear();
  backdate();

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


bool DefFrontier::preschedule(const SplitCoord& splitCoord, double dRand) {
  unsigned int bufIdx;
  if (preschedulable(splitCoord, bufIdx)) {
    preCand[splitCoord.nodeIdx].emplace_back(splitCoord, bufIdx, getRandLow(dRand));
    return true;
  }
  else {
    return false;
  }
}


bool DefFrontier::preschedulable(const SplitCoord& splitCoord, unsigned int& bufIdx) {
  reachFlush(splitCoord);
  return !layer[0]->isSingleton(splitCoord, bufIdx);
}


bool DefFrontier::isUnsplitable(IndexT splitIdx) const {
  return frontier->isUnsplitable(splitIdx);
}


void DefFrontier::restage(const MRRA& mrra) const {
  layer[mrra.del]->rankRestage(obsPart.get(), mrra, layer[0].get());
}


DefFrontier::~DefFrontier() {
  for (auto & defLayer : layer) {
    defLayer->flush();
  }
  layer.clear();
}


void DefFrontier::overlap(IndexT splitNext,
                IndexT bagCount,
                IndexT idxLive,
		bool nodeRel) {
  splitPrev = exchange(splitCount, splitNext);
  if (splitCount == 0) // No further splitting or restaging.
    return;

  layer.push_front(make_unique<DefLayer>(splitCount, nPred, bagCount, idxLive, nodeRel, this));

  historyPrev = move(history);
  history = vector<unsigned int>(splitCount * (layer.size()-1));

  deltaPrev = move(layerDelta);
  layerDelta = vector<unsigned char>(splitCount * nPred);

  for (auto lv = layer.begin() + 1; lv != layer.end(); lv++) {
    (*lv)->reachingPaths();
  }
}


void DefFrontier::backdate() const {
  if (layer.size() > 2 && layer[1]->isNodeRel()) {
    for (auto lv = layer.begin() + 2; lv != layer.end(); lv++) {
      if (!(*lv)->backdate(getFrontPath(1))) {
        break;
      }
    }
  }
}

  
void DefFrontier::reachingPath(const IndexSet& iSet,
			       IndexT parIdx,
			       IndexT relBase) {
  IndexT splitIdx = iSet.getSplitIdx();
  for (unsigned int backLayer = 0; backLayer < layer.size() - 1; backLayer++) {
    history[splitIdx + splitCount * backLayer] = backLayer == 0 ? parIdx : historyPrev[parIdx + splitPrev * (backLayer - 1)];
  }

  inherit(splitIdx, parIdx);
  IndexRange bufRange = iSet.getBufRange();
  layer[0]->initAncestor(splitIdx, bufRange);
  
  // Places <splitIdx, start> pair at appropriate position in every
  // reaching path.
  //
  unsigned int path = iSet.getPath();
  for (auto lv = layer.begin() + 1; lv != layer.end(); lv++) {
    (*lv)->pathInit(splitIdx, path, bufRange, relBase);
  }
}


void DefFrontier::setLive(IndexT ndx,
		IndexT targIdx,
		IndexT stx,
		unsigned int path,
		IndexT ndBase) {
  layer[0]->setLive(ndx, path, targIdx, ndBase);

  if (!layer.back()->isNodeRel()) {
    stPath->setLive(stx, path, targIdx);  // Irregular write.
  }
}


void DefFrontier::setExtinct(IndexT nodeIdx,
			     IndexT stIdx) {
  layer[0]->setExtinct(nodeIdx);
  setExtinct(stIdx);
}


void DefFrontier::setExtinct(IndexT stIdx) {
  if (!layer.back()->isNodeRel()) {
    stPath->setExtinct(stIdx);
  }
}


IndexT DefFrontier::getSplitCount(unsigned int del) const {
  return layer[del]->getSplitCount();
}


void DefFrontier::addDef(const MRRA& defCoord,
			 bool singleton) {
  if (layer[0]->define(defCoord, singleton)) {
    layerDelta[defCoord.splitCoord.strideOffset(nPred)] = 0;
  }
}
  

IndexT DefFrontier::getHistory(const DefLayer *reachLayer,
                   IndexT splitIdx) const {
  return reachLayer == layer[0].get() ? splitIdx : history[splitIdx + (reachLayer->getDel() - 1) * splitCount];
}


SplitCoord DefFrontier::getHistory(const DefLayer* reachLayer,
		   const SplitCoord& coord) const {
  return reachLayer == layer[0].get() ? coord :
    SplitCoord(history[coord.nodeIdx + splitCount * (reachLayer->getDel() - 1)], coord.predIdx);
}


const IdxPath* DefFrontier::getFrontPath(unsigned int del) const {
  return layer[del]->getFrontPath();
}
