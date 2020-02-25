// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitfrontier.cc

   @brief Methods to implement splitting of frontier.

   @author Mark Seligman
 */


#include "frontier.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "defmap.h"
#include "algparam.h"
#include "runset.h"
#include "samplenux.h"
#include "obspart.h"
#include "summaryframe.h"
#include "rankedframe.h"
#include "sample.h"
#include "ompthread.h"

// Post-split consumption:
#include "pretree.h"

SplitFrontier::SplitFrontier(const SummaryFrame* frame_,
                             Frontier* frontier_,
                             const Sample* sample) :
  frame(frame_),
  rankedFrame(frame->getRankedFrame()),
  frontier(frontier_),
  nPred(frame->getNPred()),
  obsPart(sample->predictors()),
  run(make_unique<Run>(frontier->getNCtg(), frame->getNRow())) {
}


SplitFrontier::~SplitFrontier() {
}


IndexT* SplitFrontier::getBufferIndex(const SplitNux* nux) const {
  return obsPart->getBufferIndex(nux);
}


RunSet* SplitFrontier::getRunSet(PredictorT setIdx) const {
  return run->rSet(setIdx);
}


IndexRange SplitFrontier::getRunBounds(const SplitNux* nux, PredictorT slot) const {
  return run->getBounds(nux, slot);
}


SampleRank* SplitFrontier::getPredBase(const SplitNux* nux) const {
  return obsPart->getPredBase(nux->getDefCoord());
}


IndexT SplitFrontier::getDenseRank(const SplitNux* nux) const {
  return rankedFrame->getDenseRank(nux->getPredIdx());
}


vector<DefCoord>
SplitFrontier::precandidates(const DefMap* defMap) {
  return CandType::precandidates(this, defMap);
}


void
SplitFrontier::preschedule(const DefCoord& defCoord,
			  vector<DefCoord>& preCand) const {
  preCand.emplace_back(defCoord);
}


void SplitFrontier::setCandOff(const vector<PredictorT>& nCand) {
  candOff = vector<IndexT>(nCand.size());
  IndexT tot = 0;
  IndexT i = 0;
  for (auto nc : nCand) {
    candOff[i++] = tot;
    tot += nc;
  }
  this->nCand = move(nCand);
}


IndexT SplitFrontier::getNoSet() const {
  return frame->getNPredFac() * nSplit;
}


void SplitFrontier::setPrebias() {
  for (IndexT splitIdx = 0; splitIdx < nSplit; splitIdx++) {
    setPrebias(splitIdx, frontier->getSum(splitIdx), frontier->getSCount(splitIdx));
  }
}


bool SplitFrontier::isFactor(const SplitCoord& splitCoord) const {
  return frame->isFactor(splitCoord.predIdx);
}


PredictorT SplitFrontier::getNumIdx(PredictorT predIdx) const {
  return frame->getNumIdx(predIdx);
}


vector<StageCount> SplitFrontier::stage(const Sample* sample) {
  return sample->stage(obsPart.get());
}


void SplitFrontier::nuxReplay(SplitNux* nux,
			      BranchSense* branchSense,
			      vector<SumCount>& ctgCrit,
			      bool exclusive) const {
  if (nux->getCardinality(frame) > 0) {
    runReplay(nux, branchSense, ctgCrit, exclusive);
  }
  else {
    exclusive ? rangeReplayExcl(nux, branchSense, ctgCrit) : rangeReplay(nux, nux->getEncodedRange(), branchSense, ctgCrit);
  }
}


void SplitFrontier::runReplay(SplitNux* nux,
			      BranchSense* branchSense,
			      vector<SumCount>& ctgCrit,
			      bool exclusive) const {
  PredictorT slotStart, slotEnd;
  if (nux->trueEncoding()) {
    slotStart = 0;
    slotEnd = run->getRunsLH(nux);
  }
  else { // Replay indices explicit on false branch.
    slotStart = run->getRunsLH(nux);
    slotEnd = run->getRunCount(nux);
  }
  for (PredictorT outSlot = slotStart; outSlot < slotEnd; outSlot++) {
    exclusive ? rangeReplayExcl(nux, branchSense, ctgCrit) : rangeReplay(nux, getRunBounds(nux, outSlot), branchSense, ctgCrit);
  }
}


void SplitFrontier::rangeReplay(SplitNux* nux,
				const IndexRange& range,
				BranchSense* branchSense,
				vector<SumCount>& ctgCrit) const {
  IndexT* sIdx;
  SampleRank* spn = obsPart->buffers(nux->getDefCoord(), sIdx);
  for (IndexT opIdx = range.getStart(); opIdx < range.getEnd(); opIdx++) {
    spn[opIdx].accum(nux, ctgCrit);
    branchSense->set(sIdx[opIdx], nux->trueEncoding());
  }
  nux->encExtent(range);
}


void SplitFrontier::rangeReplayExcl(SplitNux* nux,
				    BranchSense* branchSense,
				    vector<SumCount>& ctgCrit) const {
  IndexT* sIdx;
  SampleRank* spn = obsPart->buffers(nux->getDefCoord(), sIdx);
  IndexRange range = nux->getCutRange();
  for (IndexT opIdx = range.getStart(); opIdx < range.getEnd(); opIdx++) {
    if (branchSense->setExclusive(sIdx[opIdx], nux->trueEncoding())) {
      nux->bumpExtent();
      spn[opIdx].accum(nux, ctgCrit);
    }
  }
}


void SplitFrontier::restageAndSplit(vector<IndexSet>& indexSet, DefMap* defMap, BranchSense* branchSense, PreTree* pretree) {
  init(branchSense);
  unsigned int flushCount = defMap->flushRear(this);
  vector<DefCoord> preCand = precandidates(defMap);

  defMap->backdate();
  restage(defMap);
  defMap->eraseLayers(flushCount);
  vector<SplitNux> postCand = postSchedule(defMap, preCand);

  split(indexSet, postCand, branchSense);
  consumeFrontier(pretree);
}


/**
   @brief Initializes frontier about to be split
 */
void SplitFrontier::init(BranchSense* branchSense) {
  branchSense->frontierReset();
  nSplit = frontier->getNSplit();
  prebias = vector<double>(nSplit);

  layerPreset(); // virtual
  setPrebias();
}


vector<SplitNux>
SplitFrontier::postSchedule(class DefMap* defMap, vector<DefCoord>& preCand) {
  vector<PredictorT> runCount;
  vector<SplitNux> postCand;
  vector<PredictorT> nCand(nSplit);
  fill(nCand.begin(), nCand.end(), 0);
  for (auto & pc : preCand) {
    postSchedule(defMap, pc, runCount, nCand, postCand);
  }

  setCandOff(nCand);
  run->setOffsets(runCount, frontier->getNCtg());

  return postCand;
}

void
SplitFrontier::postSchedule(const DefMap* defMap,
			    const DefCoord& preCand,
			    vector<PredictorT>& runCount,
			    vector<PredictorT>& nCand,
			    vector<SplitNux>& postCand) const {
  if (!defMap->isSingleton(preCand)) {
    PredictorT setIdx = getSetIdx(defMap->getRunCount(preCand), runCount);
    postCand.emplace_back(preCand, this, setIdx, defMap->adjustRange(preCand, this), defMap->getImplicitCount(preCand));
    nCand[preCand.splitCoord.nodeIdx]++;
  }
}


PredictorT
SplitFrontier::getSetIdx(PredictorT rCount,
			 vector<PredictorT>& runCount) const {
  PredictorT setIdx;
  if (rCount > 1) {
    setIdx = runCount.size();
    runCount.push_back(rCount);
  }
  else {
    setIdx = getNoSet();
  }
  return setIdx;
}


void SplitFrontier::lHBits(SplitNux* nux,
			   PredictorT lhBits) const {
  run->lHBits(nux, lhBits);
}


void SplitFrontier::lHSlots(SplitNux* nux,
			    PredictorT cutSlot) const {
  run->lHSlots(nux, cutSlot);
}


void SplitFrontier::appendSlot(SplitNux* nux) const {
  run->appendSlot(nux);
}


void SplitFrontier::restage(const DefMap* defMap) {
  OMPBound idxTop = restageCoord.size();
  
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound nodeIdx = 0; nodeIdx < idxTop; nodeIdx++) {
      defMap->restage(obsPart.get(), restageCoord[nodeIdx]);
    }
  }

  restageCoord.clear();
}


void SplitFrontier::consumeFrontier(PreTree* pretree) {
  consumeNodes(pretree);
  clear();
}


void SplitFrontier::consumeCriterion(PreTree* pretree,
				     const SplitNux* nux) const {
  if (nux->getCardinality(frame) > 0) {
    pretree->critBits(nux, nux->getCardinality(frame), run->getLHBits(nux));
  }
  else {
    pretree->critCut(nux);
  }
}


void SplitFrontier::clear() {
  prebias.clear();
  run->clear();
}


bool SplitFrontier::isUnsplitable(IndexT splitIdx) const {
  return frontier->isUnsplitable(splitIdx);
}


IndexRange SplitFrontier::getBufRange(const DefCoord& preCand) const {
  return frontier->getBufRange(preCand);
}


double
SplitFrontier::getSum(const SplitCoord& splitCoord) const {
  return frontier->getSum(splitCoord);
}


IndexT
SplitFrontier::getSCount(const SplitCoord& splitCoord) const {
  return frontier->getSCount(splitCoord);
}


IndexT
SplitFrontier::getPTId(const SplitCoord& splitCoord) const {
  return frontier->getPTId(splitCoord);
}

