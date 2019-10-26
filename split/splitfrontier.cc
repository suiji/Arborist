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
#include "cand.h"
#include "runset.h"
#include "samplenux.h"
#include "obspart.h"
#include "callback.h"
#include "summaryframe.h"
#include "rankedframe.h"
#include "sample.h"
#include "ompthread.h"

// Post-split consumption:
#include "pretree.h"

SplitFrontier::SplitFrontier(const Cand* cand_,
			     const SummaryFrame* frame_,
                             Frontier* frontier_,
                             const Sample* sample) :
  cand(cand_),
  frame(frame_),
  rankedFrame(frame->getRankedFrame()),
  frontier(frontier_),
  nPred(frame->getNPred()),
  obsPart(sample->predictors()) {
}


SplitFrontier::~SplitFrontier() {
}


RunSet *SplitFrontier::rSet(IndexT setIdx) const {
  return run->rSet(setIdx);
}


SampleRank* SplitFrontier::getPredBase(const SplitNux* cand) const {
  return obsPart->getPredBase(cand->getDefCoord());
}


IndexT SplitFrontier::getDenseRank(const SplitNux* cand) const {
  return rankedFrame->getDenseRank(cand->getPredIdx());
}


vector<DefCoord>
SplitFrontier::precandidates(const DefMap* defMap) {
  return cand->precandidates(this, defMap);
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


/**
   @brief Initializes frontier about to be split
 */
void SplitFrontier::init() {
  nSplit = frontier->getNSplit();
  prebias = vector<double>(nSplit);

  layerPreset(); // virtual
  setPrebias();
}


IndexT SplitFrontier::getNoSet() const {
  return frame->getNPredFac() * nSplit;
}


void SplitFrontier::setPrebias() {
  for (IndexT splitIdx = 0; splitIdx < nSplit; splitIdx++) {
    setPrebias(splitIdx, frontier->getSum(splitIdx), frontier->getSCount(splitIdx));
  }
}


/**
   @brief Base method.  Clears per-frontier vectors.
 */
void SplitFrontier::clear() {
  prebias.clear();
  run->clear();
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


void SplitFrontier::restageAndSplit(DefMap* defMap) {
  init();
  unsigned int flushCount = defMap->flushRear(this);
  vector<DefCoord> preCand = precandidates(defMap);

  defMap->backdate();
  restage(defMap);

  defMap->eraseLayers(flushCount);
  vector<SplitNux> postCand = postSchedule(defMap, preCand);
  split(postCand);
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
  setRunOffsets(runCount);

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


IndexT SplitFrontier::lHBits(PredictorT setIdx,
			     PredictorT lhBits,
			     IndexT& lhSCount) const {
  return rSet(setIdx)->lHBits(lhBits, lhSCount);
}


IndexT SplitFrontier::lHSlots(PredictorT setIdx,
			      PredictorT lhBits,
			      IndexT& lhSCount) const {
  return rSet(setIdx)->lHSlots(lhBits, lhSCount);
}


void
SplitFrontier::restage(const DefMap* defMap) {
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


vector<unique_ptr<SplitNux> > SplitFrontier::maxCandidates(const vector<SplitNux>& sc) {
  vector<unique_ptr<SplitNux> > nuxMax(nSplit); // Info initialized to zero.

  OMPBound splitTop = nSplit;
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitIdx = 0; splitIdx < splitTop; splitIdx++) {
      nuxMax[splitIdx] = maxSplit(sc, candOff[splitIdx], nCand[splitIdx]);
    }
  }

  return nuxMax;
}


unique_ptr<SplitNux>
SplitFrontier::maxSplit(const vector<SplitNux>& sc,
			IndexT splitBase,
			IndexT nCandSplit) const {
  IndexT argMax = splitBase + nCandSplit;
  double runningMax = 0.0;
  for (IndexT splitOff = splitBase; splitOff < splitBase + nCandSplit; splitOff++) {
    if (sc[splitOff].maxInfo(runningMax)) {
      argMax = splitOff;
    }
  }

  return runningMax > 0.0 ? make_unique<SplitNux>(sc[argMax]) : make_unique<SplitNux>();
}


SplitSurvey SplitFrontier::consume(PreTree* pretree, vector<IndexSet>& indexSet, Replay* replay) {
  SplitSurvey survey;
  for (auto & iSet : indexSet) {
    consume(pretree, iSet, replay, survey);
  }
  clear();

  return survey;
}


void SplitFrontier::consume(PreTree* pretree,
                            IndexSet& iSet,
                            Replay* replay,
                            SplitSurvey& survey) const {
  if (isInformative(&iSet)) {
    branch(pretree, &iSet, replay);
    survey.splitNext += frontier->splitCensus(iSet, survey);
  }
  else {
    survey.leafCount++;
  }
}


void SplitFrontier::branch(PreTree* pretree,
                           IndexSet* iSet,
                           Replay* replay) const {
  pretree->nonterminal(getInfo(iSet), iSet); // Once per node.

  // Once per criterion:
  consumeCriterion(iSet);
  if (getCardinality(iSet) > 0) {
    critRun(pretree, iSet, replay);
  }
  else {
    critCut(pretree, iSet, replay);
  }
}


void SplitFrontier::consumeCriterion(IndexSet* iSet) const {
  nuxMax[iSet->getSplitIdx()]->consume(iSet);
}


void SplitFrontier::critCut(PreTree* pretree,
                            IndexSet* iSet,
			    Replay* replay) const {
  pretree->critCut(iSet, getPredIdx(iSet), getQuantRank(iSet));
  vector<SumCount> ctgCrit(iSet->getNCtg());
  double sumExpl = blockReplay(iSet, getExplicitRange(iSet), leftIsExplicit(iSet), replay, ctgCrit);
  iSet->criterionLR(sumExpl, ctgCrit, leftIsExplicit(iSet));
}


double
SplitFrontier::blockReplay(class IndexSet* iSet,
			   const IndexRange& range,
			   bool leftExpl,
			   class Replay* replay,
			   vector<SumCount>& ctgCrit) const {
  return obsPart->blockReplay(this, iSet, range, leftExpl, replay, ctgCrit);
}



void SplitFrontier::critRun(PreTree* pretree,
			    IndexSet* iSet,
			    Replay* replay) const {
  pretree->critBits(iSet, getPredIdx(iSet), getCardinality(iSet));
  bool leftExpl;
  vector<SumCount> ctgCrit(iSet->getNCtg());
  double sumExpl = run->branch(this, iSet, pretree, replay, ctgCrit, leftExpl);
  iSet->criterionLR(sumExpl, ctgCrit, leftExpl);
}


bool SplitFrontier::isUnsplitable(IndexT splitIdx) const {
  return frontier->isUnsplitable(splitIdx);
}


IndexRange SplitFrontier::getBufRange(const DefCoord& preCand) const {
  return frontier->getBufRange(preCand);
}


double
SplitFrontier::getSum(const SplitCoord& splitCoord) const {
  return frontier->getSum(splitCoord.nodeIdx);
}


IndexT
SplitFrontier::getSCount(const SplitCoord& splitCoord) const {
  return frontier->getSCount(splitCoord.nodeIdx);
}


bool SplitFrontier::isInformative(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()]->getInfo() > iSet->getMinInfo();
}


IndexT SplitFrontier::getLHExtent(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()]->getExtent();
}


IndexT SplitFrontier::getPredIdx(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()]->getPredIdx();
}

unsigned int SplitFrontier::getBufIdx(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()]->getBufIdx();
}


DefCoord SplitFrontier::getDefCoord(const IndexSet* iSet) const {
  return DefCoord(SplitCoord(iSet->getSplitIdx(), nuxMax[iSet->getSplitIdx()]->getPredIdx()), nuxMax[iSet->getSplitIdx()]->getBufIdx());
}


PredictorT SplitFrontier::getCardinality(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()]->getCardinality(frame);
}


double SplitFrontier::getInfo(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()]->getInfo();
}


IndexRange SplitFrontier::getExplicitRange(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()]->getExplicitRange();
}


double SplitFrontier::getQuantRank(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()]->getQuantRank();
}


bool SplitFrontier::leftIsExplicit(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()]->leftIsExplicit();
}


IndexT SplitFrontier::getSetIdx(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()]->getSetIdx();
}
  
