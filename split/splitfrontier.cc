// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitfrontier.cc

   @brief Methods to implement splitting of index-tree levels.

   @author Mark Seligman
 */


#include "frontier.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "level.h"
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

SplitFrontier::SplitFrontier(const SummaryFrame* frame_,
                             Frontier* frontier_,
                             const Sample* sample) :
  frame(frame_),
  rankedFrame(frame->getRankedFrame()),
  frontier(frontier_),
  noSet(sample->getBagCount() * frame->getNPredFac()),
  obsPart(sample->predictors()) {
}


SplitFrontier::~SplitFrontier() {
}


RunSet *SplitFrontier::rSet(IndexT setIdx) const {
  return run->rSet(setIdx);
}


SampleRank* SplitFrontier::getPredBase(const SplitNux* cand) const {
  return obsPart->getPredBase(cand);
}


IndexT SplitFrontier::getDenseRank(const SplitNux* cand) const {
  return rankedFrame->getDenseRank(cand->getPredIdx());
}


IndexT SplitFrontier::preschedule(const SplitCoord& splitCoord,
                                  unsigned int bufIdx) {
  splitCand.emplace_back(SplitNux(this, frontier, splitCoord, bufIdx, noSet));
  return frontier->getExtent(splitCoord.nodeIdx);
}


/**
   @brief Walks the list of split candidates and invalidates those which
   restaging has marked unsplitable as well as singletons persisting since
   initialization or as a result of bagging.  Fills in run counts, which
   values restaging has established precisely.
*/
void SplitFrontier::scheduleSplits(const Level* levelFront) {
  vector<unsigned int> runCount;
  vector<SplitNux> sc2;
  IndexT splitPrev = splitCount;
  for (auto & sg : splitCand) {
    if (sg.schedule(levelFront, frontier, runCount)) {
      IndexT splitThis = sg.getSplitCoord().nodeIdx;
      nCand[splitThis]++;
      if (splitPrev != splitThis) {
        candOff[splitThis] = sc2.size();
        splitPrev = splitThis;
      }
      sc2.push_back(sg);
    }
  }
  splitCand = move(sc2);

  setRunOffsets(runCount);
  splitCandidates();
}


/**
   @brief Initializes level about to be split
 */
void SplitFrontier::init() {
  splitCount = frontier->getNSplit();
  prebias = vector<double>(splitCount);
  nCand = vector<IndexT>(splitCount);
  fill(nCand.begin(), nCand.end(), 0);
  candOff = vector<IndexT>(splitCount);
  fill(candOff.begin(), candOff.end(), splitCount); // inattainable.

  levelPreset(); // virtual
  setPrebias();
}


void SplitFrontier::setPrebias() {
  for (IndexT splitIdx = 0; splitIdx < splitCount; splitIdx++) {
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


void SplitFrontier::restage(Level* levelFrom,
                            Level* levelTo,
                            const SplitCoord& mrra,
                            unsigned int bufIdx) const {
  obsPart->restage(levelFrom, levelTo, mrra, bufIdx);
}


void SplitFrontier::splitCandidates() {
  OMPBound splitTop = splitCand.size();
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitPos = 0; splitPos < splitTop; splitPos++) {
      split(&splitCand[splitPos]);
    }
  }

  nuxMax = maxCandidates();
}


vector<SplitNux> SplitFrontier::maxCandidates() {
  vector<SplitNux> nuxMax(splitCount); // Info initialized to zero.

  OMPBound splitTop = splitCount;
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitIdx = 0; splitIdx < splitTop; splitIdx++) {
      nuxMax[splitIdx] = maxSplit(candOff[splitIdx], nCand[splitIdx]);
    }
  }
  splitCand.clear();
  candOff.clear();
  nCand.clear();

  return nuxMax;
}


SplitNux SplitFrontier::maxSplit(IndexT splitBase,
                                 IndexT nCandSplit) const {
  IndexT argMax = splitBase + nCandSplit;
  double runningMax = 0.0;
  for (IndexT splitOff = splitBase; splitOff < splitBase + nCandSplit; splitOff++) {
    if (splitCand[splitOff].maxInfo(runningMax)) {
      argMax = splitOff;
    }
  }

  return runningMax > 0.0 ? SplitNux(splitCand[argMax]) : SplitNux();
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
  nuxMax[iSet->getSplitIdx()].consume(iSet);
}


void SplitFrontier::critCut(PreTree* pretree,
                            IndexSet* iSet,
			    Replay* replay) const {
  pretree->critCut(iSet, getPredIdx(iSet), getRankRange(iSet));
  vector<SumCount> ctgCrit(iSet->getNCtg());
  double sumExpl = blockReplay(iSet, getExplicitRange(iSet), leftIsExplicit(iSet), replay, ctgCrit);
  iSet->criterionLR(sumExpl, ctgCrit, leftIsExplicit(iSet));
}


double SplitFrontier::blockReplay(IndexSet* iSet,
                                  const IndexRange& range,
                                  bool leftExpl,
                                  Replay* replay,
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


bool SplitFrontier::isInformative(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getInfo() > iSet->getMinInfo();
}


IndexT SplitFrontier::getLHExtent(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getExtent();
}


IndexT SplitFrontier::getPredIdx(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getPredIdx();
}

unsigned int SplitFrontier::getBufIdx(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getBufIdx();
}


PredictorT SplitFrontier::getCardinality(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getCardinality(frame);
}


double SplitFrontier::getInfo(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getInfo();
}


IndexRange SplitFrontier::getExplicitRange(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getExplicitRange();
}


IndexRange SplitFrontier::getRankRange(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getRankRange();
}


bool SplitFrontier::leftIsExplicit(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].leftIsExplicit();
}


IndexT SplitFrontier::getSetIdx(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getSetIdx();
}
  
