// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitnode.cc

   @brief Methods to implement splitting of index-tree levels.

   @author Mark Seligman
 */


#include "frontier.h"
#include "splitfrontier.h"
#include "splitcand.h"
#include "splitnux.h"
#include "level.h"
#include "runset.h"
#include "samplenux.h"
#include "obspart.h"
#include "callback.h"
#include "summaryframe.h"
#include "rankedframe.h"
#include "sample.h"
#include "bv.h"
#include "ompthread.h"

// Post-split consumption:
#include "pretree.h"

vector<double> SFReg::mono; // Numeric monotonicity constraints.


SplitFrontier::SplitFrontier(const SummaryFrame* frame_,
                             Frontier* frontier_,
                             const Sample* sample) :
  frame(frame_),
  rankedFrame(frame->getRankedFrame()),
  frontier(frontier_),
  bagCount(sample->getBagCount()),
  noSet(bagCount * frame->getNPredFac()),
  obsPart(sample->predictors()) {
}


SplitFrontier::~SplitFrontier() {
}


void SFReg::Immutables(const SummaryFrame* frame,
                       const vector<double> &bridgeMono) {
  auto numFirst = frame->getNumFirst();
  auto numExtent = frame->getNPredNum();
  auto monoCount = count_if(bridgeMono.begin() + numFirst, bridgeMono.begin() + numExtent, [] (double prob) { return prob != 0.0; });
  if (monoCount > 0) {
    mono = vector<double>(frame->getNPredNum());
    mono.assign(bridgeMono.begin() + frame->getNumFirst(), bridgeMono.begin() + frame->getNumFirst() + frame->getNPredNum());
  }
}


void SFReg::DeImmutables() {
  mono.clear();
}


SFReg::SFReg(const SummaryFrame* frame,
             Frontier* frontier,
	     const Sample* sample) :
  SplitFrontier(frame, frontier, sample),
  ruMono(vector<double>(0)) {
  run = make_unique<Run>(0, frame->getNRow());
}


/**
   @brief Constructor.
 */
SFCtg::SFCtg(const SummaryFrame* frame,
             Frontier* frontier,
	     const Sample* sample,
	     PredictorT nCtg_):
  SplitFrontier(frame, frontier, sample),
  nCtg(nCtg_) {
  run = make_unique<Run>(nCtg, frame->getNRow());
}


RunSet *SplitFrontier::rSet(IndexT setIdx) const {
  return run->rSet(setIdx);
}


IndexT SplitFrontier::getDenseRank(const SplitCand* cand) const {
  return rankedFrame->getDenseRank(cand->getSplitCoord().predIdx);
}


/**
   @brief Sets quick lookup offets for Run object.

   @return void.
 */
void SFReg::setRunOffsets(const vector<unsigned int> &runCount) {
  run->offsetsReg(runCount);
}


/**
   @brief Sets quick lookup offsets for Run object.
 */
void SFCtg::setRunOffsets(const vector<unsigned int> &runCount) {
  run->offsetsCtg(runCount);
}


IndexT SplitFrontier::preschedule(const SplitCoord& splitCoord,
                                  unsigned int bufIdx) {
  splitCand.emplace_back(SplitCand(this, frontier, splitCoord, bufIdx, noSet));
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
  vector<SplitCand> sc2;
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
  split();
}


/**
   @brief Initializes level about to be split
 */
void SplitFrontier::init() {
  splitCount = frontier->getNSplit();
  prebias = vector<double>(splitCount);
  nCand = vector<unsigned int>(splitCount);
  fill(nCand.begin(), nCand.end(), 0);
  candOff = vector<unsigned int>(splitCount);
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


double SFCtg::getSumSquares(const SplitCand *cand) const {
  return sumSquares[cand->getSplitCoord().nodeIdx];
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


const vector<double>& SFCtg::getSumSlice(const SplitCand* cand) {
  return ctgSum[cand->getSplitCoord().nodeIdx];
}


double* SFCtg::getAccumSlice(const SplitCand *cand) {
  return &ctgSumAccum[getNumIdx(cand->getSplitCoord().predIdx) * splitCount * nCtg + cand->getSplitCoord().nodeIdx * nCtg];
}

/**
   @brief Run objects should not be deleted until after splits have been consumed.
 */
void SFReg::clear() {
  SplitFrontier::clear();
}


SFReg::~SFReg() {
}


SFCtg::~SFCtg() {
}


void SFCtg::clear() {
  SplitFrontier::clear();
}


/**
   @brief Sets level-specific values for the subclass.

   @param index contains the current level's index sets and state.

   @return void.
*/
void SFReg::levelPreset() {
  if (!mono.empty()) {
    ruMono = CallBack::rUnif(splitCount * mono.size());
  }
}


void SFCtg::levelPreset() {
  levelInitSumR(frame->getNPredNum());
  ctgSum = vector<vector<double> >(splitCount);

  // Hoist to replay().
  sumSquares = frontier->sumsAndSquares(ctgSum);
}


void SFCtg::levelInitSumR(PredictorT nPredNum) {
  if (nPredNum > 0) {
    ctgSumAccum = vector<double>(nPredNum * nCtg * splitCount);
    fill(ctgSumAccum.begin(), ctgSumAccum.end(), 0.0);
  }
}


int SFReg::getMonoMode(const SplitCand* cand) const {
  if (mono.empty())
    return 0;

  PredictorT numIdx = getNumIdx(cand->getSplitCoord().predIdx);
  double monoProb = mono[numIdx];
  double prob = ruMono[cand->getSplitCoord().nodeIdx * mono.size() + numIdx];
  if (monoProb > 0 && prob < monoProb) {
    return 1;
  }
  else if (monoProb < 0 && prob < -monoProb) {
    return -1;
  }
  else {
    return 0;
  }
}


void SplitFrontier::split() {
  splitCandidates();

  nuxMax = maxCandidates();
}


void SFCtg::splitCandidates() {
  OMPBound splitPos;
  OMPBound splitTop = splitCand.size();
#pragma omp parallel default(shared) private(splitPos) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitPos = 0; splitPos < splitTop; splitPos++) {
      splitCand[splitPos].split(this, obsPart.get());
    }
  }
}


void SFReg::splitCandidates() {
  OMPBound splitPos;
  OMPBound splitTop = splitCand.size();
#pragma omp parallel default(shared) private(splitPos) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitPos = 0; splitPos < splitTop; splitPos++) {
      splitCand[splitPos].split(this, obsPart.get());
    }
  }
}


vector<SplitNux> SplitFrontier::maxCandidates() {
  vector<SplitNux> nuxMax(splitCount); // Info initialized to zero.

  OMPBound splitIdx;
  OMPBound splitTop = splitCount;
#pragma omp parallel default(shared) private(splitIdx) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitIdx = 0; splitIdx < splitTop; splitIdx++) {
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

  return runningMax > 0.0 ? SplitNux(splitCand[argMax], frame) : SplitNux();
}


SplitSurvey SplitFrontier::consume(PreTree* pretree, vector<IndexSet>& indexSet, BV* replayExpl, BV* replayLeft) {
  replayExpl->clear();
  replayLeft->saturate();
  SplitSurvey survey;
  for (auto & iSet : indexSet) {
    consume(pretree, iSet, replayExpl, replayLeft, survey);
  }
  clear();

  return survey;
}


void SplitFrontier::consume(PreTree* pretree,
                            IndexSet& iSet,
                            BV* replayExpl,
                            BV* replayLeft,
                            SplitSurvey& survey) const {
  if (isInformative(&iSet)) {
    branch(pretree, &iSet, replayExpl, replayLeft);
    survey.splitNext += frontier->splitCensus(iSet, survey);
  }
  else {
    survey.leafCount++;
  }
}


void SplitFrontier::branch(PreTree* pretree,
                           IndexSet* iSet,
                           BV* replayExpl,
                           BV* replayLeft) const {
  consumeCriterion(iSet);
  pretree->nonterminal(getInfo(iSet), iSet);

  if (getCardinality(iSet) > 0) {
    critRun(pretree, iSet, replayExpl, replayLeft);
  }
  else {
    critCut(pretree, iSet, replayExpl, replayLeft);
  }
}


void SplitFrontier::consumeCriterion(IndexSet* iSet) const {
  nuxMax[iSet->getSplitIdx()].consume(iSet);
}


void SplitFrontier::critCut(PreTree* pretree,
                            IndexSet* iSet,
                            BV* replayExpl,
                            BV* replayLeft) const {
  pretree->critCut(iSet, getPredIdx(iSet), getRankRange(iSet));
  vector<SumCount> ctgCrit(iSet->getCtgLeft().size());
  double sumExpl = blockReplay(iSet, getExplicitRange(iSet), leftIsExplicit(iSet), replayExpl, replayLeft, ctgCrit);
  iSet->criterionLR(sumExpl, ctgCrit, leftIsExplicit(iSet));
}


double SplitFrontier::blockReplay(IndexSet* iSet,
                                  const IndexRange& range,
                                  bool leftExpl,
                                  BV* replayExpl,
                                  BV* replayLeft,
                                  vector<SumCount>& ctgCrit) const {
  return obsPart->blockReplay(this, iSet, range, leftExpl, replayExpl, replayLeft, ctgCrit);
}


void SplitFrontier::critRun(PreTree* pretree,
                              IndexSet* iSet,
                            BV* replayExpl,
                            BV* replayLeft) const {
  pretree->critBits(iSet, getPredIdx(iSet), getCardinality(iSet));
  bool leftExpl;
  vector<SumCount> ctgCrit(iSet->getCtgLeft().size());
  double sumExpl = run->branch(this, iSet, pretree, replayExpl, replayLeft, ctgCrit, leftExpl);
  iSet->criterionLR(sumExpl, ctgCrit, leftExpl);
}


bool SplitFrontier::isInformative(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getInfo() > iSet->getMinInfo();
}


IndexT SplitFrontier::getLHExtent(const IndexSet& iSet) const {
  return nuxMax[iSet.getSplitIdx()].getLHExtent();
}


IndexT SplitFrontier::getPredIdx(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getPredIdx();
}

unsigned int SplitFrontier::getBufIdx(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getBufIdx();
}


PredictorT SplitFrontier::getCardinality(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getCardinality();
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
  
