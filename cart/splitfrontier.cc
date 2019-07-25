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
#include "ompthread.h"

// Post-split consumption:
#include "pretree.h"

vector<double> SFReg::mono; // Numeric monotonicity constraints.

/**
  @brief Constructor.  Initializes 'runFlags' to zero for the single-split root.
 */
SplitFrontier::SplitFrontier(const SummaryFrame* frame_,
		     unsigned int bagCount) :
  frame(frame_),
  rankedFrame(frame->getRankedFrame()),
  noSet(bagCount * frame->getNPredFac()) {
}


SplitFrontier::~SplitFrontier() {
}


void SFReg::Immutables(const SummaryFrame* frame,
                       const vector<double> &bridgeMono) {
  auto numFirst = frame->getNumFirst();
  auto numExtent = frame->getNPredNum();
  auto monoCount = count_if(bridgeMono.begin() + numFirst, bridgeMono.begin() + numExtent, [](double prob) { return prob != 0.0; });
  if (monoCount > 0) {
    mono = vector<double>(frame->getNPredNum());
    mono.assign(bridgeMono.begin() + frame->getNumFirst(), bridgeMono.begin() + frame->getNumFirst() + frame->getNPredNum());
  }
}


void SFReg::DeImmutables() {
  mono.clear();
}


SFReg::SFReg(const SummaryFrame* frame,
	     unsigned int bagCount) :
  SplitFrontier(frame, bagCount),
  ruMono(vector<double>(0)) {
  run = make_unique<Run>(0, frame->getNRow(), noSet);
}


/**
   @brief Constructor.
 */
SFCtg::SFCtg(const SummaryFrame* frame,
	     unsigned int bagCount,
	     unsigned int nCtg_):
  SplitFrontier(frame, bagCount),
  nCtg(nCtg_) {
  run = make_unique<Run>(nCtg, frame->getNRow(), noSet);
}


RunSet *SplitFrontier::rSet(unsigned int setIdx) const {
  return run->rSet(setIdx);
}


unsigned int SplitFrontier::getDenseRank(const SplitCand* cand) const {
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


IndexType SplitFrontier::preschedule(const Frontier* index,
                            const SplitCoord& splitCoord,
			    unsigned int bufIdx) {
  splitCand.emplace_back(SplitCand(this, index, splitCoord, bufIdx, noSet));
  return index->getExtent(splitCoord.nodeIdx);
}


/**
   @brief Walks the list of split candidates and invalidates those which
   restaging has marked unsplitable as well as singletons persisting since
   initialization or as a result of bagging.  Fills in run counts, which
   values restaging has established precisely.
*/
void SplitFrontier::scheduleSplits(const Frontier *index,
			       const Level *levelFront) {
  vector<unsigned int> runCount;
  vector<SplitCand> sc2;
  unsigned int splitPrev = splitCount;
  for (auto & sg : splitCand) {
    if (sg.schedule(levelFront, index, runCount)) {
      unsigned int splitThis = sg.getSplitCoord().nodeIdx;
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
}


/**
   @brief Initializes level about to be split
 */
void SplitFrontier::init(Frontier *index) {
  splitCount = index->getNSplit();
  prebias = vector<double>(splitCount);
  nCand = vector<unsigned int>(splitCount);
  fill(nCand.begin(), nCand.end(), 0);
  candOff = vector<unsigned int>(splitCount);
  fill(candOff.begin(), candOff.end(), splitCount); // inattainable.

  levelPreset(index); // virtual
  setPrebias(index);
}


void SplitFrontier::setPrebias(Frontier *index) {
  for (unsigned int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    setPrebias(splitIdx, index->getSum(splitIdx), index->getSCount(splitIdx));
  }
}


/**
   @brief Base method.  Clears per-frontier vectors.

   @return void.
 */
void SplitFrontier::clear() {
  prebias.clear();
  run->clear();
}


/**
   @brief Determines whether indexed predictor is a factor.

   @param predIdx is the predictor index.

   @return true iff predictor is a factor.
 */
bool SplitFrontier::isFactor(const SplitCoord& splitCoord) const {
  return frame->isFactor(splitCoord.predIdx);
}


double SFCtg::getSumSquares(const SplitCand *cand) const {
  return sumSquares[cand->getSplitCoord().nodeIdx];
}


/**
   @brief Determines whether indexed predictor is numerica.

   @param predIdx is the predictor index.

   @return true iff predictor is numeric.
 */
unsigned int SplitFrontier::getNumIdx(unsigned int predIdx) const {
  return frame->getNumIdx(predIdx);
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
void SFReg::levelPreset(Frontier *index) {
  if (!mono.empty()) {
    ruMono = CallBack::rUnif(splitCount * mono.size());
  }
}


/**
   @brief As above, but categorical response.  Initializes per-level sum and
   FacRun vectors.
*/
void SFCtg::levelPreset(Frontier *index) {
  levelInitSumR(frame->getNPredNum());
  ctgSum = vector<vector<double> >(splitCount);

  // Hoist to replay().
  sumSquares = index->sumsAndSquares(ctgSum);
}


/**
   @brief Initializes the accumulated-sum checkerboard used by
   numerical predictors.

   @param nPredNum is the number of numerical predictors.
 */
void SFCtg::levelInitSumR(unsigned int nPredNum) {
  if (nPredNum > 0) {
    ctgSumAccum = vector<double>(nPredNum * nCtg * splitCount);
    fill(ctgSumAccum.begin(), ctgSumAccum.end(), 0.0);
  }
}


/**
   @brief Determines whether a regression pair undergoes constrained splitting.

   @return The sign of the constraint, if within the splitting probability, else zero.
*/
int SFReg::getMonoMode(const SplitCand* cand) const {
  if (mono.empty())
    return 0;

  unsigned int numIdx = getNumIdx(cand->getSplitCoord().predIdx);
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


void SplitFrontier::split(const ObsPart *samplePred) {
  splitCandidates(samplePred);

  nuxMax = move(maxCandidates());
}


void SFCtg::splitCandidates(const ObsPart *samplePred) {
  OMPBound splitPos;
  OMPBound splitTop = splitCand.size();
#pragma omp parallel default(shared) private(splitPos) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitPos = 0; splitPos < splitTop; splitPos++) {
      splitCand[splitPos].split(this, samplePred);
    }
  }
}


void SFReg::splitCandidates(const ObsPart *samplePred) {
  OMPBound splitPos;
  OMPBound splitTop = splitCand.size();
#pragma omp parallel default(shared) private(splitPos) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitPos = 0; splitPos < splitTop; splitPos++) {
      splitCand[splitPos].split(this, samplePred);
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


SplitNux SplitFrontier::maxSplit(unsigned int splitOff,
                                 unsigned int nCandSplit) const {
  const auto slotSup = splitOff + nCandSplit;
  unsigned int argMax = slotSup;
  double maxInfo = 0.0;
  for ( ; splitOff < slotSup; splitOff++) {
    if (splitCand[splitOff].getInfo() > maxInfo) {
      maxInfo = splitCand[splitOff].getInfo();
      argMax = splitOff;
    }
  }

  if (argMax != slotSup) {
    return SplitNux(splitCand[argMax], frame);
  }
  else {
    return SplitNux();
  }
}


bool SplitFrontier::branch(Frontier* frontier, PreTree* pretree, IndexSet* iSet) const {
  nonterminal(iSet);
  pretree->nonterminal(getInfo(iSet), frontier, iSet);
  if (getCardinality(iSet) > 0) {
    return critRun(frontier, pretree, iSet);
  }
  else {
    return critCut(frontier, pretree, iSet);
  }
}


void SplitFrontier::nonterminal(IndexSet* iSet) const {
  double minInfo;
  IndexType lhSCount;
  IndexType lhExtent;
  nuxMax[iSet->getSplitIdx()].nonterminal(minInfo, lhSCount, lhExtent);
  iSet->nonterminal(minInfo, lhSCount, lhExtent);
}


bool SplitFrontier::critCut(Frontier* frontier,
                            PreTree* pretree,
                            IndexSet* iSet) const {
  pretree->critCut(iSet, getPredIdx(iSet), getRankRange(iSet));
  iSet->blockReplay(frontier, getExplicitRange(iSet));
  return leftIsExplicit(iSet);
}


bool SplitFrontier::critRun(Frontier* frontier,
                            PreTree* pretree,
                            IndexSet* iSet) const {
  pretree->critBits(iSet, getPredIdx(iSet), getCardinality(iSet));
  return run->branch(this, iSet, pretree, frontier);
}


bool SplitFrontier::isInformative(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getInfo() > iSet->getMinInfo();
}


IndexType SplitFrontier::getLHExtent(const IndexSet& iSet) const {
  return nuxMax[iSet.getSplitIdx()].getLHExtent();
}


IndexType SplitFrontier::getPredIdx(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getPredIdx();
}

unsigned int SplitFrontier::getBufIdx(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getBufIdx();
}

unsigned int SplitFrontier::getCardinality(const IndexSet* iSet) const {
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


unsigned int SplitFrontier::getSetIdx(const IndexSet* iSet) const {
  return nuxMax[iSet->getSplitIdx()].getSetIdx();
}
  
