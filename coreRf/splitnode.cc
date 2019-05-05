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


#include "index.h"
#include "splitnode.h"
#include "splitcand.h"
#include "level.h"
#include "runset.h"
#include "samplenux.h"
#include "samplepred.h"
#include "callback.h"
#include "framemap.h"
#include "rankedset.h"
#include "ompthread.h"

// Post-split consumption:
#include "pretree.h"

vector<double> SPReg::mono; // Numeric monotonicity constraints.

/**
  @brief Constructor.  Initializes 'runFlags' to zero for the single-split root.
 */
SplitNode::SplitNode(const FrameMap *frameMap_,
		     const RowRank *rowRank_,
		     unsigned int bagCount) :
  rowRank(rowRank_),
  frameMap(frameMap_),
  noSet(bagCount * frameMap->getNPredFac()) {
}


/**
   @brief Destructor.
 */
SplitNode::~SplitNode() {
}


void SPReg::Immutables(const FrameMap* frameMap,
                       const vector<double> &bridgeMono) {
  auto numFirst = frameMap->NumFirst();
  auto numExtent = frameMap->getNPredNum();
  auto monoCount = count_if(bridgeMono.begin() + numFirst, bridgeMono.begin() + numExtent, [](double prob) { return prob != 0.0; });
  if (monoCount > 0) {
    mono = vector<double>(frameMap->getNPredNum());
    mono.assign(bridgeMono.begin() + frameMap->NumFirst(), bridgeMono.begin() + frameMap->NumFirst() + frameMap->getNPredNum());
  }
}


void SPReg::DeImmutables() {
  mono.clear();
}


SPReg::SPReg(const FrameMap *_frameMap,
	     const RowRank *_rowRank,
	     unsigned int bagCount) :
  SplitNode(_frameMap, _rowRank, bagCount),
  ruMono(vector<double>(0)) {
  run = make_unique<Run>(0, frameMap->getNRow(), noSet);
}


/**
   @brief Constructor.
 */
SPCtg::SPCtg(const FrameMap *frameMap_,
	     const RowRank *rowRank_,
	     unsigned int bagCount,
	     unsigned int nCtg_):
  SplitNode(frameMap_, rowRank_, bagCount),
  nCtg(nCtg_) {
  run = make_unique<Run>(nCtg, frameMap->getNRow(), noSet);
}


RunSet *SplitNode::rSet(unsigned int setIdx) const {
  return run->rSet(setIdx);
}


unsigned int SplitNode::getDenseRank(const SplitCand* cand) const {
  return rowRank->getDenseRank(cand->getPredIdx());
}


/**
   @brief Sets quick lookup offets for Run object.

   @return void.
 */
void SPReg::setRunOffsets(const vector<unsigned int> &runCount) {
  run->offsetsReg(runCount);
}


/**
   @brief Sets quick lookup offsets for Run object.
 */
void SPCtg::setRunOffsets(const vector<unsigned int> &runCount) {
  run->offsetsCtg(runCount);
}


void SplitNode::preschedule(unsigned int splitIdx,
			    unsigned int predIdx,
			    unsigned int bufIdx) {
  splitCand.emplace_back(SplitCand(splitIdx, predIdx, bufIdx, noSet));
}


/**
   @brief Walks the list of split candidates and invalidates those which
   restaging has marked unsplitable as well as singletons persisting since
   initialization or as a result of bagging.  Fills in run counts, which
   values restaging has established precisely.
*/
void SplitNode::scheduleSplits(const IndexLevel *index,
			       const Level *levelFront) {
  vector<unsigned int> runCount;
  vector<SplitCand> sc2;
  unsigned int splitPrev = splitCount;
  for (auto & sg : splitCand) {
    if (sg.schedule(this, levelFront, index, runCount)) {
      unsigned int splitThis = sg.getSplitIdx();
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
void SplitNode::levelInit(IndexLevel *index) {
  splitCount = index->getNSplit();
  prebias = vector<double>(splitCount);
  nCand = vector<unsigned int>(splitCount);
  fill(nCand.begin(), nCand.end(), 0);
  candOff = vector<unsigned int>(splitCount);
  fill(candOff.begin(), candOff.end(), splitCount); // inattainable.

  levelPreset(index); // virtual
  setPrebias(index);
}


void SplitNode::setPrebias(IndexLevel *index) {
  for (unsigned int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    setPrebias(splitIdx, index->getSum(splitIdx), index->getSCount(splitIdx));
  }
}


/**
   @brief Base method.  Clears per-level vectors.

   @return void.
 */
void SplitNode::levelClear() {
  prebias.clear();
  run->levelClear();
}


/**
   @brief Determines whether indexed predictor is a factor.

   @param predIdx is the predictor index.

   @return true iff predictor is a factor.
 */
bool SplitNode::isFactor(unsigned int predIdx) const {
  return frameMap->isFactor(predIdx);
}


double SPCtg::getSumSquares(const SplitCand *cand) const {
  return sumSquares[cand->getSplitIdx()];
}


/**
   @brief Determines whether indexed predictor is numerica.

   @param predIdx is the predictor index.

   @return true iff predictor is numeric.
 */
unsigned int SplitNode::getNumIdx(unsigned int predIdx) const {
  return frameMap->getNumIdx(predIdx);
}


const vector<double>& SPCtg::getSumSlice(const SplitCand* cand) {
  return ctgSum[cand->getSplitIdx()];
}


double* SPCtg::getAccumSlice(const SplitCand *cand) {
  return &ctgSumAccum[getNumIdx(cand->getPredIdx()) * splitCount * nCtg + cand->getSplitIdx() * nCtg];
}

/**
   @brief Run objects should not be deleted until after splits have been consumed.
 */
void SPReg::levelClear() {
  SplitNode::levelClear();
}


SPReg::~SPReg() {
}


SPCtg::~SPCtg() {
}


void SPCtg::levelClear() {
  SplitNode::levelClear();
}


/**
   @brief Sets level-specific values for the subclass.

   @param index contains the current level's index sets and state.

   @return void.
*/
void SPReg::levelPreset(IndexLevel *index) {
  if (!mono.empty()) {
    ruMono = CallBack::rUnif(splitCount * mono.size());
  }
}


/**
   @brief As above, but categorical response.  Initializes per-level sum and
   FacRun vectors.
*/
void SPCtg::levelPreset(IndexLevel *index) {
  levelInitSumR(frameMap->getNPredNum());
  ctgSum = vector<vector<double> >(splitCount);

  // Hoist to replay().
  sumSquares = index->sumsAndSquares(ctgSum);
}


/**
   @brief Initializes the accumulated-sum checkerboard used by
   numerical predictors.

   @param nPredNum is the number of numerical predictors.
 */
void SPCtg::levelInitSumR(unsigned int nPredNum) {
  if (nPredNum > 0) {
    ctgSumAccum = vector<double>(nPredNum * nCtg * splitCount);
    fill(ctgSumAccum.begin(), ctgSumAccum.end(), 0.0);
  }
}


/**
   @brief Determines whether a regression pair undergoes constrained splitting.

   @return The sign of the constraint, if within the splitting probability, else zero.
*/
int SPReg::getMonoMode(const SplitCand* cand) const {
  if (mono.empty())
    return 0;

  unsigned int numIdx = getNumIdx(cand->getPredIdx());
  double monoProb = mono[numIdx];
  double prob = ruMono[cand->getSplitIdx() * mono.size() + numIdx];
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


vector<SplitCand> SplitNode::split(const SamplePred *samplePred) {
  splitCandidates(samplePred);

  return maxCandidates();
}


void SPCtg::splitCandidates(const SamplePred *samplePred) {
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


void SPReg::splitCandidates(const SamplePred *samplePred) {
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


vector<SplitCand> SplitNode::maxCandidates() {
  vector<SplitCand> candMax(splitCount); // Info initialized to zero.

  OMPBound splitIdx;
  OMPBound splitTop = splitCount;
#pragma omp parallel default(shared) private(splitIdx) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitIdx = 0; splitIdx < splitTop; splitIdx++) {
      maxSplit(candMax[splitIdx], candOff[splitIdx], nCand[splitIdx]);
    }
  }
  splitCand.clear();
  candOff.clear();
  nCand.clear();

  return candMax;
}


void SplitNode::maxSplit(SplitCand &candMax,
                         unsigned int splitOff,
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
    candMax = splitCand[argMax];
  }
}
