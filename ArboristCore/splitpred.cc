// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitpred.cc

   @brief Methods to implement splitting of index-tree levels.

   @author Mark Seligman
 */


#include "index.h"
#include "splitpred.h"
#include "splitcand.h"
#include "level.h"
#include "runset.h"
#include "samplenux.h"
#include "samplepred.h"
#include "callback.h"
#include "framemap.h"
#include "rowrank.h"

// Post-split consumption:
#include "pretree.h"

vector<double> SPReg::mono;
unsigned int SPReg::predMono = 0;


/**
  @brief Constructor.  Initializes 'runFlags' to zero for the single-split root.
 */
SplitPred::SplitPred(const FrameTrain *_frameTrain,
		     const RowRank *_rowRank,
		     unsigned int _bagCount) :
  rowRank(_rowRank),
  frameTrain(_frameTrain),
  bagCount(_bagCount),
  noSet(bagCount * frameTrain->NPredFac()) {
}


/**
   @brief Destructor.
 */
SplitPred::~SplitPred() {
}


/**
   @brief Caches a local copy of the mono[] vector.
 */
void SPReg::Immutables(const vector<double> &feMono) {
  predMono = 0;
  for (auto monoProb : feMono) {
    mono.push_back(monoProb);
    predMono += monoProb != 0.0;
  }
}


void SPReg::DeImmutables() {
  mono.clear();
  predMono = 0;
}


/**
   @brief Constructor.
 */
SPReg::SPReg(const FrameTrain *_frameTrain,
	     const RowRank *_rowRank,
	     unsigned int _bagCount) :
  SplitPred(_frameTrain, _rowRank, _bagCount),
  ruMono(vector<double>(0)) {
  run = make_unique<Run>(0, frameTrain->NRow(), noSet);
}


/**
   @brief Constructor.

   @param sampleCtg is the sample vector for the tree, included for category lookup.
 */
SPCtg::SPCtg(const FrameTrain *frameTrain_,
	     const RowRank *rowRank_,
	     unsigned int bagCount_,
	     unsigned int nCtg_):
  SplitPred(frameTrain_, rowRank_, bagCount_),
  nCtg(nCtg_) {
  run = make_unique<Run>(nCtg, frameTrain->NRow(), noSet);
}


RunSet *SplitPred::rSet(unsigned int setIdx) const {
  return run->rSet(setIdx);
}


unsigned int SplitPred::denseRank(unsigned int predIdx) const {
  return rowRank->getDenseRank(predIdx);
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


void SplitPred::preSchedule(unsigned int splitIdx,
			    unsigned int predIdx,
			    unsigned int bufIdx) {
  splitCand.emplace_back(SplitCand(splitIdx, predIdx, bufIdx));
}


/**
   @brief Walks the list of split candidates and invalidates those which
   restaging has marked unsplitable as well as singletons persisting since
   initialization or as a result of bagging.  Fills in run counts, which
   values restaging has established precisely.
*/
void SplitPred::scheduleSplits(const IndexLevel *index,
			       const Level *levelFront) {
  vector<unsigned int> runCount;
  vector<SplitCand> sc2;
  unsigned int splitPrev = splitCount;
  for (auto & sg : splitCand) {
    if (sg.schedule(this, levelFront, index, runCount, sc2)) {
      unsigned int splitThis = sg.getSplitIdx();
      nCand[splitThis]++;
      if (splitPrev != splitThis) {
        candOff[splitThis] = sg.getVecPos();
        splitPrev = splitThis;
      }
    }
  }
  splitCand = move(sc2);

  setRunOffsets(runCount);
}


/**
   @brief Initializes level about to be split
 */
void SplitPred::levelInit(IndexLevel *index) {
  splitCount = index->getNSplit();
  prebias = move(vector<double>(splitCount));
  nCand = move(vector<unsigned int>(splitCount));
  fill(nCand.begin(), nCand.end(), 0);
  candOff = move(vector<unsigned int>(splitCount));
  fill(candOff.begin(), candOff.end(), splitCount); // inattainable.

  levelPreset(index); // virtual
  setPrebias(index);
}


void SplitPred::setPrebias(IndexLevel *index) {
  for (unsigned int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    setPrebias(splitIdx, index->getSum(splitIdx), index->getSCount(splitIdx));
  }
}


/**
   @brief Base method.  Clears per-level vectors.

   @return void.
 */
void SplitPred::levelClear() {
  prebias.clear();
  run->levelClear();
}


/**
   @brief Determines whether indexed predictor is a factor.

   @param predIdx is the predictor index.

   @return true iff predictor is a factor.
 */
bool SplitPred::isFactor(unsigned int predIdx) const {
  return frameTrain->IsFactor(predIdx);
}


/**
   @brief Determines whether indexed predictor is numerica.

   @param predIdx is the predictor index.

   @return true iff predictor is numeric.
 */
unsigned int SPCtg::getNumIdx(unsigned int predIdx) const {
  return frameTrain->NumIdx(predIdx);
}


/**
   @brief Run objects should not be deleted until after splits have been consumed.
 */
void SPReg::levelClear() {
  SplitPred::levelClear();
}


SPReg::~SPReg() {
}


SPCtg::~SPCtg() {
}


void SPCtg::levelClear() {
  SplitPred::levelClear();
}


/**
   @brief Sets level-specific values for the subclass.

   @param index contains the current level's index sets and state.

   @return void.
*/
void SPReg::levelPreset(IndexLevel *index) {
  if (predMono > 0) {
    unsigned int monoCount = splitCount * frameTrain->NPred(); // Clearly too big.
    ruMono = move(CallBack::rUnif(monoCount));
  }
}


/**
   @brief As above, but categorical response.  Initializes per-level sum and
   FacRun vectors.

   @return void.
*/
void SPCtg::levelPreset(IndexLevel *index) {
  levelInitSumR(frameTrain->NPredNum());
  sumSquares = move(vector<double>(splitCount));
  ctgSum = move(vector<double>(splitCount * nCtg));  
  fill(sumSquares.begin(), sumSquares.end(), 0.0);
  fill(ctgSum.begin(), ctgSum.end(), 0.0);
  // Hoist to replay().
  index->sumsAndSquares(nCtg, sumSquares, ctgSum);
}


/**
   @brief Initializes the accumulated-sum checkerboard used by
   numerical predictors.

   @param nPredNum is the number of numerical predictors.

   @return void.
 */
void SPCtg::levelInitSumR(unsigned int nPredNum) {
  if (nPredNum > 0) {
    ctgSumAccum = move(vector<double>(nPredNum * nCtg * splitCount));
    fill(ctgSumAccum.begin(), ctgSumAccum.end(), 0.0);
  }
}


/**
   @brief Determines whether a regression pair undergoes constrained splitting.

   @return The sign of the constraint, if within the splitting probability, else zero.
*/
int SPReg::MonoMode(unsigned int splitIdx,
		    unsigned int predIdx) const {
  if (predMono == 0)
    return 0;

  double monoProb = mono[predIdx];
  int sign = monoProb > 0.0 ? 1 : (monoProb < 0.0 ? -1 : 0);
  return sign * ruMono[splitIdx] < monoProb ? sign : 0;
}


vector<SplitCand> SplitPred::split(const SamplePred *samplePred) {
  splitCandidates(samplePred);

  return move(maxCandidates());
}


void SPCtg::splitCandidates(const SamplePred *samplePred) {
  OMPBound splitPos;
#pragma omp parallel default(shared) private(splitPos)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitPos = 0; splitPos < OMPBound(splitCand.size()); splitPos++) {
      splitCand[splitPos].split(this, samplePred);
    }
  }
}


void SPReg::splitCandidates(const SamplePred *samplePred) {
  OMPBound splitPos;
#pragma omp parallel default(shared) private(splitPos)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitPos = 0; splitPos < OMPBound(splitCand.size()); splitPos++) {
      splitCand[splitPos].split(this, samplePred);
    }
  }
}


vector<SplitCand> SplitPred::maxCandidates() {
  vector<SplitCand> candMax(splitCount);

  OMPBound splitIdx;
#pragma omp parallel default(shared) private(splitIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitIdx = 0; splitIdx < OMPBound(splitCount); splitIdx++) {
      maxSplit(candMax[splitIdx], candOff[splitIdx], nCand[splitIdx]);
    }
  }
  splitCand.clear();
  candOff.clear();
  nCand.clear();

  return move(candMax);
}


void SplitPred::maxSplit(SplitCand &candMax,
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



/**
   @brief Imputes dense rank values as residuals.

   @param[out] idxSup outputs the sup of index values having ranks below the
   dense rank.

   @param[in, out] sumDense inputs the reponse sum over the node and outputs the
   residual sum.

   @param[in, out] sCountDense inputs the response sample count over the node and
   outputs the residual count.

   @return supremum of indices to the left ot the dense rank.
*/
unsigned int SPReg::Residuals(const SampleRank spn[],
			      unsigned int idxStart,
			      unsigned int idxEnd,
			      unsigned int rankDense,
			      unsigned int &denseLeft,
			      unsigned int &denseRight,
			      double &sumDense,
			      unsigned int &sCountDense) const {
  unsigned int denseCut = idxEnd; // Defaults to highest index.
  double sumTot = 0.0;
  unsigned int sCountTot = 0;
  for (int idx = int(idxEnd); idx >= int(idxStart); idx--) {
    unsigned int sampleCount, rkThis;
    FltVal ySum;
    spn[idx].regFields(ySum, rkThis, sampleCount);
    denseCut = rkThis > rankDense ? idx : denseCut;
    sCountTot += sampleCount;
    sumTot += ySum;
  }
  sumDense -= sumTot;
  sCountDense -= sCountTot;

  // Dense blob is either left, right or neither.
  denseRight = (denseCut == idxEnd && spn[denseCut].getRank() < rankDense);  
  denseLeft = (denseCut == idxStart && spn[denseCut].getRank() > rankDense);
  
  return denseCut;
}


/**
   @brief Imputes dense rank values as residuals.

   @param idxSup outputs the sup of index values having ranks below the
   dense rank.

   @return true iff left bound has rank less than dense rank.
*/
unsigned int SPCtg::Residuals(const SampleRank spn[],
			      unsigned int splitIdx,
			      unsigned int idxStart,
			      unsigned int idxEnd,
			      unsigned int rankDense,
			      bool &denseLeft,
			      bool &denseRight,
			      double &sumDense,
			      unsigned int &sCountDense,
			      vector<double> &ctgSumDense) const {
  vector<double> ctgAccum;
  ctgSumDense.reserve(nCtg);
  ctgAccum.reserve(nCtg);
  for (unsigned int ctg = 0; ctg < nCtg; ctg++) {
    ctgSumDense.push_back(getCtgSum(splitIdx, ctg));
    ctgAccum.push_back(0.0);
  }
  unsigned int denseCut = idxEnd; // Defaults to highest index.
  double sumTot = 0.0;
  unsigned int sCountTot = 0;
  for (int idx = int(idxEnd); idx >= int(idxStart); idx--) {
    // Accumulates statistics over explicit range.
    unsigned int yCtg, rkThis;
    FltVal ySum;
    unsigned int sampleCount = spn[idx].ctgFields(ySum, rkThis, yCtg);
    ctgAccum[yCtg] += ySum;
    denseCut = rkThis >= rankDense ? idx : denseCut;
    sCountTot += sampleCount;
    sumTot += ySum;
  }
  sumDense -= sumTot;
  sCountDense -= sCountTot;
  for (unsigned int ctg = 0; ctg < nCtg; ctg++) {
    ctgSumDense[ctg] -= ctgAccum[ctg];
  }

  // Dense blob is either left, right or neither.
  denseRight = (denseCut == idxEnd && spn[denseCut].getRank() < rankDense);  
  denseLeft = (denseCut == idxStart && spn[denseCut].getRank() > rankDense);

  return denseCut;
}


void SPCtg::applyResiduals(unsigned int splitIdx,
			   unsigned int predIdx,
			   double &ssL,
			   double &ssR,
			   vector<double> &sumDenseCtg) {
  unsigned int numIdx = getNumIdx(predIdx);
  for (unsigned int ctg = 0; ctg < nCtg; ctg++) {
    double ySum = sumDenseCtg[ctg];
    double sumRCtg = accumCtgSum(splitIdx, numIdx, ctg, ySum);
    ssR += ySum * (ySum + 2.0 * sumRCtg);
    double sumLCtg = getCtgSum(splitIdx, ctg) - sumRCtg;
    ssL += ySum * (ySum - 2.0 * sumLCtg);
  }
}
