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

//#include <iostream>
//using namespace std;

#include "index.h"
#include "splitpred.h"
#include "splitsig.h"
#include "bottom.h"
#include "runset.h"
#include "samplepred.h"
#include "callback.h"
#include "sample.h"
#include "predblock.h"
#include "rowrank.h"

unsigned int SplitPred::nPred = 0;
unsigned int SplitPred::predFixed = 0;
const double *SplitPred::predProb = 0;

const double *SPReg::feMono = 0;
unsigned int SPReg::predMono = 0;
unsigned int SPCtg::ctgWidth = 0;

/**
  @brief Constructor.  Initializes 'runFlags' to zero for the single-split root.
 */
SplitPred::SplitPred(const PMTrain *_pmTrain, const RowRank *_rowRank, SamplePred *_samplePred, unsigned int _bagCount) : rowRank(_rowRank), pmTrain(_pmTrain), bagCount(_bagCount), noSet(bagCount * pmTrain->NPredFac()), samplePred(_samplePred) {
}


/**
   @brief Destructor.  Deletes dangling 'runFlags' vector, which should be
   nonempty.
 */
SplitPred::~SplitPred() {
  delete run;
}


void SplitPred::Immutables(unsigned int _nPred, unsigned int _ctgWidth, unsigned int _predFixed, const double _predProb[], const double _regMono[]) {
  nPred = _nPred; // TODO:  Derive from predProb.size(), when changed to be vector.
  predFixed = _predFixed;
  predProb = _predProb;

  if (_ctgWidth > 0) {
    SPCtg::Immutables(_ctgWidth);
  }
  else {
    SPReg::Immutables(nPred, _regMono);
  }
}


void SplitPred::DeImmutables() {
  nPred = 0;
  predFixed = 0;

  // 'ctgWidth' distinguishes regression from classification.
  if (SPCtg::CtgWidth() > 0)
    SPCtg::DeImmutables();
  else
    SPReg::DeImmutables();
}


/**
   @brief Caches a local copy of the mono[] vector.
 */

void SPReg::Immutables(unsigned int _nPred, const double _mono[]) {
  predMono = 0;
  feMono = _mono;
  for (unsigned int i = 0; i < _nPred; i++) {
    double monoProb = feMono[i];
    predMono += monoProb != 0.0;
  }
}


void SPReg::DeImmutables() {
  predMono = 0;
}


void SPCtg::Immutables(unsigned int _ctgWidth) {
  ctgWidth = _ctgWidth;
}


void SPCtg::DeImmutables() {
  ctgWidth = 0;
}


/**
   @brief Constructor.

   @param samplePred holds (re)staged node contents.
 */
SPReg::SPReg(const PMTrain *_pmTrain, const RowRank *_rowRank, SamplePred *_samplePred, unsigned int _bagCount) : SplitPred(_pmTrain, _rowRank, _samplePred, _bagCount), ruMono(0) {
  run = new Run(0, pmTrain->NRow(), noSet);
}


/**
   @brief Constructor.

   @param samplePred holds (re)staged node contents.

   @param sampleCtg is the sample vector for the tree, included for category lookup.
 */
SPCtg::SPCtg(const PMTrain *_pmTrain, const RowRank *_rowRank, SamplePred *_samplePred, const std::vector<SampleNode> &_sampleCtg, unsigned int _bagCount): SplitPred(_pmTrain, _rowRank, _samplePred, _bagCount), sampleCtg(_sampleCtg) {
  run = new Run(ctgWidth, pmTrain->NRow(), noSet);
}


/**
   @brief Sets per-level state enabling pre-bias computation.

   @param index is the Index context.

   @param levelCount is the potential number of splitting nodes in the upcoming level.

   @return split count.
*/
void SplitPred::LevelInit(IndexLevel &index) {
  levelCount = index.LevelCount();
  std::vector<bool> unsplitable(levelCount);
  std::fill(unsplitable.begin(), unsplitable.end(), false);
  LevelPreset(index, unsplitable);
  SetPrebias(index); // Depends on state from LevelPreset()
  Splitable(unsplitable);
}


/**
   @brief Sets (Gini) pre-bias value according to response type.

   @param index holds the index set nodes.

   @return void.
*/
void SplitPred::SetPrebias(IndexLevel &index) {
  for (unsigned int levelIdx = 0; levelIdx < levelCount; levelIdx++) {
    index.SetPrebias(levelIdx, Prebias(index, levelIdx));
  }
}


RunSet *SplitPred::RSet(unsigned int setIdx) const {
  return run->RSet(setIdx);
}


unsigned int SplitPred::DenseRank(unsigned int predIdx) const {
  return rowRank->DenseRank(predIdx);
}


/**
   @brief Needs a dense numbering of pred-mono split candidates.

   @return void.
 */
void SPReg::LevelInit(IndexLevel &index) {
  SplitPred::LevelInit(index);
  if (predMono > 0) {
    unsigned int monoCount = levelCount * nPred; // Clearly too big.
    ruMono = new double[monoCount];
    CallBack::RUnif(monoCount, ruMono);
  }
  else {
    ruMono = 0;
  }
}


/**
   @brief Sets quick lookup offets for Run object.

   @return void.
 */
void SPReg::RunOffsets(const std::vector<unsigned int> &runCount) {
  run->RunSets(runCount);
  run->OffsetsReg();
}


/**
   @brief Sets quick lookup offsets for Run object.
 */
void SPCtg::RunOffsets(const std::vector<unsigned int> &runCount) {
  run->RunSets(runCount);
  run->OffsetsCtg();
}


/**
   @brief Signals Bottom to schedule splitable pairs.

   @param unsplitable lists unsplitable nodes.

   @return void.
*/
void SplitPred::Splitable(const std::vector<bool> &unsplitable) {
    // TODO:  Pre-empt overflow by walking wide subtrees depth-first.
  int cellCount = levelCount * nPred;

  double *ruPred = new double[cellCount];
  CallBack::RUnif(cellCount, ruPred);

  BHPair *heap;
  if (predFixed > 0)
    heap = new BHPair[cellCount];
  else
    heap = 0;

  for (unsigned int levelIdx = 0; levelIdx < levelCount; levelIdx++) {
    if (unsplitable[levelIdx])
      continue; // No predictor splitable
    unsigned int splitOff = levelIdx * nPred;
    if (predFixed == 0) { // Probability of predictor splitable.
      PrescheduleProb(levelIdx, &ruPred[splitOff]);
    }
    else { // Fixed number of predictors splitable.
      PrescheduleFixed(levelIdx, &ruPred[splitOff], &heap[splitOff]);
    }
  }

  if (heap != 0)
    delete [] heap;
  delete [] ruPred;
}


/**
   @brief Set splitable flag by Bernoulli sampling.

   @param ruPred is a vector of uniformly-sampled variates.

   @param flags outputs the splitability predicate.

   @return void, with output vector.
 */
void SplitPred::PrescheduleProb(unsigned int levelIdx, const double ruPred[]) {
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    if (ruPred[predIdx] < predProb[predIdx]) {
      (void) Preschedule(levelIdx, predIdx);
    }
  }
}

 
/**
   @brief Sets splitable flag for a fixed number of predictors.

   @param ruPred is a vector of uniformly-sampled variates.

   @param heap orders probability-weighted variates.

   @param flags outputs the splitability predicate.

   @return void, with output vector.
 */
void SplitPred::PrescheduleFixed(unsigned int levelIdx, const double ruPred[], BHPair heap[]) {
  // Inserts negative, weighted probability value:  choose from lowest.
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    BHeap::Insert(heap, predIdx, -ruPred[predIdx] * predProb[predIdx]);
  }

  // Pops 'predFixed' items in order of increasing value.
  unsigned int schedCount = 0;
  for (unsigned int heapSize = nPred; heapSize > 0; heapSize--) {
    unsigned int predIdx = BHeap::SlotPop(heap, heapSize - 1);
    schedCount += Preschedule(levelIdx, predIdx) ? 1 : 0;
    if (schedCount == predFixed)
      break;
  }
}


/**
   @brief Initializes the list of split candidates with extant cells.

   @return true iff candidate has been prescheduled.
 */
bool SplitPred::Preschedule(unsigned int levelIdx, unsigned int predIdx) {
  SplitCoord sg;
  return sg.Preschedule(bottom, levelIdx, predIdx, splitCoord);
}


bool SplitCoord::Preschedule(Bottom *bottom, unsigned int _levelIdx, unsigned int _predIdx, std::vector<SplitCoord> &splitCoord) {
  unsigned int _bufIdx;
  if (bottom->Preschedule(_levelIdx, _predIdx, _bufIdx)) {
    levelIdx = _levelIdx;
    predIdx = _predIdx;
    bufIdx = _bufIdx;
    splitCoord.push_back(*this);
    return true;
  }
  else {
    return false;
  }
}


/**
   @brief Walks the list of split candidates and invalidates those which
   restaging has marked unsplitable as well as singletons persisting since
   initialization or as a result of bagging.  Fills in run counts, which
   values restaging has established precisely.
*/
void SplitPred::ScheduleSplits(const IndexLevel &index) {
  std::vector<unsigned int> runCount;
  std::vector<SplitCoord> sc2;
  for (auto & sg : splitCoord) {
    sg.Schedule(bottom, index, noSet, runCount, sc2);
  }
  splitCoord = std::move(sc2);

  RunOffsets(runCount);
}


/**
   @brief Retains split coordinate iff target is not a singleton.  Pushes
   back run counts, if applicable.

   @param sg holds partially-initialized split coordinates.

   @param runCount accumulates nontrivial run counts.

   @param sc2 accumulates "actual" splitting coordinates.

   @return void, with output reference vectors.
 */
void SplitCoord::Schedule(const Bottom *bottom, const IndexLevel &index, unsigned int noSet, std::vector<unsigned int> &runCount, std::vector<SplitCoord> &sc2) {
  unsigned int rCount;
  if (bottom->ScheduleSplit(levelIdx, predIdx, rCount)) {
    InitLate(bottom, index, sc2.size(), rCount > 1 ? runCount.size() : noSet);
    if (rCount > 1) {
      runCount.push_back(rCount);
    }
    sc2.push_back(*this);
  }
}


/**
   @brief Initializes field values known only following restaging.  Entry
   singletons should not reach here.

   @return true iff non-singleton.
 */
void SplitCoord::InitLate(const Bottom *bottom, const IndexLevel &index, unsigned int _splitPos, unsigned int _setIdx) {
  splitPos = _splitPos;
  setIdx = _setIdx;
  unsigned int extent;
  preBias = index.SplitFields(levelIdx, idxStart, extent, sCount, sum);
  implicit = bottom->AdjustDense(levelIdx, predIdx, idxStart, extent);
  idxEnd = idxStart + extent - 1; // May overflow if singleton:  invalid.
}


/**
   @brief Base method.  Deletes per-level run and split-flags vectors.

   @return void.
 */
void SplitPred::LevelClear() {
  run->LevelClear();
}


bool SplitPred::IsFactor(unsigned int predIdx) const {
  return pmTrain->IsFactor(predIdx);
}


unsigned int SplitPred::NumIdx(unsigned int predIdx) const {
  return pmTrain->NumIdx(predIdx);
}


/**
   @brief Run objects should not be deleted until after splits have been consumed.
 */
void SPReg::LevelClear() {
  if (ruMono != 0) {
    delete [] ruMono;
    ruMono = 0;
  }
  SplitPred::LevelClear();
}


SPReg::~SPReg() {
}


SPCtg::~SPCtg() {
}


void SPCtg::LevelClear() {
  SplitPred::LevelClear();
}


/**
   @brief Currently just a stub.

   @param index is not used by this instantiation.

   @param levelCount is the number of live index nodes.

   @return vector of unsplitable indices.
*/
void SPReg::LevelPreset(const IndexLevel &index, std::vector<bool> &unsplitable) {
}


/**
  @brief Weight-variance pre-bias computation for regression response.

  @param levelIdx is the level-relative node index.

  @param sCount is the number of samples subsumed by the index node.

  @param sum is the sum of samples subsumed by the index node.

  @return square squared, divided by sample count.
*/
double SPReg::Prebias(const IndexLevel &index, unsigned int levelIdx) {
  unsigned int sCount;
  double sum;
  index.PrebiasFields(levelIdx, sCount, sum);
  return (sum * sum) / sCount;
}


/**
   @brief As above, but categorical response.  Initializes per-level sum vectors as
wells as FacRun vectors.

   @param levelCount is the number of live index nodes.

   @return vector of unsplitable indices.
*/
void SPCtg::LevelPreset(const IndexLevel &index, std::vector<bool> &unsplitable) {
  LevelInitSumR(pmTrain->NPredNum());
  sumSquares = std::move(std::vector<double>(levelCount));
  ctgSum = std::move(std::vector<double>(levelCount * ctgWidth));
  std::fill(sumSquares.begin(), sumSquares.end(), 0.0);
  std::fill(ctgSum.begin(), ctgSum.end(), 0.0);
  index.SumsAndSquares(ctgWidth, sumSquares, ctgSum, unsplitable);
}


/**
   @brief Gini pre-bias computation for categorical response.

   @param levelIdx is the level-relative node index.

   @param sCount is the number of samples subsumed by the index node.

   @param sum is the sum of samples subsumed by the index node.

   @return sum of squares divided by sum.
 */
double SPCtg::Prebias(const IndexLevel &index, unsigned int levelIdx) {
  unsigned int sCount;
  double sum;
  index.PrebiasFields(levelIdx, sCount, sum);
  return sumSquares[levelIdx] / sum;
}


/**
   @brief Initializes the accumulated-sum checkerboard.

   @return void.
 */
void SPCtg::LevelInitSumR(unsigned int nPredNum) {
  if (nPredNum > 0) {
    ctgSumAccum = std::move(std::vector<double>(nPredNum * ctgWidth * levelCount));
    std::fill(ctgSumAccum.begin(), ctgSumAccum.end(), 0.0);
  }
}


/**
   @brief Determines whether a regression pair undergoes constrained splitting.

   @return The sign of the constraint, if within the splitting probability, else zero.
*/
int SPReg::MonoMode(unsigned int levelIdx, unsigned int predIdx) const {
  if (predMono == 0)
    return 0;

  double monoProb = feMono[predIdx];
  int sign = monoProb > 0.0 ? 1 : (monoProb < 0.0 ? -1 : 0);
  return sign * ruMono[levelIdx] < monoProb ? sign : 0;
}


void SPReg::Split(const IndexLevel &index) {
  ScheduleSplits(index);

  // Guards cast to int for OpenMP 2.0 back-compatibility.
  int splitPos;
#pragma omp parallel default(shared) private(splitPos)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitPos = 0; splitPos < int(splitCoord.size()); splitPos++) {
      splitCoord[splitPos].Split(this, bottom, samplePred, index);
    }
  }

  splitCoord.clear();
}


void SPCtg::Split(const IndexLevel &index) {
  ScheduleSplits(index);
  
  // Guards cast to int for OpenMP 2.0 back-compatibility.
  int splitPos;
#pragma omp parallel default(shared) private(splitPos)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitPos = 0; splitPos < int(splitCoord.size()); splitPos++) {
      splitCoord[splitPos].Split(this, bottom, samplePred, index);
    }
  }
  splitCoord.clear();
}


/**
   @brief  Regression splitting based on type:  numeric or factor.
 */
void SplitCoord::Split(const SPReg *spReg, const Bottom *bottom, const SamplePred *samplePred, const IndexLevel &index) {
  if (spReg->IsFactor(predIdx)) {
    SplitFac(spReg, bottom, samplePred->PredBase(predIdx, bufIdx));
  }
  else {
    SplitNum(spReg, bottom, samplePred->PredBase(predIdx, bufIdx));
  }
}


/**
   @brief Categorical splitting based on type:  numeric or factor.
 */
void SplitCoord::Split(SPCtg *spCtg, const Bottom *bottom, const SamplePred *samplePred, const IndexLevel &index) {
  if (spCtg->IsFactor(predIdx)) {
    SplitFac(spCtg, bottom, samplePred->PredBase(predIdx, bufIdx));
  }
  else {
    SplitNum(spCtg, bottom, samplePred->PredBase(predIdx, bufIdx));
  }
}


void SplitCoord::SplitNum(const SPReg *spReg, const Bottom *bottom, const SPNode spn[]) {
  NuxLH nux;
  if (SplitNum(spReg, spn, nux)) {
    bottom->SSWrite(levelIdx, predIdx, setIdx, bufIdx, nux);
  }
}


/**
   @brief Gini-based splitting method.

   @return void.
*/
void SplitCoord::SplitNum(SPCtg *spCtg, const Bottom *bottom, const SPNode spn[]) {
  NuxLH nux;
  if (SplitNum(spCtg, spn, nux)) {
    bottom->SSWrite(levelIdx, predIdx, setIdx, bufIdx, nux);
  }
}


void SplitCoord::SplitFac(const SPReg *spReg, const Bottom *bottom, const SPNode spn[]) {
  NuxLH nux;
  if (SplitFac(spReg, spn, nux)) {
    bottom->SSWrite(levelIdx, predIdx, setIdx, bufIdx, nux);
  }
}


void SplitCoord::SplitFac(const SPCtg *spCtg, const Bottom *bottom, const SPNode spn[]) {
  NuxLH nux;
  if (SplitFac(spCtg, spn, nux)) {
    bottom->SSWrite(levelIdx, predIdx, setIdx, bufIdx, nux);
  }
}


bool SplitCoord::SplitFac(const SPCtg *spCtg, const SPNode spn[], NuxLH &nux) {
  RunSet *runSet = spCtg->RSet(setIdx);
  RunsCtg(spCtg, runSet, spn);

  if (spCtg->CtgWidth() == 2) {
    return SplitBinary(spCtg, runSet, nux);
  }
  else {
    return SplitRuns(spCtg, runSet, nux);
  }
}


// The four major classes of splitting supported here are based on either
// Gini impurity or weighted variance.  New variants may be supplied in
// future.


/**
   @brief Weighted-variance splitting method.

   @param nux outputs split nucleus.

   @return true iff pair splits.
 */
bool SplitCoord::SplitFac(const SPReg *spReg, const SPNode spn[], NuxLH &nux) {
  RunSet *runSet = spReg->RSet(setIdx);
  RunsReg(runSet, spn, spReg->DenseRank(predIdx));
  runSet->HeapMean();
  runSet->DePop();

  return HeapSplit(runSet, nux);
}


/**
   @brief Invokes regression/numeric splitting method, currently only Gini available.

   @param indexSet[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @return void.
*/
bool SplitCoord::SplitNum(const SPReg *spReg, const SPNode spn[], NuxLH &nux) {
  int monoMode = spReg->MonoMode(splitPos, predIdx);
  if (monoMode != 0) {
    return implicit > 0 ? SplitNumDenseMono(monoMode > 0, spn, spReg, nux) : SplitNumMono(monoMode > 0, spn, nux);
  }
  else {
    return implicit > 0 ? SplitNumDense(spn, spReg, nux) : SplitNum(spn, nux);
  }

}


/**
   @brief Weighted-variance splitting method.

   @return void.
*/
bool SplitCoord::SplitNum(const SPNode spn[], NuxLH &nux) {
  unsigned int rkRight, sampleCount;
  FltVal ySum;
  spn[idxEnd].RegFields(ySum, rkRight, sampleCount);
  double sumR = ySum;
  unsigned int sCountL = sCount - sampleCount; // >= 1: counts up to, including, this index. 
  unsigned int lhSampCt = 0;
  unsigned int lhSup = idxEnd;
  double maxInfo = preBias;

  // Walks samples backward from the end of nodes so that ties are not split.
  // Signing values avoids decrementing below zero.
  for (int i = int(idxEnd) - 1; i >= int(idxStart); i--) {
    unsigned int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    unsigned int rkThis;
    spn[i].RegFields(ySum, rkThis, sampleCount);
    if (idxGini > maxInfo && rkThis != rkRight) {
      lhSampCt = sCountL;
      lhSup = i;
      maxInfo = idxGini;
    }
    sCountL -= sampleCount;
    sumR += ySum;
    rkRight = rkThis;
  }

  if (maxInfo > preBias) {
    nux.InitNum(idxStart, lhSup + 1 - idxStart, lhSampCt, maxInfo - preBias, spn[lhSup].Rank(), spn[lhSup+1].Rank());
    return true;
  }
  else {
    return false;
  }
}


/**
   @brief Experimental.  Needs refactoring.

   @return void.
*/
bool SplitCoord::SplitNumDense(const SPNode spn[], const SPReg *spReg, NuxLH &nux) {
  unsigned int denseRank = spReg->DenseRank(predIdx);
  double sumDense = sum;
  unsigned int sCountDense = sCount;
  unsigned int denseLeft, denseRight;
  unsigned int denseCut = spReg->Residuals(spn, idxStart, idxEnd, denseRank, denseLeft, denseRight, sumDense, sCountDense);

  unsigned int idxNext, idxFinal;
  unsigned int rkRight, sampleCount;
  FltVal ySum;
  if (denseRight) {
    ySum = sumDense;
    rkRight = denseRank;
    sampleCount = sCountDense;
    idxNext = idxEnd;
    idxFinal = idxStart;
  }
  else {
    spn[idxEnd].RegFields(ySum, rkRight, sampleCount);
    idxNext = idxEnd - 1;
    idxFinal = denseLeft ? idxStart : denseCut;
  }
  double sumR = ySum;
  unsigned int sCountL = sCount - sampleCount;
  unsigned int lhSampCt = 0;
  double maxInfo = preBias;

  unsigned int rankLH = 0;
  unsigned int rankRH = 0; // Splitting rank bounds.
  unsigned int rhInf = idxEnd + 1;  // Always non-negative.
  for (int i = int(idxNext); i >= int(idxFinal); i--) {
    unsigned int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    unsigned int rkThis;
    spn[i].RegFields(ySum, rkThis, sampleCount);
    if (idxGini > maxInfo && rkThis != rkRight) {
      lhSampCt = sCountL;
      rankLH = rkThis;
      rankRH = rkRight;
      rhInf = i + 1;
      maxInfo = idxGini;
    }
    sCountL -= sampleCount;
    sumR += ySum;
    rkRight = rkThis;
  }

  // Evaluates the dense component, if not of highest rank.
  if (denseCut != idxEnd) {
    unsigned int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    if (idxGini > maxInfo) {
      lhSampCt = sCountL;
      rhInf = idxFinal;
      rankLH = denseRank;
      rankRH = rkRight;
      maxInfo = idxGini;
    }
  
    if (!denseLeft) { // Walks remaining indices, if any, with rank below dense.
      sCountL -= sCountDense;
      sumR += sumDense;
      rkRight = denseRank;
      for (int i = idxFinal - 1; i >= int(idxStart); i--) {
	unsigned int sCountR = sCount - sCountL;
	double sumL = sum - sumR;
	double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
	unsigned int rkThis;
	spn[i].RegFields(ySum, rkThis, sampleCount);
	if (idxGini > maxInfo && rkThis != rkRight) {
	  lhSampCt = sCountL;
	  rhInf = i + 1;
	  rankLH = rkThis;
	  rankRH = rkRight;
	  maxInfo = idxGini;
	}
	sCountL -= sampleCount;
	sumR += ySum;
	rkRight = rkThis;
      }
    }
  }

  if (maxInfo > preBias) {
    unsigned int lhDense = rankLH >= denseRank ? implicit : 0;
    unsigned int lhIdxTot = rhInf - idxStart + lhDense;
    nux.InitNum(idxStart, lhIdxTot, lhSampCt, maxInfo - preBias, rankLH, rankRH, lhDense);
    return true;
  }
  else {
    return false;
  }
}


/**
   @brief TODO:  Merge with counterparts.

   @return void.
*/
bool SplitCoord::SplitNumDenseMono(bool increasing, const SPNode spn[], const SPReg *spReg, NuxLH &nux) {
  unsigned int denseRank = spReg->DenseRank(predIdx);
  double sumDense = sum;
  unsigned int sCountDense = sCount;
  unsigned int denseLeft, denseRight;
  unsigned int denseCut = spReg->Residuals(spn, idxStart, idxEnd, denseRank, denseLeft, denseRight, sumDense, sCountDense);

  unsigned int idxNext, idxFinal;
  unsigned int rkRight, sampleCount;
  FltVal ySum;
  if (denseRight) {
    ySum = sumDense;
    rkRight = denseRank;
    sampleCount = sCountDense;
    idxNext = idxEnd;
    idxFinal = idxStart;
  }
  else {
    spn[idxEnd].RegFields(ySum, rkRight, sampleCount);
    idxNext = idxEnd - 1;
    idxFinal = denseLeft ? idxStart : denseCut;
  }
  double sumR = ySum;
  unsigned int sCountL = sCount - sampleCount;
  unsigned int lhSampCt = 0;
  double maxInfo = preBias;

  unsigned int rankLH = 0;
  unsigned int rankRH = 0; // Splitting rank bounds.
  unsigned int rhInf = idxEnd + 1;  // Always non-negative.
  for (int i = int(idxNext); i >= int(idxFinal); i--) {
    unsigned int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    unsigned int rkThis;
    spn[i].RegFields(ySum, rkThis, sampleCount);
    if (idxGini > maxInfo && rkThis != rkRight) {
      bool up = (sumL * sCountR <= sumR * sCountL);
      if (increasing ? up : !up) {
        lhSampCt = sCountL;
        rankLH = rkThis;
        rankRH = rkRight;
        rhInf = i + 1;
        maxInfo = idxGini;
      }
    }
    sCountL -= sampleCount;
    sumR += ySum;
    rkRight = rkThis;
  }

  // Evaluates the dense component, if not of highest rank.
  if (denseCut != idxEnd) {
    unsigned int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    if (idxGini > maxInfo) {
      lhSampCt = sCountL;
      rhInf = idxFinal;
      rankLH = denseRank;
      rankRH = rkRight;
      maxInfo = idxGini;
    }
  
    if (!denseLeft) {  // Walks remaining indices, if any, with rank below dense.
      sCountL -= sCountDense;
      sumR += sumDense;
      rkRight = denseRank;
      for (int i = idxFinal - 1; i >= int(idxStart); i--) {
	unsigned int sCountR = sCount - sCountL;
	double sumL = sum - sumR;
	double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
	unsigned int rkThis;
	spn[i].RegFields(ySum, rkThis, sampleCount);
	if (idxGini > maxInfo && rkThis != rkRight) {
	  bool up = (sumL * sCountR <= sumR * sCountL);
	  if (increasing ? up : !up) {
	    lhSampCt = sCountL;
	    rhInf = i + 1;
	    rankLH = rkThis;
	    rankRH = rkRight;
	    maxInfo = idxGini;
	  }
	}
	sCountL -= sampleCount;
	sumR += ySum;
	rkRight = rkThis;
      }
    }
  }

  if (maxInfo > preBias) {
    unsigned int lhDense = rankLH >= denseRank ? implicit : 0;
    unsigned int lhIdxTot = rhInf - idxStart + lhDense;
    nux.InitNum(idxStart, lhIdxTot, lhSampCt, maxInfo - preBias, rankLH, rankRH, lhDense);
    return true;
  }
  else {
    return false;
  }
}


/**
   @brief Weighted-variance splitting method.

   @return void.
*/
bool SplitCoord::SplitNumMono(bool increasing, const SPNode spn[], NuxLH &nux) {
  unsigned int rkRight, sampleCount;
  FltVal ySum;
  spn[idxEnd].RegFields(ySum, rkRight, sampleCount);
  double sumR = ySum;
  unsigned int sCountL = sCount - sampleCount; // >= 1: counts up to, including, this index. 
  unsigned int lhSampCt = 0;
  unsigned int lhSup = idxEnd;
  double maxInfo = preBias;

  // Walks samples backward from the end of nodes so that ties are not split.
  // Signing values avoids decrementing below zero.
  for (int i = int(idxEnd) - 1; i >= int(idxStart); i--) {
    int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    unsigned int rkThis;
    spn[i].RegFields(ySum, rkThis, sampleCount);
    if (idxGini > maxInfo && rkThis != rkRight) {
      bool up = (sumL * sCountR <= sumR * sCountL);
      if (increasing ? up : !up) {
        lhSampCt = sCountL;
        lhSup = i;
        maxInfo = idxGini;
      }
    }
    sCountL -= sampleCount;
    sumR += ySum;
    rkRight = rkThis;
  }
  if (maxInfo > preBias) {
    nux.InitNum(idxStart, lhSup + 1 - idxStart, lhSampCt, maxInfo - preBias, spn[lhSup].Rank(), spn[lhSup + 1].Rank());
    return true;
  }
  else {
    return false;
  }
}


/**
   @brief Imputes dense rank values as residuals.

   @param idxSup outputs the sup of index values having ranks below the
   dense rank.

   @param sumDense inputs the reponse sum over the node and outputs the
   residual sum.

   @param sCount dense input the response sample count over the node and
   outputs the residual count.

   @return supremum of indices to the left ot the dense rank.
*/
unsigned int SPReg::Residuals(const SPNode spn[], unsigned int idxStart, unsigned int idxEnd, unsigned int denseRank, unsigned int &denseLeft, unsigned int &denseRight, double &sumDense, unsigned int &sCountDense) const {
  unsigned int denseCut = idxEnd; // Defaults to highest index.
  double sumTot = 0.0;
  unsigned int sCountTot = 0;
  for (int idx = int(idxEnd); idx >= int(idxStart); idx--) {
    unsigned int sampleCount, rkThis;
    FltVal ySum;
    spn[idx].RegFields(ySum, rkThis, sampleCount);
    denseCut = rkThis > denseRank ? idx : denseCut;
    sCountTot += sampleCount;
    sumTot += ySum;
  }
  sumDense -= sumTot;
  sCountDense -= sCountTot;

  // Dense blob is either left, right or neither.
  denseRight = (denseCut == idxEnd && spn[denseCut].Rank() < denseRank);  
  denseLeft = (denseCut == idxStart && spn[denseCut].Rank() > denseRank);
  
  return denseCut;
}


/**
   @brief Imputes dense rank values as residuals.

   @param idxSup outputs the sup of index values having ranks below the
   dense rank.

   @return true iff left bound has rank less than dense rank.
*/
unsigned int SPCtg::Residuals(const SPNode spn[], unsigned int levelIdx, unsigned int idxStart, unsigned int idxEnd, unsigned int denseRank, bool &denseLeft, bool &denseRight, double &sumDense, unsigned int &sCountDense, std::vector<double> &ctgSumDense) const {
  std::vector<double> ctgAccum;
  ctgSumDense.reserve(ctgWidth);
  ctgAccum.reserve(ctgWidth);
  for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
    ctgSumDense.push_back(CtgSum(levelIdx, ctg));
    ctgAccum.push_back(0.0);
  }
  unsigned int denseCut = idxStart; // Defaults to lowest index.
  double sumTot = 0.0;
  unsigned int sCountTot = 0;
  for (int idx = int(idxEnd); idx >= int(idxStart); idx--) {
    // Accumulates statistics over explicit range.
    unsigned int yCtg, rkThis;
    FltVal ySum;
    unsigned int sampleCount = spn[idx].CtgFields(ySum, rkThis, yCtg);
    ctgAccum[yCtg] += ySum;
    denseCut = rkThis >= denseRank ? idx : denseCut;
    sCountTot += sampleCount;
    sumTot += ySum;
  }
  sumDense -= sumTot;
  sCountDense -= sCountTot;
  for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
    ctgSumDense[ctg] -= ctgAccum[ctg];
  }

  // Dense blob is either left, right or neither.
  denseRight = (denseCut == idxEnd && spn[idxEnd].Rank() < denseRank);  
  denseLeft = (denseCut == idxStart && spn[idxStart].Rank() > denseRank);
  
  return denseCut;
}


void SPCtg::ApplyResiduals(unsigned int levelIdx, unsigned int predIdx, double &ssL, double &ssR, std::vector<double> &sumDenseCtg) {
  unsigned int numIdx = NumIdx(predIdx);
  for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
    double ySum = sumDenseCtg[ctg];
    double sumRCtg = CtgSumAccum(levelIdx, numIdx, ctg, ySum);
    ssR += ySum * (ySum + 2.0 * sumRCtg);
    double sumLCtg = CtgSum(levelIdx, ctg) - sumRCtg;
    ssL += ySum * (ySum - 2.0 * sumLCtg);
  }
}


bool SplitCoord::SplitNum(SPCtg *spCtg, const SPNode spn[], NuxLH &nux) {
  if (implicit > 0) {
    return NumCtgDense(spCtg, spn, nux);
  }
  else {
    return NumCtg(spCtg, spn, nux);
  }
}


bool SplitCoord::NumCtg(SPCtg *spCtg, const SPNode spn[], NuxLH &nux) {
  unsigned int sCountL = sCount;
  unsigned int rkRight = spn[idxEnd].Rank();
  double sumL = sum;
  double ssL = spCtg->SumSquares(levelIdx);
  double ssR = 0.0;
  double maxInfo = preBias;
  unsigned int rankRH = 0;
  unsigned int rankLH = 0;
  unsigned int rhInf = idxEnd;
  unsigned int lhSampCt = 0;
  lhSampCt = NumCtgGini(spCtg, spn, idxEnd, idxStart, sCountL, rkRight, sumL, ssL, ssR, maxInfo, rankLH, rankRH, rhInf);

  if (maxInfo > preBias) {
    unsigned int lhIdxTot = rhInf - idxStart;
    nux.InitNum(idxStart, lhIdxTot, lhSampCt, maxInfo - preBias, rankLH, rankRH, 0);
    return true;
  }
  else {
    return false;
  }
}


unsigned int SplitCoord::NumCtgGini(SPCtg *spCtg, const SPNode spn[], unsigned int idxNext, unsigned int idxFinal, unsigned int &sCountL, unsigned int &rkRight, double &sumL, double &ssL, double &ssR, double &maxGini, unsigned int &rankLH, unsigned int &rankRH, unsigned int &rhInf) {
  unsigned int lhSampCt = 0;
  unsigned int numIdx = spCtg->NumIdx(predIdx);
  // Signing values avoids decrementing below zero.
  for (int idx = int(idxNext); idx >= int(idxFinal); idx--) {
    FltVal ySum;    
    unsigned int yCtg, rkThis;
    unsigned int sampleCount = spn[idx].CtgFields(ySum, rkThis, yCtg);
    FltVal sumR = sum - sumL;
    if (rkThis != rkRight && spCtg->StableDenoms(sumL, sumR)) {
      FltVal cutGini = ssL / sumL + ssR / sumR;
      if (cutGini > maxGini) {
        lhSampCt = sCountL;
	rankLH = rkThis;
	rankRH = rkRight;
	rhInf = idx + 1;
        maxGini = cutGini;
      }
    }
    rkRight = rkThis;

    sCountL -= sampleCount;
    sumL -= ySum;

    double sumRCtg = spCtg->CtgSumAccum(levelIdx, numIdx, yCtg, ySum);
    ssR += ySum * (ySum + 2.0 * sumRCtg);
    double sumLCtg = spCtg->CtgSum(levelIdx, yCtg) - sumRCtg;
    ssL += ySum * (ySum - 2.0 * sumLCtg);
  }

  return lhSampCt;
}


bool SplitCoord::NumCtgDense(SPCtg *spCtg, const SPNode spn[], NuxLH &nux) {
  unsigned int denseRank = spCtg->DenseRank(predIdx);
  double sumDense = sum;
  unsigned int sCountDense = sCount;
  bool denseLeft, denseRight;
  std::vector<double> sumDenseCtg;
  unsigned int denseCut = spCtg->Residuals(spn, levelIdx, idxStart, idxEnd, denseRank, denseLeft, denseRight, sumDense, sCountDense, sumDenseCtg);

  unsigned int idxFinal;
  unsigned int sCountL = sCount;
  unsigned int rkRight;
  double sumL = sum;
  double ssL = spCtg->SumSquares(levelIdx);
  double ssR = 0.0;
  if (denseRight) { // Implicit values to the far right.
    idxFinal = idxStart;
    rkRight = denseRank;
    spCtg->ApplyResiduals(levelIdx, predIdx, ssL, ssR, sumDenseCtg);
    sCountL -= sCountDense;
    sumL -= sumDense;
  }
  else {
    idxFinal = denseLeft ? idxStart : denseCut + 1;
    rkRight = spn[idxEnd].Rank();
  }

  double maxInfo = preBias;
  unsigned int rankRH = 0;
  unsigned int rankLH = 0;
  unsigned int rhInf = idxEnd;
  unsigned int lhSampCt = NumCtgGini(spCtg, spn, idxEnd, idxFinal, sCountL, rkRight, sumL, ssL, ssR, maxInfo, rankLH, rankRH, rhInf);

  // Evaluates the dense component, if not of highest rank.
  if (denseCut != idxEnd) {
    FltVal sumR = sum - sumL;
    if (spCtg->StableDenoms(sumL, sumR)) {
      FltVal cutGini = ssL / sumL + ssR / sumR;
      if (cutGini >  maxInfo) {
	lhSampCt = sCountL;
	rhInf = idxFinal;
	rankLH = denseRank;
	rankRH = rkRight;
	maxInfo = cutGini;
      }
    }

    if (!denseLeft) {  // Walks remaining indices, if any with ranks below dense.
      spCtg->ApplyResiduals(levelIdx, predIdx, ssR, ssL, sumDenseCtg);
      sCountL -= sCountDense;
      sumL -= sumDense;
      lhSampCt = NumCtgGini(spCtg, spn, denseCut, idxStart, sCountL, rkRight, sumL, ssL, ssR, maxInfo, rankLH, rankRH, rhInf);
    }
  }

  if (maxInfo > preBias) {
    unsigned int lhDense = rankLH >= denseRank ? implicit : 0;
    unsigned int lhIdxTot = rhInf - idxStart + lhDense;
    nux.InitNum(idxStart, lhIdxTot, lhSampCt, maxInfo - preBias, rankLH, rankRH, lhDense);
    return true;
  }
  else {
    return false;
  }
}


/**
   Regression runs always maintained by heap.
*/
void SplitCoord::RunsReg(RunSet *runSet, const SPNode spn[], unsigned int denseRank) const {
  double sumHeap = 0.0;
  unsigned int sCountHeap = 0;
  unsigned int rkThis = spn[idxEnd].Rank();
  unsigned int frEnd = idxEnd;

  // Signing values avoids decrementing below zero.
  //
  for (int i = int(idxEnd); i >= int(idxStart); i--) {
    unsigned int rkRight = rkThis;
    unsigned int sampleCount;
    FltVal ySum;
    spn[i].RegFields(ySum, rkThis, sampleCount);

    if (rkThis == rkRight) { // Same run:  counters accumulate.
      sumHeap += ySum;
      sCountHeap += sampleCount;
    }
    else { // New run:  flush accumulated counters and reset.
      runSet->Write(rkRight, sCountHeap, sumHeap, frEnd - i, i+1);

      sumHeap = ySum;
      sCountHeap = sampleCount;
      frEnd = i;
    }
  }
  
  // Flushes the remaining run.  Also flushes the implicit run, if dense.
  //
  runSet->Write(rkThis, sCountHeap, sumHeap, frEnd - idxStart + 1, idxStart);
  if (implicit > 0) {
    runSet->WriteImplicit(denseRank, sCount, sum, implicit);
  }
}


/**
   @brief Splits runs sorted by binary heap.

   @param runSet contains all run parameters.

   @param outputs computed split parameters.

   @return true iff node splits.
*/
bool SplitCoord::HeapSplit(RunSet *runSet, NuxLH &nux) const {
  unsigned int lhSCount = 0;
  double sumL = 0.0;
  int cut = -1; // Top index of lh ords in 'facOrd' (q.v.).
  double maxGini = preBias;
  for (unsigned int outSlot = 0; outSlot < runSet->RunCount() - 1; outSlot++) {
    unsigned int sCountRun;
    sumL += runSet->SumHeap(outSlot, sCountRun);
    lhSCount += sCountRun;
    unsigned int sCountR = sCount - lhSCount;
    double sumR = sum - sumL;
    double cutGini = (sumL * sumL) / lhSCount + (sumR * sumR) / sCountR;
    if (cutGini > maxGini) {
      maxGini = cutGini;
      cut = outSlot;
    }
  }

  if (cut >= 0) {
    unsigned int lhIdxCount = runSet->LHSlots(cut, lhSCount);
    nux.Init(idxStart, lhIdxCount, lhSCount, maxGini - preBias);
    return true;
  }
  else {
    return false;
  }
}


/**
   @brief Builds categorical runs.  Very similar to regression case, but the runs
   also resolve response sum by category.  Further, heap is optional, passed only
   when run count has been estimated to be wide:

*/
void SplitCoord::RunsCtg(const SPCtg *spCtg, RunSet *runSet, const SPNode spn[]) const {
  double sumLoc = 0.0;
  unsigned int sCountLoc = 0;
  unsigned int rkThis = spn[idxEnd].Rank();

  // Signing values avoids decrementing below zero.
  unsigned int frEnd = idxEnd;
  for (int i = int(idxEnd); i >= int(idxStart); i--) {
    unsigned int rkRight = rkThis;
    unsigned int yCtg;
    FltVal ySum;
    unsigned int sampleCount = spn[i].CtgFields(ySum, rkThis, yCtg);

    if (rkThis == rkRight) { // Current run's counters accumulate.
      sumLoc += ySum;
      sCountLoc += sampleCount;
    }
    else { // Flushes current run and resets counters for next run.
      runSet->Write(rkRight, sCountLoc, sumLoc, frEnd - i, i + 1);

      sumLoc = ySum;
      sCountLoc = sampleCount;
      frEnd = i;
    }
    runSet->AccumCtg(yCtg, ySum);
  }

  
  // Flushes remaining run.
  runSet->Write(rkThis, sCountLoc, sumLoc, frEnd - idxStart + 1, idxStart);
  if (implicit > 0) {
    runSet->WriteImplicit(spCtg->DenseRank(predIdx), sCount, sum, implicit, spCtg->ColumnSums(levelIdx));
  }
}


/**
   @brief Splits blocks of categorical runs.

   @param sum is the sum of response values for this index node.

   @param maxGini outputs the highest observed Gini value.
   
   @param lhSampCt outputs LHS sample count.

   @return index count of LHS, with output reference parameters.

   Nodes are now represented compactly as a collection of runs.
   For each node, subsets of these collections are examined, looking for the
   Gini argmax beginning from the pre-bias.

   Iterates over nontrivial subsets, coded by integers as bit patterns.  By
   convention, the final run is incorporated into the RHS of the split, if any.
   Excluding the final run, then, the number of candidate LHS subsets is
   '2^(runCount-1) - 1'.
*/
bool SplitCoord::SplitRuns(const SPCtg *spCtg, RunSet *runSet, NuxLH &nux) {
  unsigned int countEff = runSet->DeWide();

  unsigned int slotSup = countEff - 1; // Uses post-shrink value.
  unsigned int lhBits = 0;
  unsigned int leftFull = (1 << slotSup) - 1;
  double maxGini = preBias;
  // Nonempty subsets as binary-encoded integers:
  for (unsigned int subset = 1; subset <= leftFull; subset++) {
    double sumL = 0.0;
    double ssL = 0.0;
    double ssR = 0.0;
    for (unsigned int yCtg = 0; yCtg < spCtg->CtgWidth(); yCtg++) {
      double sumCtg = 0.0; // Sum at this category over subset slots.
      for (unsigned int slot = 0; slot < slotSup; slot++) {
	if ((subset & (1 << slot)) != 0) {
	  sumCtg += runSet->SumCtg(slot, yCtg);
	}
      }
      double totSum = spCtg->CtgSum(levelIdx, yCtg); // Sum at this category over node.
      sumL += sumCtg;
      ssL += sumCtg * sumCtg;
      ssR += (totSum - sumCtg) * (totSum - sumCtg);
    }
    double sumR = sum - sumL;
    // Only relevant for case weighting:  otherwise sums are >= 1.
    if (spCtg->StableSums(sumL, sumR)) {
      double subsetGini = ssR / sumR + ssL / sumL;
      if (subsetGini > maxGini) {
        maxGini = subsetGini;
        lhBits = subset;
      }
    }
  }

  if (lhBits > 0) {
    unsigned int lhSampCt;
    unsigned int lhIdxCount = runSet->LHBits(lhBits, lhSampCt);
    nux.Init(idxStart, lhIdxCount, lhSampCt, maxGini - preBias);
    return true;
  }
  else {
    return false;
  }
}


/**
   @brief Adapated from SplitRuns().  Specialized for two-category case in
   which LH subsets accumulate.  This permits running LH 0/1 sums to be
   maintained, as opposed to recomputed, as the LH set grows.

   @return 
 */
bool SplitCoord::SplitBinary(const SPCtg *spCtg, RunSet *runSet, NuxLH &nux) {
  runSet->HeapBinary();
  runSet->DePop();

  double maxGini = preBias;
  double totR0 = spCtg->CtgSum(levelIdx, 0); // Sum at this category over node.
  double totR1 = spCtg->CtgSum(levelIdx, 1);
  double sumL0 = 0.0; // Running sum at category 0 over subset slots.
  double sumL1 = 0.0; // "" 1 " 
  int cut = -1;
  for (unsigned int outSlot = 0; outSlot < runSet->RunCount() - 1; outSlot++) {
    double cell0, cell1;
    bool splitable = runSet->SumBinary(outSlot, cell0, cell1);

    sumL0 += cell0;
    sumL1 += cell1;

    FltVal sumL = sumL0 + sumL1;
    FltVal sumR = sum - sumL;
    // sumR, sumL magnitudes can be ignored if no large case/class weightings.
    if (splitable && spCtg->StableDenoms(sumL, sumR)) {
      FltVal ssL = sumL0 * sumL0 + sumL1 * sumL1;
      FltVal ssR = (totR0 - sumL0) * (totR0 - sumL0) + (totR1 - sumL1) * (totR1 - sumL1);
      FltVal cutGini = ssR / sumR + ssL / sumL;
       if (cutGini > maxGini) {
        maxGini = cutGini;
        cut = outSlot;
      }
    } 
  }

  if (cut >= 0) {
    unsigned int sCountL;
    unsigned int lhIdxCount = runSet->LHSlots(cut, sCountL);
    nux.Init(idxStart, lhIdxCount, sCountL, maxGini - preBias);
    return true;
  }
  else {
    return false;
  }
}
