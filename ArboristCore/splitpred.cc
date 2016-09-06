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
using namespace std;

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
SplitPred::SplitPred(const RowRank *_rowRank, SamplePred *_samplePred, unsigned int _bagCount) : rowRank(_rowRank), bagCount(_bagCount), samplePred(_samplePred) {
}


/**
   @brief Destructor.  Deletes dangling 'runFlags' vector, which should be
   nonempty.
 */
SplitPred::~SplitPred() {
  delete run;
}


void SplitPred::Immutables(unsigned int _nPred, unsigned int _ctgWidth, unsigned int _predFixed, const double _predProb[], const double _regMono[]) {
  nPred = _nPred;
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
SPReg::SPReg(const RowRank *_rowRank, SamplePred *_samplePred, unsigned int _bagCount) : SplitPred(_rowRank, _samplePred, _bagCount), ruMono(0) {
  run = new Run(0, PBTrain::NRow(), _bagCount);
}


/**
   @brief Constructor.

   @param samplePred holds (re)staged node contents.

   @param sampleCtg is the sample vector for the tree, included for category lookup.
 */
SPCtg::SPCtg(const RowRank *_rowRank, SamplePred *_samplePred, const std::vector<SampleNode> &_sampleCtg, unsigned int _bagCount): SplitPred(_rowRank, _samplePred, _bagCount), sampleCtg(_sampleCtg) {
  run = new Run(ctgWidth, PBTrain::NRow(), _bagCount);
}


/**
   @brief Sets per-level state enabling pre-bias computation.

   @param index is the Index context.

   @param levelCount is the potential number of splitting nodes in the upcoming level.

   @return split count.
*/
void SplitPred::LevelInit(Index *index, IndexNode indexNode[], unsigned int _levelCount) {
  levelCount = _levelCount;
  std::vector<unsigned int> safeCount;
  bool *unsplitable = LevelPreset(index);
  Splitable(unsplitable, safeCount);
  delete [] unsplitable;

  SetPrebias(indexNode); // Depends on state from LevelPreset()
  RunOffsets(safeCount);
}


/**
   @brief Sets (Gini) pre-bias value according to response type.

   @param indexNode is the index tree vector for the current level.

   @return void.
*/
void SplitPred::SetPrebias(IndexNode indexNode[]) {
  for (unsigned int levelIdx = 0; levelIdx < levelCount; levelIdx++) {
    IndexNode *idxNode = &indexNode[levelIdx];
    unsigned int sCount;
    double sum;
    idxNode->PrebiasFields(sCount, sum);
    idxNode->Prebias() = Prebias(levelIdx, sCount, sum);
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
void SPReg::LevelInit(Index *index, IndexNode indexNode[], unsigned int _levelCount) {
  SplitPred::LevelInit(index, indexNode, _levelCount);
  if (predMono > 0) {
    unsigned int monoCount = _levelCount * nPred; // Clearly too big.
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
void SPReg::RunOffsets(const std::vector<unsigned int> &safeCount) {
  run->RunSets(safeCount);
  run->OffsetsReg();
}


/**
   @brief Sets quick lookup offsets for Run object.
 */
void SPCtg::RunOffsets(const std::vector<unsigned int> &safeCount) {
  run->RunSets(safeCount);
  run->OffsetsCtg();
}


/**
   @brief Signals Bottom to schedule splitable pairs.

   @param unsplitable lists unsplitable nodes.

   @return void.
*/
void SplitPred::Splitable(const bool unsplitable[], std::vector<unsigned int> &safeCount) {
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
      ScheduleProb(levelIdx, &ruPred[splitOff], safeCount);
    }
    else { // Fixed number of predictors splitable.
      ScheduleFixed(levelIdx, &ruPred[splitOff], &heap[splitOff], safeCount);
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
void SplitPred::ScheduleProb(unsigned int levelIdx, const double ruPred[], std::vector<unsigned int> &safeCount) {
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    if (ruPred[predIdx] < predProb[predIdx]) {
      (void) ScheduleSplit(levelIdx, predIdx, safeCount);
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
void SplitPred::ScheduleFixed(unsigned int levelIdx, const double ruPred[], BHPair heap[], std::vector<unsigned int> &safeCount) {
  // Inserts negative, weighted probability value:  choose from lowest.
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    BHeap::Insert(heap, predIdx, -ruPred[predIdx] * predProb[predIdx]);
  }

  // Pops 'predFixed' items in order of increasing value.
  unsigned int schedCount = 0;
  for (unsigned int heapSize = nPred; heapSize > 0; heapSize--) {
    unsigned int predIdx = BHeap::SlotPop(heap, heapSize - 1);
    schedCount += ScheduleSplit(levelIdx, predIdx, safeCount) ? 1 : 0;
    if (schedCount == predFixed)
      break;
  }
}


bool SplitPred::ScheduleSplit(unsigned int levelIdx, unsigned int predIdx, std::vector<unsigned int> &safeCount) {
  unsigned int runCount, bufIdx;
  if (bottom->ScheduleSplit(levelIdx, predIdx, runCount, bufIdx)) {
    SplitCoord sg;
    if (runCount > 0) {
      sg.InitEarly(splitCoord.size(), levelIdx, predIdx, bufIdx, safeCount.size());
      safeCount.push_back(runCount);
    }
    else {
      sg.InitEarly(splitCoord.size(), levelIdx, predIdx, bufIdx, run->NoRun());
    }
    splitCoord.push_back(sg);
    return true;
  }
  else {
    return false;
  }
}


/**
   @brief Base method.  Deletes per-level run and split-flags vectors.

   @return void.
 */
void SplitPred::LevelClear() {
  run->LevelClear();
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
  if (PredBlock::NPredNum() > 0) {
    delete [] ctgSumR;
  }
  delete [] ctgSum;
  delete [] sumSquares;
  ctgSum = sumSquares = ctgSumR = 0;
  SplitPred::LevelClear();
}


/**
   @brief Currently just a stub.

   @param index is not used by this instantiation.

   @param levelCount is the number of live index nodes.

   @return vector of unsplitable indices.
*/
bool *SPReg::LevelPreset(const Index *index) {
  bool* unsplitable = new bool[levelCount];
  for (unsigned int levelIdx = 0; levelIdx < levelCount; levelIdx++)
    unsplitable[levelIdx] = false;

  return unsplitable;
}


/**
  @brief Weight-variance pre-bias computation for regression response.

  @param levelIdx is the level-relative node index.

  @param sCount is the number of samples subsumed by the index node.

  @param sum is the sum of samples subsumed by the index node.

  @return square squared, divided by sample count.
*/
double SPReg::Prebias(unsigned int levelIdx, unsigned int sCount, double sum) {
  return (sum * sum) / sCount;
}


/**
   @brief As above, but categorical response.  Initializes per-level sum vectors as
wells as FacRun vectors.

   @param levelCount is the number of live index nodes.

   @return vector of unsplitable indices.
*/
bool *SPCtg::LevelPreset(const Index *index) {
  if (PredBlock::NPredNum() > 0)
    LevelInitSumR();

  bool *unsplitable = new bool[levelCount];
  for (unsigned int levelIdx = 0; levelIdx < levelCount; levelIdx++)
    unsplitable[levelIdx] = false;
  SumsAndSquares(index, unsplitable);

  return unsplitable;
}


/**
 */
void SPCtg::SumsAndSquares(const Index *index, bool unsplitable[]) {
  sumSquares = new double[levelCount];
  ctgSum = new double[levelCount * ctgWidth];
  unsigned int levelWidth = index->LevelWidth();
  double *sumTemp = new double[levelWidth * ctgWidth];
  unsigned int *sCountTemp = new unsigned int[levelWidth * ctgWidth];
  for (unsigned int i = 0; i < levelWidth * ctgWidth; i++) {
    sumTemp[i] = 0.0;
    sCountTemp[i] = 0;
  }

  // Sums each category for each node in the upcoming level, including
  // leaves.  Since these appear in arbitrary order, a second pass copies
  // those columns corresponding to nonterminals in split-index order, for
  // ready access by splitting methods.
  //
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    unsigned int levelOff;
    bool atLevel = index->LevelOffSample(sIdx, levelOff);
    if (atLevel) {
      FltVal sum;
      unsigned int sCount;
      unsigned int ctg = sampleCtg[sIdx].Ref(sum, sCount);
      sumTemp[levelOff * ctgWidth + ctg] += sum;
      sCountTemp[levelOff * ctgWidth + ctg] += sCount;
    }
  }

  // Reorders by split index, omitting any intervening leaf sums.  Could
  // instead index directly by level offset, but this would require more
  // complex accessor methods.
  //
  for (unsigned int levelIdx = 0; levelIdx < levelCount; levelIdx++) {
    int levelOff = index->LevelOffSplit(levelIdx);
    unsigned int indexSCount = index->SCount(levelIdx);
    double ss = 0.0;
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      unsigned int sCount = sCountTemp[levelOff * ctgWidth + ctg];
      if (sCount == indexSCount) { // Singleton response:  avoid splitting.
	unsplitable[levelIdx] = true;
      }
      double sum = sumTemp[levelOff * ctgWidth + ctg];
      ctgSum[levelIdx * ctgWidth + ctg] = sum;
      ss += sum * sum;
    }
    sumSquares[levelIdx] = ss;
  }

  delete [] sumTemp;
  delete [] sCountTemp;
}


/**
   @brief Gini pre-bias computation for categorical response.

   @param levelIdx is the level-relative node index.

   @param sCount is the number of samples subsumed by the index node.

   @param sum is the sum of samples subsumed by the index node.

   @return sum of squares divided by sum.
 */
double SPCtg::Prebias(unsigned int levelIdx, unsigned int sCount, double sum) {
  return sumSquares[levelIdx] / sum;
}


/**
   @brief Initializes the accumulated-sum checkerboard.

   @return void.
 */
void SPCtg::LevelInitSumR() {
  unsigned int length = PredBlock::NPredNum() * ctgWidth * levelCount;
  ctgSumR = new double[length];
  for (unsigned int i = 0; i < length; i++)
    ctgSumR[i] = 0.0;
}


/**
   @brief Determines whether a regression pair undergoes constrained splitting.

   @return The sign of the constraint, if within the splitting probability, else zero.
*/
int SPReg::MonoMode(unsigned int splitIdx, unsigned int predIdx) const {
  if (predMono == 0)
    return 0;

  double monoProb = feMono[predIdx];
  int sign = monoProb > 0.0 ? 1 : (monoProb < 0.0 ? -1 : 0);
  return sign * ruMono[splitIdx] < monoProb ? sign : 0;
}


/**
 @brief Sets those fields known before restaging.
  isDense set following restage.
 runCount may be reset to unity following restage.

 @return void.
*/
void SplitCoord::InitEarly(unsigned int _splitPos, unsigned int _levelIdx, unsigned int _predIdx, unsigned int _bufIdx, unsigned int _setIdx) {
  splitPos = _splitPos;
  levelIdx = _levelIdx;
  predIdx = _predIdx;
  bufIdx = _bufIdx;
  setIdx = _setIdx;
}


void SplitCoord::InitLate(const Bottom *bottom, const IndexNode indexNode[]) {
  unsigned int idxCount;
  preBias = indexNode[levelIdx].SplitFields(idxStart, idxCount, sCount, sum);
  denseCount = bottom->DenseCount(levelIdx, predIdx);
  idxEnd = idxStart + idxCount - (1 + denseCount);
}

 
void SPReg::Split(const IndexNode indexNode[]) {
  // Guards cast to int for OpenMP 2.0 back-compatibility.
  int splitPos;
#pragma omp parallel default(shared) private(splitPos)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitPos = 0; splitPos < int(splitCoord.size()); splitPos++) {
      splitCoord[splitPos].Split(this, bottom, samplePred, indexNode);
    }
  }

  splitCoord.clear();
}


void SPCtg::Split(const IndexNode indexNode[]) {
  // Guards cast to int for OpenMP 2.0 back-compatibility.
  int splitPos;
#pragma omp parallel default(shared) private(splitPos)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitPos = 0; splitPos < int(splitCoord.size()); splitPos++) {
      splitCoord[splitPos].Split(this, bottom, samplePred, indexNode);
    }
  }
  splitCoord.clear();
}


/**
   @brief  Regression splitting based on type:  numeric or factor.
 */
void SplitCoord::Split(const SPReg *spReg, const Bottom *bottom, const SamplePred *samplePred, const IndexNode indexNode[]) {
  // Restaging may precipitate new singletons after scheduling.
  //
  if (bottom->Singleton(levelIdx, predIdx))
    return;

  InitLate(bottom, indexNode);
  if (PBTrain::IsFactor(predIdx)) {
    SplitFac(spReg, bottom, samplePred->PredBase(predIdx, bufIdx));
  }
  else {
    SplitNum(spReg, bottom, samplePred->PredBase(predIdx, bufIdx));
  }
}


/**
   @brief Categorical splitting based on type:  numeric or factor.
 */
void SplitCoord::Split(SPCtg *spCtg, const Bottom *bottom, const SamplePred *samplePred, const IndexNode indexNode[]) {
  // Restaging may precipitate new singletons after scheduling.
  //
  if (bottom->Singleton(levelIdx, predIdx))
    return;

  InitLate(bottom, indexNode);
  if (PBTrain::IsFactor(predIdx)) {
    SplitFac(spCtg, bottom, samplePred->PredBase(predIdx, bufIdx));
  }
  else {
    SplitNum(spCtg, bottom, samplePred->PredBase(predIdx, bufIdx));
  }
}


void SplitCoord::SplitNum(const SPReg *spReg, const Bottom *bottom, const SPNode spn[]) {
  SplitNux nux;
  if (SplitNum(spReg, spn, nux)) {
    bottom->SSWrite(levelIdx, predIdx, setIdx, bufIdx, nux);
  }
}


/**
   @brief Gini-based splitting method.

   @return void.
*/
void SplitCoord::SplitNum(SPCtg *spCtg, const Bottom *bottom, const SPNode spn[]) {
  SplitNux nux;
  if (SplitNum(spCtg, spn, nux)) {
    bottom->SSWrite(levelIdx, predIdx, setIdx, bufIdx, nux);
  }
}


void SplitCoord::SplitFac(const SPReg *spReg, const Bottom *bottom, const SPNode spn[]) {
  SplitNux nux;
  unsigned int runCount;
  if (SplitFac(spReg, spn, runCount, nux)) {
    bottom->SSWrite(levelIdx, predIdx, setIdx, bufIdx, nux);
  }
  bottom->SetRunCount(levelIdx, predIdx, runCount);
}


void SplitCoord::SplitFac(const SPCtg *spCtg, const Bottom *bottom, const SPNode spn[]) {
  SplitNux nux;
  unsigned int runCount;
  if (SplitFac(spCtg, spn, runCount, nux)) {
    bottom->SSWrite(levelIdx, predIdx, setIdx, bufIdx, nux);
  }
  bottom->SetRunCount(levelIdx, predIdx, runCount);
}


bool SplitCoord::SplitFac(const SPCtg *spCtg, const SPNode spn[], unsigned int &runCount, SplitNux &nux) {
  RunSet *runSet = spCtg->RSet(setIdx);
  runCount = RunsCtg(runSet, spn, spCtg->DenseRank(setIdx));
  if (spCtg->CtgWidth() == 2) {
    return SplitBinary(runSet, spCtg, nux);
  }
  else {
    return SplitRuns(runSet, spCtg, nux);
  }
}


// The four major classes of splitting supported here are based on either
// Gini impurity or weighted variance.  New variants may be supplied in
// future.


/**
   @brief Weighted-variance splitting method.

   @param runCount outputs recently-updated run count.

   @param nux outputs split nucleus.

   @return true iff pair splits.
 */
bool SplitCoord::SplitFac(const SPReg *spReg, const SPNode spn[], unsigned int &runCount, SplitNux &nux) {
  RunSet *runSet = spReg->RSet(setIdx);
  runCount = RunsReg(runSet, spn, spReg->DenseRank(predIdx));
  runSet->HeapMean();
  runSet->DePop();

  return HeapSplit(runSet, nux);
}


/**
   @brief Invokes regression/numeric splitting method, currently only Gini available.

   @param indexNode[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @return void.
*/
bool SplitCoord::SplitNum(const SPReg *spReg, const SPNode spn[], SplitNux &nux) {
  int monoMode = spReg->MonoMode(splitPos, predIdx);
  if (monoMode != 0) {
    return SplitNumMono(monoMode > 0, spn, nux);
  }
  else {
    return SplitNum(spn, nux);
  }

}


/**
   @brief Weighted-variance splitting method.

   @return void.
*/
bool SplitCoord::SplitNum(const SPNode spn[], SplitNux &nux) {
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
  nux.Init(lhSup + 1 - idxStart, lhSampCt, maxInfo - preBias);

  return lhSup < idxEnd;
}


/**
   @brief Weighted-variance splitting method.

   @return void.
*/
bool SplitCoord::SplitNumMono(bool increasing, const SPNode spn[], SplitNux &nux) {
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
      FltVal meanL = sumL / sCountL;
      FltVal meanR = sumR / sCountR;
      bool doSplit = increasing ? meanL <= meanR : meanL >= meanR;
      if (doSplit) {
        lhSampCt = sCountL;
        lhSup = i;
        maxInfo = idxGini;
      }
    }
    sCountL -= sampleCount;
    sumR += ySum;
    rkRight = rkThis;
  }
  nux.Init(lhSup + 1 - idxStart, lhSampCt, maxInfo - preBias);

  return lhSup < idxEnd;
}


bool SplitCoord::SplitNum(SPCtg *spCtg, const SPNode spn[], SplitNux &nux) {
  int numIdx = PredBlock::NumIdx(predIdx);
  unsigned int sCountL = sCount;
  double sumL = sum;
  double maxGini = preBias;

  double ssL = spCtg->SumSquares(levelIdx);
  double ssR = 0.0;
  unsigned int rkRight = spn[idxEnd].Rank();
  unsigned int rkStart = spn[idxStart].Rank();
  unsigned int lhSampCt = 0;

  // Signing values avoids decrementing below zero.
  unsigned int lhSup = idxEnd;
  for (int i = int(idxEnd); i >= int(idxStart); i--) {
    unsigned int rkThis = spn[i].Rank();
    FltVal sumR = sum - sumL;
    if (rkThis != rkRight && sumL > SPCtg::minDenom && sumR > SPCtg::minDenom) {
      FltVal cutGini = ssL / sumL + ssR / sumR;
      if (cutGini > maxGini) {
        lhSampCt = sCountL;
        lhSup = i;
        maxGini = cutGini;
      }
    }
    if (rkRight == rkStart) // Last valid cut already checked.
      break;

    unsigned int yCtg;
    FltVal ySum;    
    sCountL -= spn[i].CtgFields(ySum, yCtg);

    // Maintains sums of category squares incrementally, via update.
    //
    // Right sum is post-incremented with 'ySum', hence is exclusive.
    // Left sum is inclusive.
    //
    double sumRCtg = spCtg->CtgSumRight(levelIdx, numIdx, yCtg, ySum);
    double sumLCtg = spCtg->CtgSum(levelIdx, yCtg) - sumRCtg;
    ssR += ySum * (ySum + 2.0 * sumRCtg);
    ssL += ySum * (ySum - 2.0 * sumLCtg);
    sumL -= ySum;
    rkRight = rkThis;
  }

  if (lhSup < idxEnd) {
    nux.Init(lhSup + 1 - idxStart, lhSampCt, maxGini - preBias);
    return true;
  }
  else {
    return false;
  }
}


/**
   Regression runs always maintained by heap.
*/
unsigned int SplitCoord::RunsReg(RunSet *runSet, const SPNode spn[], unsigned int denseRank) const {
  double sumHeap = 0.0;
  unsigned int sCountHeap = 0;
  unsigned int rkThis = spn[idxEnd].Rank();

  // Signing values avoids decrementing below zero.
  unsigned int frEnd = idxEnd;
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
      runSet->Write(rkRight, sCountHeap, sumHeap, i+1, frEnd);

      sumHeap = ySum;
      sCountHeap = sampleCount;
      frEnd = i;
    }
  }
  
  // Flushes the remaining run.  Also flushes the implicit run, if dense.
  //
  runSet->Write(rkThis, sCountHeap, sumHeap, idxStart, frEnd);
  if (denseCount > 0) {
    runSet->ImplicitRun(denseRank, sCount, sum);
  }

  return runSet->RunCount();
}


/**
   @brief Splits runs sorted by binary heap.

   @param runSet contains all run parameters.

   @param outputs computed split parameters.

   @return true iff node splits.
*/
bool SplitCoord::HeapSplit(RunSet *runSet, SplitNux &nux) const {
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
    nux.Init(lhIdxCount, lhSCount, maxGini - preBias);
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
unsigned int SplitCoord::RunsCtg(RunSet *runSet, const SPNode spn[], unsigned int denseRank) const {
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
      runSet->Write(rkRight, sCountLoc, sumLoc, i+1, frEnd);

      sumLoc = ySum;
      sCountLoc = sampleCount;
      frEnd = i;
    }
    runSet->SumCtg(yCtg) += ySum;
  }

  // Flushes remaining run.
  runSet->Write(rkThis, sCountLoc, sumLoc, idxStart, frEnd);
  if (denseCount > 0) {
    runSet->ImplicitRun(denseRank, sCount, sum);
  }

  return runSet->RunCount();
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
bool SplitCoord::SplitRuns(RunSet *runSet, const SPCtg *spCtg, SplitNux &nux) {
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
    if (sumL > SPCtg::minSumL && sumR > SPCtg::minSumR) {
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
    nux.Init(lhIdxCount, lhSampCt, maxGini - preBias);
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
bool SplitCoord::SplitBinary(RunSet *runSet, const SPCtg *spCtg, SplitNux &nux) {
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
    if (splitable && sumL > SPCtg::minDenom && sumR > SPCtg::minDenom) {
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
    nux.Init(lhIdxCount, sCountL, maxGini - preBias);
    return true;
  }
  else {
    return false;
  }
}
