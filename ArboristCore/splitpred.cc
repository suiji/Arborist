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

unsigned int SplitPred::nPred = 0;
unsigned int SplitPred::predFixed = 0;
const double *SplitPred::predProb = 0;

const double *SPReg::feMono = 0;
unsigned int SPReg::predMono = 0;
unsigned int SPCtg::ctgWidth = 0;

/**
  @brief Constructor.  Initializes 'runFlags' to zero for the single-split root.
 */
SplitPred::SplitPred(SamplePred *_samplePred, unsigned int bagCount) : samplePred(_samplePred) {
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
SPReg::SPReg(SamplePred *_samplePred, unsigned int bagCount) : SplitPred(_samplePred, bagCount), ruMono(0) {
  run = new Run(0);
}


/**
   @brief Constructor.

   @param samplePred holds (re)staged node contents.

   @param sampleCtg is the sample vector for the tree, included for category lookup.
 */
SPCtg::SPCtg(SamplePred *_samplePred, SampleNode _sampleCtg[], unsigned int bagCount): SplitPred(_samplePred, bagCount), sampleCtg(_sampleCtg) {
  run = new Run(ctgWidth);
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
    // TODO:  Pre-empt overflow.
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
      SplitPredProb(levelIdx, &ruPred[splitOff], safeCount);
    }
    else { // Fixed number of predictors splitable.
      SplitPredFixed(levelIdx, &ruPred[splitOff], &heap[splitOff], safeCount);
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
void SplitPred::SplitPredProb(unsigned int levelIdx, const double ruPred[], std::vector<unsigned int> &safeCount) {
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    if (ruPred[predIdx] < predProb[predIdx]) {
      unsigned int rc = bottom->ScheduleSplit(levelIdx, predIdx, safeCount.size());
      if (rc > 1) {
	safeCount.push_back(rc);
      }
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
void SplitPred::SplitPredFixed(unsigned int levelIdx, const double ruPred[], BHPair heap[], std::vector<unsigned int> &safeCount) {
  // Inserts negative, weighted probability value:  choose from lowest.
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    BHeap::Insert(heap, predIdx, -ruPred[predIdx] * predProb[predIdx]);
  }

  // Pops 'predFixed' items in order of increasing value.
  unsigned int schedCount = 0;
  for (unsigned int heapSize = nPred; heapSize > 0; heapSize--) {
    unsigned int predIdx = BHeap::SlotPop(heap, heapSize - 1);
    unsigned int rc = bottom->ScheduleSplit(levelIdx, predIdx, safeCount.size());
    if (rc > 1) {
      safeCount.push_back(rc);
    }
    schedCount += rc == 1 ? 0 : 1;
    if (schedCount == predFixed)
      break;
  }
}


/**
   @brief Base method.  Deletes per-level run and split-flags vectors.

   @return void.
 */
void SplitPred::LevelClear() {
  run->LevelClear();
}


void SplitPred::Split(unsigned int splitIdx, const IndexNode *indexNode, const SPNode *spn) {
  if (bottom->HasRuns(splitIdx)) {
    SplitFac(splitIdx, indexNode, spn);
  }
  else {
    SplitNum(splitIdx, indexNode, spn);
  }
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
  for (unsigned int sIdx = 0; sIdx < index->BagCount(); sIdx++) {
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


// The four splitting functions are specialized according to
// response x predictor type.
//
// Each invokes its own Gini splitting method.  As non-Gini splitting methods
// are provided, however, these must be reparametrized to accommodate newly-
// introduced varieties.
//

/**
   @brief Invokes regression/numeric splitting method, currently only Gini available.

   @param indexNode[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @return void.
 */
void SPReg::SplitNum(unsigned int splitIdx, const IndexNode *indexNode, const SPNode spn[]) {
  int monoMode = MonoMode(splitIdx);
  if (monoMode != 0) {
    SplitNumMono(splitIdx, indexNode, spn, monoMode > 0);
  }
  else {
    SplitNumWV(splitIdx, indexNode, spn);
  }
}


/**
   @brief Determines whether a regression pair undergoes constrained splitting.

   @return The sign of the constraint, if within the splitting probability, else zero.
 */
int SPReg::MonoMode(unsigned int splitIdx) {
  if (predMono == 0)
    return 0;
  
  unsigned int levelIdx, predIdx;
  bottom->SplitRef(splitIdx, levelIdx, predIdx);
  double monoProb = feMono[predIdx];
  int sign = monoProb > 0.0 ? 1 : (monoProb < 0.0 ? -1 : 0);
  return sign * ruMono[splitIdx] < monoProb ? sign : 0;
}


/**
   @brief Invokes regression/factor splitting method, currently only Gini available.

   @param indexNode[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @return void.
 */
void SPReg::SplitFac(unsigned int splitIdx, const IndexNode indexNode[], const SPNode spn[]) {
  SplitFacWV(splitIdx, indexNode, spn);
}


/**
   @brief Invokes categorical/numeric splitting method, currently only Gini available.

   @param indexNode[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @return void.
 */
void SPCtg::SplitNum(unsigned int splitIdx, const IndexNode *indexNode, const SPNode spn[]) {
  SplitNumGini(splitIdx, indexNode, spn);
}


/**
   @brief Invokes categorical/factor splitting method, currently only Gini available.

   @param indexNode[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @return void.
 */
void SPCtg::SplitFac(unsigned int splitIdx, const IndexNode *indexNode, const SPNode spn[]) {
  SplitFacGini(splitIdx, indexNode, spn);
}


/**
   @brief Weighted-variance splitting method.

   @return void.
*/
void SPReg::SplitNumWV(unsigned int splitIdx, const IndexNode *indexNode, const SPNode spn[]) {
  // Walks samples backward from the end of nodes so that ties are not split.
  unsigned int _start, _end;
  unsigned int sCount;
  double sum;
  FltVal preBias, maxGini;
  maxGini = preBias = indexNode->SplitFields(_start, _end, sCount, sum);

  unsigned int rkRight, sampleCount;
  FltVal ySum;
  spn[_end].RegFields(ySum, rkRight, sampleCount);
  double sumR = ySum;
  int sCountL = sCount - sampleCount; // >= 1: counts up to, including, this index. 
  int lhSampCt = 0;

  // Signing values avoids decrementing below zero.
  int start = _start;
  int end = _end;
  int lhSup = end;
  for (int i = end-1; i >= start; i--) {
    int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    unsigned int rkThis;
    spn[i].RegFields(ySum, rkThis, sampleCount);
    if (idxGini > maxGini && rkThis != rkRight) {
      lhSampCt = sCountL;
      lhSup = i;
      maxGini = idxGini;
    }
    sCountL -= sampleCount;
    sumR += ySum;
    rkRight = rkThis;
  }

  if (lhSup < end) {
    bottom->SSWrite(splitIdx, lhSampCt, lhSup + 1 - start, maxGini - preBias);
  }
}


/**
   @brief Weighted-variance splitting method.

   @return void.
*/
void SPReg::SplitNumMono(unsigned int splitIdx, const IndexNode *indexNode, const SPNode spn[], bool increasing) {
  // Walks samples backward from the end of nodes so that ties are not split.
  unsigned int _start, _end;
  unsigned int sCount;
  double sum;
  FltVal preBias, maxGini;
  maxGini = preBias = indexNode->SplitFields(_start, _end, sCount, sum);

  unsigned int rkRight, sampleCount;
  FltVal yVal;
  spn[_end].RegFields(yVal, rkRight, sampleCount);
  double sumR = yVal;
  int sCountL = sCount - sampleCount; // >= 1: counts up to, including, this index. 
  int lhSampCt = 0;

  // Signing values avoids decrementing below zero.
  int start = _start;
  int end = _end;
  int lhSup = end;
  for (int i = end-1; i >= start; i--) {
    int sCountR = sCount - sCountL;
    FltVal sumL = sum - sumR;
    FltVal idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    unsigned int rkThis;
    spn[i].RegFields(yVal, rkThis, sampleCount);
    if (idxGini > maxGini && rkThis != rkRight) {
      FltVal meanL = sumL / sCountL;
      FltVal meanR = sumR / sCountR;
      bool doSplit = increasing ? meanL <= meanR : meanL >= meanR;
      if (doSplit) {
        lhSampCt = sCountL;
        lhSup = i;
        maxGini = idxGini;
      }
    }
    sCountL -= sampleCount;
    sumR += yVal;
    rkRight = rkThis;
  }

  if (lhSup < end) {
    bottom->SSWrite(splitIdx, lhSampCt, lhSup + 1 - start, maxGini - preBias);
  }
}


/**
   @brief Gini-based splitting method.

   @return void.
 */
void SPCtg::SplitNumGini(unsigned int splitIdx, const IndexNode *indexNode, const SPNode spn[]) {
  unsigned int levelIdx, predIdx;
  bottom->SplitRef(splitIdx, levelIdx, predIdx);
  int numIdx = PredBlock::NumIdx(predIdx);
  unsigned int _start, _end;
  unsigned int sCountL;
  double sum;
  FltVal preBias, maxGini;
  maxGini = preBias = indexNode->SplitFields(_start, _end, sCountL, sum);

  double ssL = sumSquares[levelIdx];
  double ssR = 0.0;
  double sumL = sum;
  unsigned int rkRight = spn[_end].Rank();
  unsigned int rkStart = spn[_start].Rank();
  unsigned int lhSampCt = 0;

  // Signing values avoids decrementing below zero.
  int start = _start;
  int end = _end;
  int lhSup = end;
  for (int i = end; i >= start; i--) {
    unsigned int rkThis = spn[i].Rank();
    FltVal sumR = sum - sumL;
    if (rkThis != rkRight && sumL > minDenom && sumR > minDenom) {
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
    double sumRCtg = CtgSumRight(levelIdx, numIdx, yCtg, ySum);
    double sumLCtg = CtgSum(levelIdx, yCtg) - sumRCtg;
    ssR += ySum * (ySum + 2.0 * sumRCtg);
    ssL += ySum * (ySum - 2.0 * sumLCtg);
    sumL -= ySum;
    rkRight = rkThis;
  }

  if (lhSup < end) {
    bottom->SSWrite(splitIdx, lhSampCt, lhSup + 1 - start, maxGini - preBias);
  }
}


/**
   @brief Gini-based splitting method.

   @return void.
 */
void SPCtg::SplitFacGini(unsigned int splitIdx, const IndexNode *indexNode, const SPNode spn[]) {
  unsigned int start, end;
  unsigned int dummy;
  double sum, preBias, maxGini;
  maxGini = preBias = indexNode->SplitFields(start, end, dummy, sum);

  unsigned int levelIdx, predIdx;
  int setIdx;
  bottom->SplitRef(splitIdx, levelIdx, predIdx, setIdx);
  RunSet *runSet = run->RSet(setIdx);
  bottom->SetRunCount(levelIdx, predIdx, BuildRuns(runSet, spn, start, end));
  
  unsigned int lhIdxCount, lhSampCt;
  if (ctgWidth == 2)  {
    lhIdxCount = SplitBinary(runSet, levelIdx, sum, maxGini, lhSampCt);
  }
  else {
    lhIdxCount = SplitRuns(runSet, levelIdx, sum, maxGini, lhSampCt);
  }

  if (lhIdxCount > 0) {
    bottom->SSWrite(splitIdx, lhSampCt, lhIdxCount, maxGini - preBias);
  }
}

 
/**
   @brief Builds categorical runs.  Very similar to regression case, but the runs
   also resolve response sum by category.  Further, heap is optional, passed only
   when run count has been estimated to be wide:

*/
unsigned int SPCtg::BuildRuns(RunSet *runSet, const SPNode spn[], unsigned int _start, unsigned int _end) {
  unsigned int frEnd = _end;
  double sum = 0.0;
  unsigned int sCount = 0;
  unsigned int rkThis = spn[_end].Rank();

  // Signing values avoids decrementing below zero.
  int start = _start;
  int end = _end;
  for (int i = end; i >= start; i--) {
    unsigned int rkRight = rkThis;
    unsigned int yCtg;
    FltVal ySum;
    unsigned int sampleCount = spn[i].CtgFields(ySum, rkThis, yCtg);

    if (rkThis == rkRight) { // Current run's counters accumulate.
      sum += ySum;
      sCount += sampleCount;
    }
    else { // Flushes current run and resets counters for next run.
      runSet->Write(rkRight, sCount, sum, i+1, frEnd);

      sum = ySum;
      sCount = sampleCount;
      frEnd = i;
    }
    runSet->SumCtg(yCtg) += ySum;
  }

  // Flushes remaining run.
  runSet->Write(rkThis, sCount, sum, start, frEnd);

  return runSet->RunCount();
}


/**
   @brief Splits blocks of runs.

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
unsigned int SPCtg::SplitRuns(RunSet *runSet, unsigned int levelIdx, double sum, double &maxGini, unsigned int &lhSampCt) {
  unsigned int countEff = runSet->DeWide();

  unsigned int slotSup = countEff - 1; // Uses post-shrink value.
  unsigned int lhBits = 0;
  unsigned int leftFull = (1 << slotSup) - 1;
  // Nonempty subsets as binary-encoded integers:
  for (unsigned int subset = 1; subset <= leftFull; subset++) {
    double sumL = 0.0;
    double ssL = 0.0;
    double ssR = 0.0;
    for (unsigned int yCtg = 0; yCtg < ctgWidth; yCtg++) {
      double sumCtg = 0.0; // Sum at this category over subset slots.
      for (unsigned int slot = 0; slot < slotSup; slot++) {
	if ((subset & (1 << slot)) != 0) {
	  sumCtg += runSet->SumCtg(slot, yCtg);
	}
      }
      double totSum = CtgSum(levelIdx, yCtg); // Sum at this category over node.
      sumL += sumCtg;
      ssL += sumCtg * sumCtg;
      ssR += (totSum - sumCtg) * (totSum - sumCtg);
    }
    double sumR = sum - sumL;
    // Only relevant for case weighting:  otherwise sums are >= 1.
    if (sumL > minSumL && sumR > minSumR) {
      double subsetGini = ssR / sumR + ssL / sumL;
      if (subsetGini > maxGini) {
        maxGini = subsetGini;
        lhBits = subset;
      }
    }
  }

  return runSet->LHBits(lhBits, lhSampCt);
}


/**
   @brief Adapated from SplitRuns().  Specialized for two-category case in
   which LH subsets accumulate.  This permits running LH 0/1 sums to be
   maintained, as opposed to recomputed, as the LH set grows.

   @return 
 */
unsigned int SPCtg::SplitBinary(RunSet *runSet, unsigned int levelIdx, double sum, double &maxGini, unsigned int &sCount) {
  runSet->HeapBinary();
  runSet->DePop();

  double totR0 = CtgSum(levelIdx, 0); // Sum at this category over node.
  double totR1 = CtgSum(levelIdx, 1);
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
    if (splitable && sumL > minDenom && sumR > minDenom) {
      FltVal ssL = sumL0 * sumL0 + sumL1 * sumL1;
      FltVal ssR = (totR0 - sumL0) * (totR0 - sumL0) + (totR1 - sumL1) * (totR1 - sumL1);
      FltVal cutGini = ssR / sumR + ssL / sumL;
       if (cutGini > maxGini) {
        maxGini = cutGini;
        cut = outSlot;
      }
    } 
  }

  return runSet->LHSlots(cut, sCount);
}


/**
   @brief Weighted-variance splitting method.

   @return void.
 */
void SPReg::SplitFacWV(unsigned int splitIdx, const IndexNode *indexNode, const SPNode spn[]) {
  unsigned int start, end;
  unsigned int sCount;
  double sum, preBias, maxGini;
  maxGini = preBias = indexNode->SplitFields(start, end, sCount, sum);

  unsigned int levelIdx, predIdx;
  int setIdx;
  bottom->SplitRef(splitIdx, levelIdx, predIdx, setIdx);
  RunSet *runSet = run->RSet(setIdx);
  bottom->SetRunCount(levelIdx, predIdx, BuildRuns(runSet, spn, start, end));
  runSet->HeapMean();

  unsigned int idxCountL;
  unsigned int sCountL = HeapSplit(runSet, sum, sCount, idxCountL, maxGini);
  if (sCountL > 0) {
    bottom->SSWrite(splitIdx, sCountL, idxCountL, maxGini - preBias);
  }
}


/**
   Regression runs always maintained by heap.
*/
unsigned int SPReg::BuildRuns(RunSet *runSet, const SPNode spn[], unsigned int _start, unsigned int _end) {
  unsigned int frEnd = _end;
  double sum = 0.0;
  unsigned int sCount = 0;
  unsigned int rkThis = spn[_end].Rank();

  // Signing values avoids decrementing below zero.
  int start = _start;
  int end = _end;
  for (int i = end; i >= start; i--) {
    unsigned int rkRight = rkThis;
    unsigned int sampleCount;
    FltVal ySum;
    spn[i].RegFields(ySum, rkThis, sampleCount);

    if (rkThis == rkRight) { // Same run:  counters accumulate.
      sum += ySum;
      sCount += sampleCount;
    }
    else { // New run:  flush accumulated counters and reset.
      runSet->Write(rkRight, sCount, sum, i+1, frEnd);

      sum = ySum;
      sCount = sampleCount;
      frEnd = i;
    }
  }
  
  // Flushes the remaining run.
  //
  runSet->Write(rkThis, sCount, sum, start, frEnd);

  return runSet->RunCount();
}


/**
   @brief Splits runs sorted by binary heap.

   @param runSet contains all run parameters.

   @param sum is the sum of response values for this index node.

   @param sCountNode is the sample count of the node being split.

   @param _sCount outputs the index count of the argmax LHS.

   @param maxGini outputs the max Gini value.

   @return sample count of LH indices.
*/
unsigned int SPReg::HeapSplit(RunSet *runSet, double sum, unsigned int sCountNode, unsigned int &lhIdxCount, double &maxGini) {
  unsigned int sCountL = 0;
  double sumL = 0.0;
  int cut = -1; // Top index of lh ords in 'facOrd' (q.v.).
  runSet->DePop();
  for (unsigned int outSlot = 0; outSlot < runSet->RunCount() - 1; outSlot++) {
    unsigned int sCountRun;
    sumL += runSet->SumHeap(outSlot, sCountRun);
    sCountL += sCountRun;
    unsigned int sCountR = sCountNode - sCountL;
    double sumR = sum - sumL;
    double cutGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    if (cutGini > maxGini) {
      maxGini = cutGini;
      cut = outSlot;
    }
  }

  lhIdxCount = runSet->LHSlots(cut, sCountL);
  return sCountL;
}
