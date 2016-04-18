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
#include "run.h"
#include "samplepred.h"
#include "callback.h"
#include "sample.h"
#include "predblock.h"

unsigned int SplitPred::nPred = 0;
int SplitPred::predFixed = 0;
std::vector<double> SplitPred::predProb(0);

double *SPReg::mono = 0;
unsigned int SPReg::predMono = 0;
unsigned int SPCtg::ctgWidth = 0;

/**
  @brief Constructor.  Initializes 'runFlags' to zero for the single-split root.
 */
SplitPred::SplitPred(SamplePred *_samplePred) : samplePred(_samplePred) {
  run = new Run();
}


/**
   @brief Destructor.  Deletes dangling 'runFlags' vector, which should be
   nonempty.
 */
SplitPred::~SplitPred() {
  delete run;
}


void SplitPred::Immutables(unsigned int _nPred, unsigned int _ctgWidth, int _predFixed, const double _predProb[], const double _regMono[]) {
  nPred = _nPred;
  predFixed = _predFixed;
  predProb = std::vector<double>(nPred);
  for (unsigned int i = 0; i < nPred; i++)
    predProb[i] = _predProb[i];

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
  mono = new double[_nPred];
  for (unsigned int i = 0; i < _nPred; i++) {
    double monoProb = _mono[i];
    predMono += monoProb != 0.0;
    mono[i] = _mono[i];
  }
}


void SPReg::DeImmutables() {
  predMono = 0;
  delete [] mono;
}


void SPCtg::Immutables(unsigned int _ctgWidth) {
  ctgWidth = _ctgWidth;
}


void SPCtg::DeImmutables() {
  ctgWidth = 0;
}


/**
   @brief Static entry for regression.
 */
SplitPred *SplitPred::FactoryReg(SamplePred *_samplePred) {
  return new SPReg(_samplePred);
}


/**
   @brief Static entry for classification.
 */
SplitPred *SplitPred::FactoryCtg(SamplePred *_samplePred, SampleNode *_sampleCtg) {
  return new SPCtg(_samplePred, _sampleCtg);
}


/**
   @brief Constructor.

   @param samplePred holds (re)staged node contents.
 */
SPReg::SPReg(SamplePred *_samplePred) : SplitPred(_samplePred), ruMono(0) {
}


/**
   @brief Constructor.

   @param samplePred holds (re)staged node contents.

   @param sampleCtg is the sample vector for the tree, included for category lookup.
 */
SPCtg::SPCtg(SamplePred *_samplePred, SampleNode _sampleCtg[]): SplitPred(_samplePred), sampleCtg(_sampleCtg) {
}


/**
   @brief Sets per-level state enabling pre-bias computation.

   @param index is the Index context.

   @param splitCount is the number of splits in the upcoming level.

   @return void.
*/
void SplitPred::LevelInit(Index *index, int _splitCount) {
  splitCount = _splitCount;
  run->LevelInit(splitCount);
  bool *unsplitable = LevelPreset(index);
  SplitFlags(unsplitable);
  index->SetPrebias(); // Depends on state from LevelPreset()
  spPair = PairInit(nPred, pairCount);
  RunOffsets();
}

void SPReg::LevelInit(Index *index, int _splitCount) {
  SplitPred::LevelInit(index, _splitCount);
  if (predMono > 0) {
    ruMono = new double[pairCount];
    CallBack::RUnif(pairCount, ruMono);
  }
  else {
    ruMono = 0;
  }
}


/**
   @brief Sets quick lookup offets for Run object.

   @return void.
 */
void SPReg::RunOffsets() {
  run->OffsetsReg();
}


/**
   @brief Sets quick lookup offsets for Run object.
 */
void SPCtg::RunOffsets() {
  run->OffsetsCtg();
}


/**
   @brief Sets run length sups for next row.
 */
void SplitPred::LengthVec(int splitNext) {
  run->LengthVec(splitNext);
}


/**
 */
void SplitPred::LengthTransmit(int splitIdx, int lNext, int rNext) {
  run->LengthTransmit(splitIdx, lNext, rNext);
}


/**
 */
unsigned int &SplitPred::LengthNext(int splitIdx, unsigned int predIdx) {
  return run->LengthNext(splitIdx, predIdx);
}
 

/**
   @brief Builds the run workspace using the most recent count values for
   factors splitable at this level.
 */
SPPair *SplitPred::PairInit(unsigned int nPred, int &pairCount) {
  // Whether it is a better idea to avoid the counting step and simply
  // allocate a conservatively-sized buffer is open to debate.
  //
  int idx = 0;
  pairCount = 0;
  int runSets = 0; // Counts splitable multi-run pairs.
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      int rl = run->RunLength(splitIdx, predIdx);
      if (splitFlags[idx] && rl != 1) {
	pairCount++;
        runSets += (rl > 1 ? 1 : 0);
      }
      idx++;
    }
  }
  run->RunSets(runSets);

  SPPair *spPair = new SPPair[pairCount];
  idx = 0;
  int pairIdx = 0;
  int rSetIdx = 0;
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      int rl = run->RunLength(splitIdx, predIdx);
      if (splitFlags[idx] && rl != 1) {
        SPPair *pair = &spPair[pairIdx];
	pair->Init(splitIdx, predIdx, pairIdx);
	if (rl > 1) {
	  run->CountSafe(rSetIdx) = rl;
	  pair->SetRSet(rSetIdx++);
	}
	else {
	  pair->SetRSet(-1);
	}
	pairIdx++;
      }
      idx++;
    }
  }
  
  return spPair;
}


/**
   @brief Initializes pair-specific information, such as splitflags.

   @param unsplitable lists unsplitable nodes.

   @return void.
*/
void SplitPred::SplitFlags(bool unsplitable[]) {
  int cellCount = splitCount * nPred;
  double *ruPred = new double[cellCount];
  CallBack::RUnif(cellCount, ruPred);
  splitFlags = new bool[cellCount];

  BHPair *heap;
  if (predFixed > 0)
    heap = new BHPair[cellCount];
  else
    heap = 0;

  int splitIdx, splitOff;
#pragma omp parallel default(shared) private(splitOff, splitIdx)
  {
#pragma omp for schedule(dynamic, 1)
  for (splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    splitOff = splitIdx * nPred;
    if (unsplitable[splitIdx]) { // No predictor splitable
      SplitPredNull(&splitFlags[splitOff]);
    }
    else if (predFixed == 0) { // Probability of predictor splitable.
      SplitPredProb(&ruPred[splitOff], &splitFlags[splitOff]);
    }
    else { // Fixed number of predictors splitable.
      SplitPredFixed(&ruPred[splitOff], &heap[splitOff], &splitFlags[splitOff]);
    }
  }
  }

  if (heap != 0)
    delete [] heap;
  delete [] ruPred;
  delete [] unsplitable;
}


/**
 */
void SplitPred::SplitPredNull(bool flags[]) {
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    flags[predIdx] = false;
  }
}


/**
   @brief Set splitable flag by Bernoulli sampling.

   @param ruPred is a vector of uniformly-sampled variates.

   @param flags outputs the splitability predicate.

   @return void, with output vector.
 */
void SplitPred::SplitPredProb(const double ruPred[], bool flags[]) {
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    flags[predIdx] = ruPred[predIdx] < predProb[predIdx];
  }
}

 
/**
   @brief Sets splitable flag for a fixed number of predictors.

   @param ruPred is a vector of uniformly-sampled variates.

   @param heap orders probability-weighted variates.

   @param flags outputs the splitability predicate.

   @return void, with output vector.
 */
void SplitPred::SplitPredFixed(const double ruPred[], BHPair heap[], bool flags[]) {
  // Inserts negative, weighted probability value:  choose from lowest.
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    BHeap::Insert(heap, predIdx, -ruPred[predIdx] * predProb[predIdx]);
    flags[predIdx] = false;
  }

  // Pops 'predFixed' items in order of Dbincreasing value.
  for (unsigned int i = nPred - 1; i >= nPred - predFixed; i--){
    unsigned int predIdx = BHeap::SlotPop(heap, i);
    flags[predIdx] = true;
  }
}


/**
 */
bool SplitPred::Singleton(int splitIdx, unsigned int predIdx) {
  return run->Singleton(splitCount, splitIdx, predIdx);
}

 
/**
 */
void SplitPred::LevelSplit(const IndexNode indexNode[], unsigned int level, int splitCount, SplitSig *splitSig) {
  LevelSplit(indexNode, samplePred->NodeBase(level), splitCount, splitSig);
}


/**
   @brief Base method.  Deletes per-level run and split-flags vectors.

   @return void.
 */
void SplitPred::LevelClear() {
  run->LevelClear();
  delete [] spPair;
  delete [] splitFlags;
  splitFlags = 0;
  spPair = 0;
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
   @brief Splits the current level's index nodes.

   @return void.
 */
void SplitPred::LevelSplit(const IndexNode indexNode[], SPNode *nodeBase, int splitCount, SplitSig *splitSig) {
  Split(indexNode, nodeBase, splitSig);
}


/**
   @brief Currently just a stub.

   @param index is not used by this instantiation.

   @param splitCount is the number of live index nodes.

   @return vector of unsplitable indices.
*/
bool *SPReg::LevelPreset(const Index *index) {
  bool* unsplitable = new bool[splitCount];
  for (int i = 0; i < splitCount; i++)
    unsplitable[i] = false;

  return unsplitable;
}


/**
  @brief Weight-variance pre-bias computation for regression response.

  @param splitIdx is the split index.

  @param sCount is the number of samples subsumed by the index node.

  @param sum is the sum of samples subsumed by the index node.

  @return square squared, divided by sample count.
*/
double SPReg::Prebias(int splitIdx, unsigned int sCount, double sum) {
  return (sum * sum) / sCount;
}


/**
   @brief As above, but categorical response.  Initializes per-level sum vectors as
wells as FacRun vectors.

   @param splitCount is the number of live index nodes.

   @return vector of unsplitable indices.
*/
bool *SPCtg::LevelPreset(const Index *index) {
  if (PredBlock::NPredNum() > 0)
    LevelInitSumR();

  bool *unsplitable = new bool[splitCount];
  for (int i = 0; i < splitCount; i++)
    unsplitable[i] = false;
  SumsAndSquares(index, unsplitable);

  return unsplitable;
}


/**
 */
void SPCtg::SumsAndSquares(const Index *index, bool unsplitable[]) {
  sumSquares = new double[splitCount];
  ctgSum = new double[splitCount * ctgWidth];
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
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    int levelOff = index->LevelOffSplit(splitIdx);
    unsigned int indexSCount = index->SCount(splitIdx);
    double ss = 0.0;
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      unsigned int sCount = sCountTemp[levelOff * ctgWidth + ctg];
      if (sCount == indexSCount) { // Singleton response:  avoid splitting.
	unsplitable[splitIdx] = true;
      }
      double sum = sumTemp[levelOff * ctgWidth + ctg];
      ctgSum[splitIdx * ctgWidth + ctg] = sum;
      ss += sum * sum;
    }
    sumSquares[splitIdx] = ss;
  }

  delete [] sumTemp;
  delete [] sCountTemp;
}


/**
   @brief Gini pre-bias computation for categorical response.

   @param splitIdx is the split index.

   @param sCount is the number of samples subsumed by the index node.

   @param sum is the sum of samples subsumed by the index node.

   @return sum of squares divided by sum.
 */
double SPCtg::Prebias(int splitIdx, unsigned int sCount, double sum) {
  return sumSquares[splitIdx] / sum;
}


/**
   @brief Initializes the accumulated-sum checkerboard.

   @return void.
 */
void SPCtg::LevelInitSumR() {
  unsigned int length = PredBlock::NPredNum() * ctgWidth * splitCount;
  ctgSumR = new double[length];
  for (unsigned int i = 0; i < length; i++)
    ctgSumR[i] = 0.0;
}


/**
   @brief Dispatches splitting of live pairs in no particular order.

   @return void.
 */
void SplitPred::Split(const IndexNode indexNode[], SPNode *nodeBase, SplitSig *splitSig) {
  int pairIdx;

#pragma omp parallel default(shared) private(pairIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (pairIdx = 0; pairIdx < pairCount; pairIdx++) {
	spPair[pairIdx].Split(this, indexNode, nodeBase, samplePred, splitSig);
      }
    }
}


void SPPair::Split(SplitPred *splitPred, const IndexNode indexNode[], SPNode *nodeBase, const SamplePred *samplePred, SplitSig *splitSig) {
  int splitIdx;
  unsigned int predIdx;
  Coords(splitIdx, predIdx);
  if (setIdx >= 0) {
    splitPred->SplitFac(this, &indexNode[splitIdx], samplePred->PredBase(nodeBase, predIdx), splitSig);
  }
  else {
    splitPred->SplitNum(this, &indexNode[splitIdx], samplePred->PredBase(nodeBase, predIdx), splitSig);
  }
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
void SPReg::SplitNum(const SPPair *spPair, const IndexNode *indexNode, const SPNode spn[], SplitSig *splitSig) {
  int monoMode = MonoMode(spPair);
  if (monoMode != 0) {
    SplitNumMono(spPair, indexNode, spn, splitSig, monoMode > 0);
  }
  else {
    SplitNumWV(spPair, indexNode, spn, splitSig);
  }
}


/**
   @brief Determines whether a regression pair undergoes constrained splitting.

   @return The sign of the constraint, if within the splitting probability, else zero.
 */
int SPReg::MonoMode(const SPPair *pair) {
  if (predMono == 0)
    return 0;
  
  int splitIdx;
  unsigned int predIdx;
  pair->Coords(splitIdx, predIdx);
  double monoProb = mono[predIdx];
  int sign = monoProb > 0.0 ? 1 : (monoProb < 0.0 ? -1 : 0);
  return sign * ruMono[pair->PairIdx()] < monoProb ? sign : 0;
}


/**
   @brief Invokes regression/factor splitting method, currently only Gini available.

   @param indexNode[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @return void.
 */
void SPReg::SplitFac(const SPPair *spPair, const IndexNode indexNode[], const SPNode spn[], SplitSig *splitSig) {
  SplitFacWV(spPair, indexNode, spn, splitSig);
}


/**
   @brief Invokes categorical/numeric splitting method, currently only Gini available.

   @param indexNode[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @return void.
 */
void SPCtg::SplitNum(const SPPair *spPair, const IndexNode *indexNode, const SPNode spn[], SplitSig *splitSig) {
  SplitNumGini(spPair, indexNode, spn, splitSig);
}


/**
   @brief Invokes categorical/factor splitting method, currently only Gini available.

   @param indexNode[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @return void.
 */
void SPCtg::SplitFac(const SPPair *spPair, const IndexNode *indexNode, const SPNode spn[], SplitSig *splitSig) {
  SplitFacGini(spPair, indexNode, spn, splitSig);
}


/**
   @brief Weighted-variance splitting method.

   @return void.
*/
void SPReg::SplitNumWV(const SPPair *spPair, const IndexNode *indexNode, const SPNode spn[], SplitSig *splitSig) {
  // Walks samples backward from the end of nodes so that ties are not split.
  int start, end;
  unsigned int sCount;
  double sum;
  FltVal preBias, maxGini;
  maxGini = preBias = indexNode->SplitFields(start, end, sCount, sum);

  int lhSup = end;
  unsigned int rkRight, sampleCount;
  FltVal ySum;
  spn[end].RegFields(ySum, rkRight, sampleCount);
  double sumR = ySum;
  int sCountL = sCount - sampleCount; // >= 1: counts up to, including, this index. 
  int lhSampCt = 0;
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
    splitSig->Write(spPair, lhSampCt, lhSup + 1 - start, maxGini - preBias);
  }
}


/**
   @brief Weighted-variance splitting method.

   @return void.
*/
void SPReg::SplitNumMono(const SPPair *spPair, const IndexNode *indexNode, const SPNode spn[], SplitSig *splitSig, bool increasing) {
  // Walks samples backward from the end of nodes so that ties are not split.
  int start, end;
  unsigned int sCount;
  double sum;
  FltVal preBias, maxGini;
  maxGini = preBias = indexNode->SplitFields(start, end, sCount, sum);

  int lhSup = end;
  unsigned int rkRight, sampleCount;
  FltVal yVal;
  spn[end].RegFields(yVal, rkRight, sampleCount);
  double sumR = yVal;
  int sCountL = sCount - sampleCount; // >= 1: counts up to, including, this index. 
  int lhSampCt = 0;
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
    splitSig->Write(spPair, lhSampCt, lhSup + 1 - start, maxGini - preBias);
  }
}


/**
   @brief Gini-based splitting method.

   @return void.
 */
void SPCtg::SplitNumGini(const SPPair *spPair, const IndexNode *indexNode, const SPNode spn[], SplitSig *splitSig) {
  int splitIdx;
  unsigned int predIdx;
  spPair->Coords(splitIdx, predIdx);
  int numIdx = PredBlock::NumIdx(predIdx);
  int start, end;
  unsigned int sCountL;
  double sum;
  FltVal preBias, maxGini;
  maxGini = preBias = indexNode->SplitFields(start, end, sCountL, sum);

  double ssL = sumSquares[splitIdx];
  double ssR = 0.0;
  double sumL = sum;
  int lhSup = end;
  unsigned int rkRight = spn[end].Rank();
  unsigned int rkStart = spn[start].Rank();
  unsigned int lhSampCt = 0;
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
    double sumRCtg = CtgSumRight(splitIdx, numIdx, yCtg, ySum);
    double sumLCtg = CtgSum(splitIdx, yCtg) - sumRCtg;
    ssR += ySum * (ySum + 2.0 * sumRCtg);
    ssL += ySum * (ySum - 2.0 * sumLCtg);
    sumL -= ySum;
    rkRight = rkThis;
  }

  if (lhSup < end) {
    splitSig->Write(spPair, lhSampCt, lhSup + 1 - start, maxGini - preBias);
  }
}


/**
   @brief Gini-based splitting method.

   @return void.
 */
void SPCtg::SplitFacGini(const SPPair *spPair, const IndexNode *indexNode, const SPNode spn[], SplitSig *splitSig) {
  int start, end;
  unsigned int dummy;
  double sum, preBias, maxGini;
  maxGini = preBias = indexNode->SplitFields(start, end, dummy, sum);

  int splitIdx;
  unsigned int predIdx;
  RunSet *runSet = run->RSet(spPair->RSet());
  spPair->Coords(splitIdx, predIdx);
  run->RunLength(splitIdx, predIdx) = BuildRuns(runSet, spn, start, end);

  unsigned int lhIdxCount, lhSampCt;
  if (ctgWidth == 2)  {
    lhIdxCount = SplitBinary(runSet, splitIdx, sum, maxGini, lhSampCt);
  }
  else {
    lhIdxCount = SplitRuns(runSet, splitIdx, sum, maxGini, lhSampCt);
  }

  if (lhIdxCount > 0) {
    splitSig->Write(spPair, lhSampCt, lhIdxCount, maxGini - preBias);
  }
}

 
/**
   @brief Builds categorical runs.  Very similar to regression case, but the runs
   also resolve response sum by category.  Further, heap is optional, passed only
   when run count has been estimated to be wide:

*/
unsigned int SPCtg::BuildRuns(RunSet *runSet, const SPNode spn[], int start, int end) {
  int frEnd = end;
  double sum = 0.0;
  unsigned int sCount = 0;
  unsigned int rkThis = spn[end].Rank();
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
unsigned int SPCtg::SplitRuns(RunSet *runSet, int splitIdx, double sum, double &maxGini, unsigned int &lhSampCt) {
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
      double totSum = CtgSum(splitIdx, yCtg); // Sum at this category over node.
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
unsigned int SPCtg::SplitBinary(RunSet *runSet, int splitIdx, double sum, double &maxGini, unsigned int &sCount) {
  runSet->HeapBinary();
  runSet->DePop();

  double totR0 = CtgSum(splitIdx, 0); // Sum at this category over node.
  double totR1 = CtgSum(splitIdx, 1);
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
void SPReg::SplitFacWV(const SPPair *spPair, const IndexNode *indexNode, const SPNode spn[], SplitSig *splitSig) {
  int start, end;
  unsigned int sCount;
  double sum, preBias, maxGini;
  maxGini = preBias = indexNode->SplitFields(start, end, sCount, sum);

  int splitIdx;
  unsigned int predIdx;
  RunSet *runSet = run->RSet(spPair->RSet());
  spPair->Coords(splitIdx, predIdx);
  run->RunLength(splitIdx, predIdx) = BuildRuns(runSet, spn, start, end);
  runSet->HeapMean();

  unsigned int idxCountL;
  unsigned int sCountL = HeapSplit(runSet, sum, sCount, idxCountL, maxGini);
  if (sCountL > 0) {
    splitSig->Write(spPair, sCountL, idxCountL, maxGini - preBias);
  }
}


/**
   Regression runs always maintained by heap.
*/
unsigned int SPReg::BuildRuns(RunSet *runSet, const SPNode spn[], int start, int end) {
  int frEnd = end;
  double sum = 0.0;
  unsigned int sCount = 0;
  unsigned int rkThis = spn[end].Rank();
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
