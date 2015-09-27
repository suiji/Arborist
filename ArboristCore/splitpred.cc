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
#include "predictor.h"
#include "samplepred.h"
#include "callback.h"
#include "sample.h"

int SplitPred::nPred = -1;
int SplitPred::nPredNum = -1;
int SplitPred::nPredFac = -1;

unsigned int SPCtg::ctgWidth = 0;

/**
   @brief Lights off base class initializations.

   @return void.
 */
void SplitPred::Immutables() {
  nPred = Predictor::NPred();
  nPredNum = Predictor::NPredNum();
  nPredFac = Predictor::NPredFac();
}


/**
   @brief Restores static values to initial.

   @return void.
 */
void SplitPred::DeImmutables() {
  nPred = -1;
  nPredNum = -1;
  nPredFac = -1;
}


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


/**
   @brief Static entry for regression.
 */
SplitPred *SplitPred::FactoryReg(SamplePred *_samplePred) {
  return new SPReg(_samplePred);
}


/**
   @brief Static entry for classification.
 */
SplitPred *SplitPred::FactoryCtg(SamplePred *_samplePred, SampleNodeCtg *_sampleCtg) {
  return new SPCtg(_samplePred, _sampleCtg);
}


void SplitPred::ImmutablesReg(unsigned int _nRow, int _nSamp) {
  Immutables();
  SPReg::Immutables(_nRow, _nSamp);
}


/**
   @brief Immutable initializations.

   @param _nRow is the number of rows.

   @param _nSamp is the number of samples.

   @return void.
 */
void SPReg::Immutables(unsigned int _nRow, int _nSamp) {
  SamplePred::Immutables(nPred, _nSamp, _nRow, 0);
  Run::Immutables(nPred);
}


void SplitPred::ImmutablesCtg(unsigned int _nRow, int _nSamp, unsigned int _ctgWidth) {
  Immutables();
  SPCtg::Immutables(_nRow, _nSamp, _ctgWidth);
}


/**
   @brief Lights off initializers for categorical tree.

   @param _ctgWidth is the response cardinality.

   @return void.
 */
void SPCtg::Immutables(unsigned int _nRow, int _nSamp, unsigned int _ctgWidth) {
  ctgWidth = _ctgWidth;
  SamplePred::Immutables(nPred, _nSamp, _nRow, ctgWidth);
  Run::Immutables(nPred, ctgWidth);
}


/**
   @brief If factor predictors, finalizes subclass.
 */
void SPCtg::DeImmutables() {
  SplitPred::DeImmutables();
  ctgWidth = 0;
  Run::DeImmutables();
}


/**
   @brief Finalizer.
 */
void SPReg::DeImmutables() {
  SplitPred::DeImmutables();
  Run::DeImmutables();
}


SPReg::SPReg(SamplePred *_samplePred) : SplitPred(_samplePred) {
}


SPCtg::SPCtg(SamplePred *_samplePred, SampleNodeCtg _sampleCtg[]) : SplitPred(_samplePred), sampleCtg(_sampleCtg) {
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
  spPair = PairInit(pairCount);
  RunOffsets();
}


void SPReg::RunOffsets() {
  run->OffsetsReg();
}


void SPCtg::RunOffsets() {
  run->OffsetsCtg();
}


void SplitPred::LengthVec(int splitNext) {
  run->LengthVec(splitNext);
}


void SplitPred::LengthTransmit(int splitIdx, int lNext, int rNext) {
  run->LengthTransmit(splitIdx, lNext, rNext);
}


unsigned int &SplitPred::LengthNext(int splitIdx, int predIdx) {
  return run->LengthNext(splitIdx, predIdx);
}
 

/**
   @brief Builds the run workspace using the most recent count values for
   factors splitable at this level.
 */
SPPair *SplitPred::PairInit(int &pairCount) {
  // Whether it is a better idea to avoid the counting step and simply
  // allocate a conservatively-sized buffer is open to debate.
  //
  int idx = 0;
  pairCount = 0;
  int runSets = 0; // Counts splitable multi-run pairs.
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    for (int predIdx = 0; predIdx < nPred; predIdx++) {
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
    for (int predIdx = 0; predIdx < nPred; predIdx++) {
      int rl = run->RunLength(splitIdx, predIdx);
      if (splitFlags[idx] && rl != 1) {
        SPPair *pair = &spPair[pairIdx];
	pair->SetCoords(splitIdx, predIdx);
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
  int len = splitCount * nPred;
  double *ruPred = new double[len];
  CallBack::RUnif(len, ruPred);
  splitFlags = new bool[len];

  int predFixed = Predictor::PredFixed();
  BHPair *heap;
  if (predFixed > 0)
    heap = new BHPair[splitCount * nPred];
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
      SplitPredFixed(predFixed, &ruPred[splitOff], &heap[splitOff], &splitFlags[splitOff]);
    }
  }
  }

  if (heap != 0)
    delete [] heap;
  delete [] ruPred;
  delete [] unsplitable;
}


void SplitPred::SplitPredNull(bool flags[]) {
  for (int predIdx = 0; predIdx < nPred; predIdx++) {
    flags[predIdx] = false;
  }
}


void SplitPred::SplitPredProb(const double ruPred[], bool flags[]) {
  for (int predIdx = 0; predIdx < nPred; predIdx++) {
    double predProb = Predictor::PredProb(predIdx);
    flags[predIdx] = ruPred[predIdx] < predProb;
  }
}

 
void SplitPred::SplitPredFixed(int predFixed, const double ruPred[], BHPair heap[], bool flags[]) {
  for (int predIdx = 0; predIdx < nPred; predIdx++) {
    double predProb = Predictor::PredProb(predIdx);
    BHeap::Insert(heap, predIdx, ruPred[predIdx] * predProb);
    flags[predIdx] = false;
  }

  // Pops 'predFixed' items with highest scores.
  for (int i = nPred - 1; i >= nPred - predFixed; i--){
    int predIdx = BHeap::SlotPop(heap, i);
    flags[predIdx] = true;
  }
}


/**
 */
bool SplitPred::Singleton(int splitIdx, int predIdx) {
  return run->Singleton(splitCount, splitIdx, predIdx);
}

 
/**
 */
void SplitPred::LevelSplit(const IndexNode indexNode[], int level, int splitCount, SplitSig *splitSig) {
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
  SplitPred::LevelClear();
}


SPReg::~SPReg() {
}


SPCtg::~SPCtg() {
}


void SPCtg::LevelClear() {
  if (nPredNum > 0) {
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
double SPReg::Prebias(int splitIdx, int sCount, double sum) {
  return (sum * sum) / sCount;
}


/**
   @brief As above, but categorical response.  Initializes per-level sum vectors as
wells as FacRun vectors.

   @param splitCount is the number of live index nodes.

   @return vector of unsplitable indices.
*/
bool *SPCtg::LevelPreset(const Index *index) {
  if (nPredNum > 0)
    LevelInitSumR();

  bool *unsplitable = new bool[splitCount];
  for (int i = 0; i < splitCount; i++)
    unsplitable[i] = false;
  SumsAndSquares(index, unsplitable);

  return unsplitable;
}


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
    int levelOff = index->LevelOffSample(sIdx);
    if (levelOff >= 0) {
      FltVal sum;
      unsigned int sCount;
      unsigned int ctg = sampleCtg[sIdx].LevelFields(sum, sCount);
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
double SPCtg::Prebias(int splitIdx, int sCount, double sum) {
  return sumSquares[splitIdx] / sum;
}


/**
   @brief Initializes the accumulated-sum checkerboard.

   @return void.
 */
void SPCtg::LevelInitSumR() {
  unsigned int length = nPredNum * ctgWidth * splitCount;
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
	spPair[pairIdx].Split(this, indexNode, nodeBase, splitSig);
      }
    }
}


void SPPair::Split(SplitPred *splitPred, const IndexNode indexNode[], SPNode *nodeBase, SplitSig *splitSig) {
  int splitIdx, predIdx;
  Coords(splitIdx, predIdx);
  if (setIdx >= 0) {
    splitPred->SplitFac(this, &indexNode[splitIdx], SamplePred::PredBase(nodeBase, predIdx), splitSig);
  }
  else {
    splitPred->SplitNum(this, &indexNode[splitIdx], SamplePred::PredBase(nodeBase, predIdx), splitSig);
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
  SplitNumWV(spPair, indexNode, spn, splitSig);
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
  unsigned int rkRight, rowRun;
  FltVal yVal;
  spn[end].RegFields(yVal, rkRight, rowRun);
  double sumR = yVal;
  int numL = sCount - rowRun; // >= 1: counts up to, including, this index. 
  int lhSampCt = 0;
  for (int i = end-1; i >= start; i--) {
    int numR = sCount - numL;
    FltVal sumL = sum - sumR;
    FltVal idxGini = (sumL * sumL) / numL + (sumR * sumR) / numR;
    unsigned int rkThis;
    spn[i].RegFields(yVal, rkThis, rowRun);
    if (idxGini > maxGini && rkThis != rkRight) {
      lhSampCt = numL; // Recomputable: rowRun sum on indices <= 'lhSup'.
      lhSup = i;
      maxGini = idxGini;
    }
    numL -= rowRun;
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
  int splitIdx, predIdx;
  spPair->Coords(splitIdx, predIdx);
  int numIdx = Predictor::NumIdx(predIdx);
  int start, end;
  unsigned int numL;
  double sum;
  FltVal preBias, maxGini;
  maxGini = preBias = indexNode->SplitFields(start, end, numL, sum);

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
        lhSampCt = numL;
        lhSup = i;
        maxGini = cutGini;
      }
    }
    if (rkRight == rkStart) // Last valid cut already checked.
      break;

    unsigned int yCtg;
    FltVal yVal;    
    numL -= spn[i].CtgFields(yVal, yCtg);

    // Maintains sums of category squares incrementally, via update.
    //
    // Right sum is post-incremented with 'yVal', hence is exclusive.
    // Left sum is inclusive.
    //
    double sumRCtg = CtgSumRight(splitIdx, numIdx, yCtg, yVal);
    double sumLCtg = CtgSum(splitIdx, yCtg) - sumRCtg;
    ssR += yVal * (yVal + 2.0 * sumRCtg);
    ssL += yVal * (yVal - 2.0 * sumLCtg);
    sumL -= yVal;
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

  int splitIdx, predIdx;
  RunSet *runSet = run->RSet(spPair->RSet());
  spPair->Coords(splitIdx, predIdx);
  run->RunLength(splitIdx, predIdx) = BuildRuns(runSet, spn, start, end);

  int lhIdxCount;
  unsigned int lhSampCt;
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
    FltVal yVal;
    unsigned int rowRun = spn[i].CtgFields(yVal, rkThis, yCtg);

    if (rkThis == rkRight) { // Current run's counters accumulate.
      sum += yVal;
      sCount += rowRun;
    }
    else { // Flushes current run and resets counters for next run.
      runSet->Write(rkRight, sCount, sum, i+1, frEnd);

      sum = yVal;
      sCount = rowRun;
      frEnd = i;
    }
    runSet->SumCtg(yCtg) += yVal;
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
int SPCtg::SplitRuns(RunSet *runSet, int splitIdx, double sum, double &maxGini, unsigned int &lhSampCt) {
  int countEff = runSet->DeWide();

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
int SPCtg::SplitBinary(RunSet *runSet, int splitIdx, double sum, double &maxGini, unsigned int &sCount) {
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

  int splitIdx, predIdx;
  RunSet *runSet = run->RSet(spPair->RSet());
  spPair->Coords(splitIdx, predIdx);
  run->RunLength(splitIdx, predIdx) = BuildRuns(runSet, spn, start, end);
  runSet->HeapMean();

  int lhIdxCount = HeapSplit(runSet, sum, sCount, maxGini);
  if (lhIdxCount > 0) {
    splitSig->Write(spPair, sCount, lhIdxCount, maxGini - preBias);
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
    unsigned int rowRun;
    FltVal yVal;
    spn[i].RegFields(yVal, rkThis, rowRun);

    if (rkThis == rkRight) { // Same run:  counters accumulate.
      sum += yVal;
      sCount += rowRun;
    }
    else { // New run:  flush accumulated counters and reset.
      runSet->Write(rkRight, sCount, sum, i+1, frEnd);

      sum = yVal;
      sCount = rowRun;
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

   @param _sCount outputs the sample count of the argmax LHS.

   @param maxGini outputs the max Gini value.

   @return count of LH indices.
*/
int SPReg::HeapSplit(RunSet *runSet, double sum, unsigned int &sCount, double &maxGini) {
  int sCountL = 0;
  double sumL = 0.0;
  int cut = -1; // Top index of lh ords in 'facOrd' (q.v.).
  int sCountTot = sCount; // Entry value of full sample count.
  runSet->DePop();
  for (unsigned int outSlot = 0; outSlot < runSet->RunCount() - 1; outSlot++) {
    int sCount;
    sumL += runSet->SumHeap(outSlot, sCount);
    sCountL += sCount;
    int sCountR = sCountTot - sCountL;
    double sumR = sum - sumL;
    double cutGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    if (cutGini > maxGini) {
      maxGini = cutGini;
      cut = outSlot;
    }
  }

  return runSet->LHSlots(cut, sCount);
}
