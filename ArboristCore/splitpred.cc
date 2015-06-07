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

#include <iostream>
using namespace std;

#include "index.h"
#include "splitpred.h"
#include "splitsig.h"
#include "facrun.h"
#include "predictor.h"
#include "samplepred.h"
#include "callback.h"
#include "sample.h"

int SplitPred::nPred = -1;
int SplitPred::nPredNum = -1;
int SplitPred::nPredFac = -1;
int SplitPred::nFacTot = -1;
int SplitPred::predNumFirst = -1;
int SplitPred::predNumSup = -1;
int SplitPred::predFacFirst = -1;
int SplitPred::predFacSup = -1;

unsigned int SPCtg::ctgWidth = 0;
double SPCtg::minDenomNum = 1.0e-5;

/**
   @brief Lights off base class initializations.

   @return void.
 */
void SplitPred::Immutables() {
  nPred = Predictor::NPred();
  nPredNum = Predictor::NPredNum();
  nPredFac = Predictor::NPredFac();
  nFacTot = Predictor::NCardTot();
  predNumFirst = Predictor::PredNumFirst();
  predNumSup = Predictor::PredNumSup();
  predFacFirst = Predictor::PredFacFirst();
  predFacSup = Predictor::PredFacSup();
}


/**
   @brief Restores static values to initial.

   @return void.
 */
void SplitPred::DeImmutables() {
  nPred = -1;
  nPredNum = -1;
  nPredFac = -1;
  nFacTot = -1;
  predNumFirst = -1;
  predNumSup = -1;
  predFacFirst = -1;
  predFacSup = -1;
}


/**
  @brief Constructor.  Initializes 'runFlags' to zero for the single-split root.
 */
SplitPred::SplitPred(SamplePred *_samplePred) : samplePred(_samplePred) {
  runFlags = new bool[nPred];
  for (int i = 0; i < nPred; i++)
    runFlags[i] = false;
}


/**
   @brief Destructor.  Deletes dangling 'runFlags' vector, which should be
   nonempty.
 */
SplitPred::~SplitPred() {
  delete [] runFlags;
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


/**
   @brief Allocates next level's 'runFlags' vector.

   @param splitCount is the number of splits in the next level.

   @return 'runFlags' base of previous level, to be consumed by caller.
 */
bool *SplitPred::RunFlagReplace(int splitCount) {
  bool *rfPrev = runFlags;
  int flagCount = splitCount * nPred;
  runFlags = new bool[flagCount];
  for (int i = 0; i < flagCount; i++)
    runFlags[i] = false;

  return rfPrev;
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
  if (nPredFac > 0)
    FacRun::Immutables(nPredFac, nFacTot, predFacFirst);
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
  if (nPredFac > 0) {
    FacRunOrd::Immutables(nPred, nPredFac, nFacTot, Predictor::PredFacFirst(), ctgWidth);
  }
}


/**
   @brief If factor predictors, finalizes subclass.
 */
void SPCtg::DeImmutables() {
  SplitPred::DeImmutables();
  ctgWidth = 0;
  if (nPredFac > 0) {
    FacRunOrd::DeImmutables();
  }  
}


/**
   @brief Finalizer.
 */
void SPReg::DeImmutables() {
  SplitPred::DeImmutables();
  if (nPredFac > 0)
    FacRunHeap::DeImmutables();
}


SPReg::SPReg(SamplePred *_samplePred) : SplitPred(_samplePred) {
  if (nPredFac > 0)
    frHeap = new FacRunHeap();
  else
    frHeap = 0;
}


SPCtg::SPCtg(SamplePred *_samplePred, SampleNodeCtg _sampleCtg[]) : SplitPred(_samplePred), sampleCtg(_sampleCtg) {
  if (nPredFac > 0)
    frOrd = new FacRunOrd();
  else
    frOrd = 0;
}


/**
   @brief Resets per-predictor split flags at each level using PRNG call-back.

   @param splitCount is the number of live index nodes.

   @return void.
 */
void SplitPred::ProbSplitable(int splitCount) {
  int len = splitCount * nPred;
  double *ruPred = new double[len];
  CallBack::RUnif(len, ruPred);

  splitFlags = new bool[len];
  int idx = 0;
  for (int predIdx = 0; predIdx < nPred; predIdx++) {
    double predProb = Predictor::PredProb(predIdx);
    for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
      splitFlags[idx] = ruPred[idx] < predProb;
      idx++;
    }
  }

  delete [] ruPred;
}


/**
   @brief Unsets run bit for split/pred pair at current level and resets for descendents at next level.  Final level bits dirty, so per-tree reset required.

   @param predIdx is the predictor index.

   @param splitL is the LHS index in the next level.

   @param splitR is the RHS index in the next level.

   @return void.
*/
void SplitPred::TransmitRun(int splitCount, int predIdx, int splitL, int splitR) {
  if (splitL >= 0)
    SetPredRun(splitCount, splitL, predIdx);
  if (splitR >= 0)
    SetPredRun(splitCount, splitR, predIdx);
}


/**
   @brief Sets per-level state enabling pre-bias computation.

   @param index is the Index context.

   @param splitCount is the number of splits in the upcoming level.

   @return void.
*/
void SplitPred::LevelInit(Index *index, int splitCount) {
  LevelPreset(index, splitCount);
  index->SetPrebias(); // Depends on state from LevelPreset()
}


/**
 */
void SplitPred::LevelSplit(const IndexNode indexNode[], int level, int splitCount, SplitSig *splitSig) {
  LevelSplit(indexNode, samplePred->NodeBase(level), splitCount, splitSig);
}

/**
   @brief facRuns should not be deleted until after splits have been consumed.
 */

void SPReg::LevelClear() {
  if (nPredFac > 0) {
    frHeap->LevelClear();
  }
  delete [] splitFlags;
  splitFlags = 0;
}


SPReg::~SPReg() {
  if (nPredFac > 0)
    delete frHeap;
}


SPCtg::~SPCtg() {
  if (nPredFac > 0)
    delete frOrd;
}


void SPCtg::LevelClear() {
  if (nPredFac > 0) {
    frOrd->LevelClear();
  }
  if (nPredNum > 0) {
    delete [] ctgSumR;
  }
  delete [] splitFlags;
  delete [] ctgSum;
  delete [] sumSquares;
  splitFlags = 0;
  ctgSum = sumSquares = ctgSumR = 0;
}


/**
   @brief Splits the current level's index nodes.

   @return void.
 */
void SplitPred::LevelSplit(const IndexNode indexNode[], SPNode *nodeBase, int splitCount, SplitSig *splitSig) {
  ProbSplitable(splitCount);
  Split(indexNode, nodeBase, splitCount, splitSig);
}


/**
   @brief Initializes the auxilliary data structures associated with all predictors
   for every node live at this level.

   @param index is not used by this instantiation.

   @param splitCount is the number of live index nodes.

   @return void.
*/
void SPReg::LevelPreset(const Index *index, int splitCount) {
  if (nPredFac > 0)
    frHeap->LevelInit(splitCount);
}


/**
  @brief Gini pre-bias computation for regression response.

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

   @return void.
*/
void SPCtg::LevelPreset(const Index *index, int splitCount) {
  if (nPredNum > 0)
    LevelInitSumR(splitCount);
  if (nPredFac > 0)
    frOrd->LevelInit(splitCount);

  SumsAndSquares(index, splitCount);
}


void SPCtg::SumsAndSquares(const Index *index, int splitCount) {
  sumSquares = new double[splitCount];
  ctgSum = new double[splitCount * ctgWidth];
  unsigned int levelWidth = index->LevelWidth();
  double *sumTemp = new double[levelWidth * ctgWidth];
  for (unsigned int i = 0; i < levelWidth * ctgWidth; i++)
    sumTemp[i] = 0.0;

  // Sums each category for each node in the upcoming level, including
  // leaves.  Since these appear in arbitrary order, a second pass copies
  // those columns corresponding to nonterminals in split-index order, for
  // ready access by splitting methods.
  //
  for (unsigned int sIdx = 0; sIdx < index->BagCount(); sIdx++) {
    int levelOff = index->LevelOffSample(sIdx);
    if (levelOff >= 0) {
      double sum;
      unsigned int ctg = sampleCtg[sIdx].CtgAndSum(sum);
      sumTemp[levelOff * ctgWidth + ctg] += sum;
    }
  }

  // Reorders by split index, omitting any intervening leaf sums.  Could
  // instead index directly by level offset, but this would require more
  // complex accessor methods.
  //
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    int levelOff = index->LevelOffSplit(splitIdx);
    double ss = 0.0;
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      double sum = sumTemp[levelOff * ctgWidth + ctg];
      ctgSum[splitIdx * ctgWidth + ctg] = sum;
      ss += sum * sum;
    }
    sumSquares[splitIdx] = ss;
  }

  delete [] sumTemp;

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
   @brief Resets the accumulated-sum checkerboard.

   @return void.
 */
void SPCtg::LevelInitSumR(int splitCount) {
  ctgSumR = new double[nPredNum * ctgWidth * splitCount];
  for (unsigned int i = 0; i < nPredNum * ctgWidth *  splitCount; i++)
    ctgSumR[i] = 0.0;
}


/**
   @brief Looks up the bounds associated with a factor run for
   categorical response.

   @param splitIdx is the split index.

   @param predIdx is the predictor index.

   @param slot is the position of the run's factor value within the
   compressed factor vector.

   @param start outputs the starting offset of the run.  Undefined if not
   a run.

   @param end outputs the ending offset of run if run, else undefined.

   @return Factor value at slot position, or -1 if exhausted.
 */
int SPCtg::RunBounds(int splitIdx, int predIdx, int slot, int &start, int &end) {
  int fac = frOrd->FacVal(splitIdx, predIdx, slot);
  if (fac >= 0)
    frOrd->Bounds(splitIdx, predIdx, fac, start, end);

  return fac;
}


/**
   @brief As above, but regression response.
 */
int SPReg::RunBounds(int splitIdx, int predIdx, int slot, int &start, int &end) {
  int fac = frHeap->FacVal(splitIdx, predIdx, slot);
  frHeap->Bounds(splitIdx, predIdx, fac, start, end);

  return fac;
}


/**
   @brief Splits blocks of similarly-typed predictors.

   @return void.
 */
void SPReg::Split(const IndexNode indexNode[], SPNode *nodeBase, int splitCount, SplitSig *splitSig) {
    int predIdx;

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
        SplitNum(indexNode, nodeBase, splitCount, predIdx, splitSig);
      }
    }

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
        SplitFac(indexNode, nodeBase, splitCount, predIdx, splitSig);
      }
    }
}


/**
   @brief As above, but categorical response.
 */
void SPCtg::Split(const IndexNode indexNode[], SPNode *nodeBase, int splitCount, SplitSig *splitSig) {
  int predIdx;

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
	SplitNum(indexNode, nodeBase, splitCount, predIdx, splitSig);
      }
    }

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	SplitFac(indexNode, nodeBase, splitCount, predIdx, splitSig);
      }
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

   @param predIdx is the predictor index.

   @return void.
 */
void SPReg::SplitNum(const IndexNode indexNode[], SPNode *nodeBase, int splitCount, int predIdx, SplitSig *splitSig) {
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    if (Splitable(splitCount, splitIdx, predIdx)) {
      SplitNumGini(&indexNode[splitIdx], SamplePred::PredBase(nodeBase, predIdx), predIdx, splitSig);
    }
  }
}


/**
   @brief Invokes regression/factor splitting method, currently only Gini available.

   @param indexNode[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @param predIdx is the predictor index.

   @return void.
 */
void SPReg::SplitFac(const IndexNode indexNode[], SPNode *nodeBase, int splitCount, int predIdx, SplitSig *splitSig) {
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    if (Splitable(splitCount, splitIdx, predIdx)) {
      SplitHeap(&indexNode[splitIdx], SamplePred::PredBase(nodeBase, predIdx), predIdx, splitSig);
    }
  }
}


/**
   @brief Invokes categorical/numeric splitting method, currently only Gini available.

   @param indexNode[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @param predIdx is the predictor index.

   @return void.
 */
void SPCtg::SplitNum(const IndexNode indexNode[], SPNode *nodeBase, int splitCount, int predIdx, SplitSig *splitSig) {
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    if (Splitable(splitCount, splitIdx, predIdx)) {
      SplitNumGini(&indexNode[splitIdx], SamplePred::PredBase(nodeBase, predIdx), splitCount, predIdx, splitSig);
    }
  }
}


/**
   @brief Invokes categorical/factor splitting method, currently only Gini available.

   @param indexNode[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @param predIdx is the predictor index.

   @return void.
 */
void SPCtg::SplitFac(const IndexNode indexNode[], SPNode *nodeBase, int splitCount, int predIdx, SplitSig *splitSig) {
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    if (Splitable(splitCount, splitIdx, predIdx)) {
      SPCtg::SplitFacGini(&indexNode[splitIdx], SamplePred::PredBase(nodeBase, predIdx), predIdx, splitSig);
    }
  }
}


/**
   @brief Gini-based splitting method.

   @param predIdx is the predictor index.

   @return void.
*/
void SPReg::SplitNumGini(const IndexNode *indexNode, const SPNode spn[], int predIdx, SplitSig *splitSig) {
  // Walks samples backward from the end of nodes so that ties are not split.
  int splitIdx, start, end, sCount;
  double sum, preBias;
  indexNode->SplitFields(splitIdx, start, end, sCount, sum, preBias);
  double maxGini = preBias;
  
  int lhSup = -1;
  double sumR;
  unsigned int rkRight, rowRun;
  spn[end].RegFields(sumR, rkRight, rowRun);
  int numL = sCount - rowRun; // >= 1: counts up to, including, this index. 
  int lhSampCt = 0;
  for (int i = end-1; i >= start; i--) {
    int numR = sCount - numL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / numL + (sumR * sumR) / numR;
    double yVal;
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
    //    if (numL != 0) // ASSERTION
    //cout << "Row runs do not sum to sample count:  " << numL << endl;
  if (lhSup >= 0) {
    splitSig->Write(splitIdx, predIdx, lhSampCt, lhSup + 1 - start, maxGini - preBias);
  }
}


/**
   @brief Gini-based splitting method.

   @param predIdx is the predictor index.

   @return void.
 */
void SPCtg::SplitNumGini(const IndexNode *indexNode, const SPNode spn[], int splitCount, int predIdx, SplitSig *splitSig) {
  int splitIdx, start, end, sCount;
  double sum, preBias;
  indexNode->SplitFields(splitIdx, start, end, sCount, sum, preBias);
  double maxGini = preBias;
  double numeratorL = sumSquares[splitIdx];
  double numeratorR = 0.0;
  double sumR = 0.0;

  int lhSup = -1;
  unsigned int rkRight = spn[end].Rank();
  int numL = sCount;
  int lhSampCt = 0;
  for (int i = end; i >= start; i--) {
    unsigned int yCtg;
    unsigned int rkThis;
    unsigned int rowRun;
    double yVal;    
    spn[i].CtgFields(yVal, rkThis, rowRun, yCtg);
    double sumL = sum - sumR;
    double idxGini = (sumL > minDenomNum && sumR > minDenomNum) ? numeratorL / sumL + numeratorR / sumR : 0.0;

    // Far-right element does not enter test:  sumR == 0.0, so idxGini = 0.0.
    if (idxGini > maxGini && rkThis != rkRight) {
      lhSampCt = numL;
      lhSup = i;
      maxGini = idxGini;
    }
    numL -= rowRun;

    // Suggested by Andy Liaw's version.  Numerical stability?
    double sumRCtg = CtgSumRight(splitCount, splitIdx, predIdx, yCtg, yVal);
    double sumLCtg = CtgSum(splitIdx, yCtg) - sumRCtg; // Sum to left, inclusive.

    // Numerator, denominator wraparound values for next iteration
    // Gini computation employs pre-updated values.
    numeratorR += yVal * (yVal + 2.0 * sumRCtg);
    numeratorL += yVal * (yVal - 2.0 * sumLCtg);

    sumR += yVal;
    rkRight = rkThis;
  }
  if (lhSup >= 0) {
    splitSig->Write(splitIdx, predIdx, lhSampCt, lhSup + 1 - start, maxGini - preBias);
  }
}


/**
   @brief Gini-based splitting method.

   @param predIdx is the predictor index.

   @return void.
 */
void SPCtg::SplitFacGini(const IndexNode *indexNode, const SPNode spn[], int predIdx, SplitSig *splitSig) {
  int splitIdx, start, end, dummy;
  double sum, preBias;
  indexNode->SplitFields(splitIdx, start, end, dummy, sum, preBias);

  double maxGini = preBias;
  unsigned int depth = BuildRuns(spn, splitIdx, predIdx, start, end);
  depth = frOrd->Shrink(splitIdx, predIdx, depth);
  unsigned int lhBits = SplitRuns(splitIdx, predIdx, sum, depth, maxGini);
  int lhSampCt;
  int lhIdxCount = LHBits(lhBits, frOrd->PairOffset(splitIdx, predIdx), depth, lhSampCt);
  if (lhIdxCount > 0) {
    splitSig->Write(splitIdx, predIdx, lhSampCt, lhIdxCount, maxGini - preBias);
  }
}


/**
   @brief Reconstructs LHS sample and index counts from 'lhBits'.
*/
int SPCtg::LHBits(unsigned int lhBits, int pairOffset, unsigned int depth, int &lhSampCt) {
  int lhIdxCount = 0;
  lhSampCt = 0;
  if (lhBits != 0) {
    int lhTop = 0;
    for (unsigned int slot = 0; slot <= depth; slot++) {
	// If bit # 'slot' set in 'argMax', then the factor value, 'rk', at this
	// position is copied to the the next vacant position, 'lhTop'.
	// Over-writing is not a concern, as 'lhTop' <= 'slot'.
	//
      if ((lhBits & (1 << slot)) > 0) {
        frOrd->Pack(pairOffset, lhTop, slot);
        (void) frOrd->Accum(pairOffset, lhTop, lhSampCt, lhIdxCount);
	lhTop++;
      }
    }

    // Marks end of compressed rank vector.
    frOrd->SetFacVal(pairOffset, lhTop, -1);
  }

  return lhIdxCount;
}


/**
   @brief Builds runs of ranked predictors for checkerboard processing.  Final
   run is assigned to RHS, by convention, so need not be built.

   @param splitIdx is the index node index.

   @param predIdx is the predictor index.

   @param start is the starting index.

   @param end is the ending index.

   @return number of runs built.
*/
unsigned int SPCtg::BuildRuns(const SPNode spn[], int splitIdx, int predIdx, int start, int end) {
  int pairOffset = frOrd->PairOffset(splitIdx, predIdx);
  unsigned int vac = 0; // Next vacant index for compressed rank vector.
  double sumR;
  unsigned int rkThis, sCount, yCtg;
  spn[end].CtgFields(sumR, rkThis, sCount, yCtg);
  double yVal = sumR;
  unsigned int rkStart = spn[start].Rank();
  bool rhEdge = true;
  for (int i = end-1; rkThis != rkStart; i--) {
    frOrd->LeftTerminus(splitIdx, predIdx, rkThis, i + 1, yCtg, yVal, rhEdge);
    unsigned int rkRight = rkThis;
    unsigned int rowRun;
    spn[i].CtgFields(yVal, rkThis, rowRun, yCtg);
    if (rkThis == rkRight) { // No transition:  counters accumulate.
      sumR += yVal;
      sCount += rowRun;
      rhEdge = false;
    }
    else {
      // Flushes run to the right.
      frOrd->Transition(pairOffset, vac++, rkRight, sCount, sumR);

      // New run:  node reset and bounds initialized.
      //
      sumR = yVal;
      sCount = rowRun;
      rhEdge = true;
    }
  }

  return vac;
}


/**
   @brief Splits blocks of runs.

   @param splitIdx is the index node index.

   @param predIdx is the predictor index.

   @param sum is the sum of response values for this index node.

   @param depth outputs the (possibly reduced) number of runs.

   @param maxGini outputs the highest observed Gini value.

   @return binary encoding of the maximal-Gini LHS.

   Nodes are now represented compactly as a collection of runs.
   For each node, subsets of these collections are examined, looking for the
   Gini argmax beginning from the pre-bias.

   Iterates over nontrivial subsets, coded by integers as bit patterns.  By
   convention, the final run is incorporated into the RHS of the split, if any.
   That is, the value of 'depth' has already been adjusted to exclude the
   final run.  Hence the number of nonempty subsets to check is '2^depth - 1'.
*/
unsigned int SPCtg::SplitRuns(int splitIdx, int predIdx, double sum, unsigned int depth, double &maxGini) {
  unsigned int fullSet = (1 << depth) - 1;

  // Iterates over all nontrivial subsets of factors in the node.
  // 'Top' value of zero falls out as no-op.
  //
  unsigned int lhBits = 0;
  for (unsigned int subset = 1; subset <= fullSet; subset++) {
    double sumL = 0.0;
    double numerL = 0.0;
    double numerR = 0.0;
    for (unsigned int yCtg = 0; yCtg < ctgWidth; yCtg++) {
      double sumCtg = 0.0;
      for (unsigned int slot = 0; slot < depth; slot++) {
	if ((subset & (1 << slot)) != 0) {
	  sumCtg += frOrd->SlotSum(splitIdx, predIdx, slot, yCtg);
	}
      }
      double totSum = CtgSum(splitIdx, yCtg);
      sumL += sumCtg;
      numerL += sumCtg * sumCtg;
      numerR += (totSum - sumCtg) * (totSum - sumCtg);
    }
    double sumR = sum - sumL;
    double runGini = (sumL <= 1.0e-8 || sumR <= 1.0e-5) ? 0.0 : numerR / sumR + numerL / sumL;
    if (runGini > maxGini) {
      maxGini = runGini;
      lhBits = subset;
    }
  }

  return lhBits;
}


/**
   @brief Gini-based splitting method.

   @param predIdx is the predictor index.

   @return void.
 */
void SPReg::SplitHeap(const IndexNode *indexNode, const SPNode spn[], int predIdx, SplitSig *splitSig) {
  int splitIdx, start, end, sCount;
  double sum, preBias;
  indexNode->SplitFields(splitIdx, start, end, sCount, sum, preBias);
  double maxGini = preBias;
  int depth = HeapRuns(frHeap, spn, splitIdx, predIdx, start, end);

  int lhIdxCount = HeapSplit(frHeap, frHeap->PairOffset(splitIdx, predIdx), depth, sum, sCount, maxGini);
  if (lhIdxCount > 0) {
    splitSig->Write(splitIdx, predIdx, sCount, lhIdxCount, maxGini - preBias);
  }
}


/**
 @brief Builds runs and maintains using FacRunHeap coroutines.

 @param splitIdx is the index node index.

 @param predIdx is the predictor index.

 @param start is the starting index.

 @param end is the ending index.

 @return length of run vector.
*/
int SplitPred::HeapRuns(FacRunHeap *frHeap, const SPNode spn[], int splitIdx, int predIdx, int start, int end) {
  unsigned int rkThis, sCount;
  double sumR;
  int pairOffset = frHeap->PairOffset(splitIdx, predIdx);
  spn[end].RegFields(sumR, rkThis, sCount);
  frHeap->LeftTerminus(pairOffset, rkThis, end, true);
  for (int i = end-1; i >= start; i--) {
    unsigned int rkRight = rkThis;
    double yVal;
    unsigned int rowRun, dummy;
    spn[i].CtgFields(yVal, rkThis, rowRun, dummy);

    bool rhEdge;
    if (rkThis == rkRight) { // Same run:  counters accumulate.
      sumR += yVal;
      sCount += rowRun;
      rhEdge = false;
    }
    else { // New run:  flush accumulated counters and reset.
      frHeap->Transition(splitIdx, predIdx, rkRight, sCount, sumR);
      sumR = yVal;
      sCount = rowRun;
      rhEdge = true;
    }
    // Always moving left:
    frHeap->LeftTerminus(pairOffset, rkThis, i, rhEdge);
  }

  // Flushes the remaining run.
  //
  frHeap->Transition(splitIdx, predIdx, rkThis, sCount, sumR);

  return frHeap->DePop(splitIdx, predIdx);
}


/**
   @brief Splits runs sorted by FacRunHeap coroutines.

   @param splitIdx is the index node index.

   @param predIdx is the predictor index.

   @param sum is the sum of response values for this index node.

   @param _sCount outputs the sample count of the argmax LHS.

   @param _lhIdxCount outputs the index count of the argmax LHS.

   @param maxGini outputs the max Gini value.

   @return count of LH indices.

   BHeap sorts factors by mean-Y over node.  Gini scoring can be done by run, as
   all factors within a run have the same predictor pseudo-value.  Individual run
   members must be noted, however, so that node LHS can be identifed later.
*/
int SplitPred::HeapSplit(FacRunHeap *frHeap, int pairOffset, int depth, double sum, int &sCount, double &maxGini) {
  int sCountTot = sCount; // Captures entry value of full sample count.
  int sCountL = 0;
  int idxCount = 0;
  double sumL = 0.0;
  int lhTop = -1; // Top index of lh ords in 'facOrd' (q.v.).
  int lhIdxCount = 0;
  for (int slot = 0; slot < depth - 1; slot++) {
    sumL += frHeap->Accum(pairOffset, slot, sCountL, idxCount);
    int numR = sCountTot - sCountL;
    //if (numR == 0) // ASSERTION.  Will lead to division by zero.

    double sumR = sum - sumL;
    double runGini = (sumL * sumL) / sCountL + (sumR * sumR) / numR;
    if (runGini > maxGini) {
      maxGini = runGini;
      sCount = sCountL;
      lhIdxCount = idxCount;
      lhTop = slot;
    }
  }
  (void) frHeap->Accum(pairOffset, depth - 1, sCountL, idxCount);
  frHeap->SetFacVal(pairOffset, lhTop + 1, -1);
  // ASSERTIONS:
  //if (sCountL != sCountTot)
  //cout << "Incomplete sample coverage" << endl;
  //if (idxCount != idxCountTot)
  //cout << "Incomplete index coverage:  " << idxCount << " != " << idxCountTot << endl;
  return lhIdxCount;
}
