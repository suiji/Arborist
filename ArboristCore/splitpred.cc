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


/**
   @brief Immutable initializations.

   @param _nRow is the number of rows.

   @param _nSamp is the number of samples.

   @return void.
 */
void SPReg::Immutables(unsigned int _nRow, int _nSamp) {
  SplitPred::Immutables();
  SamplePred::Immutables(nPred, _nSamp, _nRow, 0);
  if (nPredFac > 0)
    FacRun::Immutables(nPredFac, nFacTot, predFacFirst);
}


/**
   @brief Lights off initializers for categorical tree.

   @param _ctgWidth is the response cardinality.

   @return void.
 */
void SPCtg::Immutables(unsigned int _nRow, int _nSamp, unsigned int _ctgWidth) {
  ctgWidth = _ctgWidth;
  SplitPred::Immutables();
  SamplePred::Immutables(nPred, _nSamp, _nRow, ctgWidth);
  if (nPredFac > 0) {
    FacRunCtg::Immutables(nPred, nPredFac, nFacTot, Predictor::PredFacFirst(), ctgWidth);
  }
}


/**
   @brief If factor predictors, finalizes subclass.
 */
void SPCtg::DeImmutables() {
  SplitPred::DeImmutables();
  ctgWidth = 0;
  if (nPredFac > 0) {
    FacRunCtg::DeImmutables();
  }  
}


/**
   @brief Finalizer.
 */
void SPReg::DeImmutables() {
  SplitPred::DeImmutables();
  if (nPredFac > 0)
    FacRunReg::DeImmutables();
}


SPReg::SPReg(SamplePred *_samplePred) : SplitPred(_samplePred) {
  if (nPredFac > 0)
    facRunReg = new FacRunReg();
  else
    facRunReg = 0;
}


SPCtg::SPCtg(SamplePred *_samplePred, SampleNodeCtg _sampleCtg[]) : SplitPred(_samplePred), sampleCtg(_sampleCtg) {
  if (nPredFac > 0)
    facRunCtg = new FacRunCtg();
  else
    facRunCtg = 0;
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
    facRunReg->LevelClear();
  }
  delete [] splitFlags;
  splitFlags = 0;
}


SPReg::~SPReg() {
  if (nPredFac > 0)
    delete facRunReg;
}


SPCtg::~SPCtg() {
  if (nPredFac > 0)
    delete facRunCtg;
}


void SPCtg::LevelClear() {
  if (nPredFac > 0) {
    facRunCtg->LevelClear();
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
    facRunReg->LevelInit(splitCount);
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
    facRunCtg->LevelInit(splitCount);

  ctgSum = new double[splitCount * ctgWidth];
  sumSquares = new double[splitCount];

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


int SPCtg::RunBounds(int splitIdx, int predIdx, int slot, int &start, int &end) {
  int fac = facRunCtg->FacVal(splitIdx, predIdx, slot);
  facRunCtg->Bounds(splitIdx, predIdx, fac, start, end);

  return fac;
}


int SPReg::RunBounds(int splitIdx, int predIdx, int slot, int &start, int &end) {
  int fac = facRunReg->FacVal(splitIdx, predIdx, slot);
  facRunReg->Bounds(splitIdx, predIdx, fac, start, end);

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
        SPReg::SplitNum(indexNode, nodeBase, splitCount, predIdx, splitSig);
      }
    }

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
        SPReg::SplitFac(indexNode, nodeBase, splitCount, predIdx, splitSig);
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
	SPCtg::SplitNum(indexNode, nodeBase, splitCount, predIdx, splitSig);
      }
    }

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	SPCtg::SplitFac(indexNode, nodeBase, splitCount, predIdx, splitSig);
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
      SPReg::SplitNumGini(&indexNode[splitIdx], SamplePred::PredBase(nodeBase, predIdx), predIdx, splitSig);
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
      SPReg::SplitFacGini(&indexNode[splitIdx], SamplePred::PredBase(nodeBase, predIdx), predIdx, splitSig);
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
      SPCtg::SplitNumGini(&indexNode[splitIdx], SamplePred::PredBase(nodeBase, predIdx), splitCount, predIdx, splitSig);
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
    splitSig->WriteNum(splitIdx, predIdx, lhSampCt, lhSup + 1 - start, maxGini - preBias);
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
    splitSig->WriteNum(splitIdx, predIdx, lhSampCt, lhSup + 1 - start, maxGini - preBias);
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
  unsigned int top = BuildRuns(spn, splitIdx, predIdx, start, end);
  unsigned int argMax = SplitRuns(splitIdx, predIdx, sum, top, maxGini);

  // Reconstructs LHS sample and index counts from 'argMax'.
  //
  if (argMax > 0) {
    int pairOffset = facRunCtg->PairOffset(splitIdx, predIdx);
    int lhSampCt = 0;
    int lhIdxCount = 0;
    // TODO:  Check 'slot' range.  Subsets may be narrower
    // than [0,top].
    int lhTop = -1;
    for (unsigned int slot = 0; slot <= top; slot++) {
	// If bit # 'slot' set in 'argMax', then the factor value, 'rk', at this
	// position is copied to the the next vacant position, 'lhTop'.
	// Over-writing is not a concern, as 'lhTop' <= 'slot'.
	//
      if ((argMax & (1 << slot)) > 0) {
        facRunCtg->Pack(pairOffset, ++lhTop, slot);
        (void) facRunCtg->Accum(pairOffset, lhTop, lhSampCt, lhIdxCount);
      }
    }
    splitSig->WriteFac(splitIdx, predIdx, lhSampCt, lhIdxCount, maxGini - preBias, lhTop);
  }
}


/**
   @brief Builds runs of ranked predictors for checkerboard processing.

   @param splitIdx is the index node index.

   @param predIdx is the predictor index.

   @param start is the starting index.

   @param end is the ending index.

   @return number of runs built.

   Retains local sumR values until a transition is noted.  On each transition, pushes
   pair consisting of local factor value (rank) and mean-Y onto node's heap.
   Pushes one more time at conclusion, to catch each node's final factor/mean-Y pair.

   N.B.:  Only the Transition() method is specific to FacRunCtg.  All other actions
   involving the heap can be implemented via FacRun.
*/
unsigned int SPCtg::BuildRuns(const SPNode spn[], int splitIdx, int predIdx, int start, int end) {
  int pairOffset = facRunCtg->PairOffset(splitIdx, predIdx);
  unsigned int top = 0; // Top index of compressed rank vector.
  double sumR;
  unsigned int rkThis, sCount, yCtg;
  spn[end].CtgFields(sumR, rkThis, sCount, yCtg);
  facRunCtg->LeftTerminus(splitIdx, predIdx, rkThis, end, yCtg, sumR, true);
  for (int i = end-1; i >= start; i--) {
    unsigned int rkRight = rkThis;
    unsigned int rowRun;
    double yVal;
    spn[i].CtgFields(yVal, rkThis, rowRun, yCtg);
    bool rhEdge;
    if (rkThis == rkRight) { // No transition:  counters accumulate.
      sumR += yVal;
      sCount += rowRun;
      rhEdge = false;
    }
    else {
      // Flushes run to the right.
      facRunCtg->Transition(pairOffset, top++, rkRight, sCount, sumR);

      // New run:  node reset and bounds initialized.
      //
      sumR = yVal;
      sCount = rowRun;
      rhEdge = true;
    }
    // Always moving left:
    facRunCtg->LeftTerminus(splitIdx, predIdx, rkThis, i, yCtg, yVal, rhEdge);
  }

  // Flushes the remaining runs.
  //
  facRunCtg->Transition(pairOffset, top++, rkThis, sCount, sumR);

  return top;
}


/**
   @brief Splits blocks of runs.

   @param splitIdx is the index node index.

   @param predIdx is the predictor index.

   @param sum is the sum of response values for this index node.

   @param top outputs the (possibly reduced) number of runs.

   @param maxGini outputs the highest observed Gini value.

   @return subset encoding of the maximal-Gini LHS.

   Nodes are now represented compactly as a collection of runs.
   For each node, subsets of these collections are examined, looking for the
   Gini argmax beginning from the pre-bias.

   Iterates over nontrivial subsets, coded by integers as bit patterns.  If the
   full factor set is not present, then all 'facCount' factors may participate
   in the split.  A practical limit of 2^10 trials is employed.  Hence a node
   with more than 11 distinct factors requires random sampling:  selects 1024
   full-width sequences with bits set ~Bernoulli(0.5).
*/
unsigned int SPCtg::SplitRuns(int splitIdx, int predIdx, double sum, unsigned int &top, double &maxGini) {
  top = facRunCtg->Shrink(splitIdx, predIdx, top);
  unsigned int fullSet = (1 << top) - 1;

  // Iterates over all nontrivial subsets of factors in the node.
  // 'Top' value of zero falls out as no-op.
  //
  unsigned int argMax = 0;
  for (unsigned int subset = 1; subset <= fullSet; subset++) {
    double sumL = 0.0;
    double numerL = 0.0;
    double numerR = 0.0;
    for (unsigned int yCtg = 0; yCtg < ctgWidth; yCtg++) {
      double sumCtg = 0.0;
      for (unsigned int slot = 0; slot  < top; slot++) {
	if ((subset & (1 << slot)) > 0) {
	  sumCtg += facRunCtg->SlotSum(splitIdx, predIdx, slot, yCtg);
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
      argMax = subset;
    }
  }

  return argMax;
}


/**
   @brief Gini-based splitting method.

   @param predIdx is the predictor index.

   @return void.
 */
void SPReg::SplitFacGini(const IndexNode *indexNode, const SPNode spn[], int predIdx, SplitSig *splitSig) {
  int splitIdx, start, end, sCount;
  double sum, preBias;
  indexNode->SplitFields(splitIdx, start, end, sCount, sum, preBias);
  double maxGini = preBias;
  BuildRuns(spn, splitIdx, predIdx, start, end);

  int lhIdxCount = end - start + 1; // Initialization for diagnostics, only.
  int lhTop = SplitRuns(splitIdx, predIdx, sum, sCount, lhIdxCount, maxGini);
  if (lhTop >= 0) {
    splitSig->WriteFac(splitIdx, predIdx, sCount, lhIdxCount, maxGini - preBias, lhTop);
  }
}


/**
 @brief Builds runs and maintains using FacRunReg coroutines.

 @param splitIdx is the index node index.

 @param predIdx is the predictor index.

 @param start is the starting index.

 @param end is the ending index.

 @return void. 
*/
void SPReg::BuildRuns(const SPNode spn[], int splitIdx, int predIdx, int start, int end) {
  unsigned int rkThis, sCount;
  double sumR;
  int pairOffset = facRunReg->PairOffset(splitIdx, predIdx);
  spn[end].RegFields(sumR, rkThis, sCount);
  facRunReg->LeftTerminus(pairOffset, rkThis, end, true);
  for (int i = end-1; i >= start; i--) {
    unsigned int rkRight = rkThis;
    double yVal;
    unsigned int rowRun;
    spn[i].RegFields(yVal, rkThis, rowRun);

    bool rhEdge;
    if (rkThis == rkRight) { // Same run:  counters accumulate.
      sumR += yVal;
      sCount += rowRun;
      rhEdge = false;
    }
    else { // New run:  flush accumulated counters and reset.
      facRunReg->Transition(splitIdx, predIdx, rkRight, sCount, sumR);
      sumR = yVal;
      sCount = rowRun;
      rhEdge = true;
    }
    // Always moving left:
    facRunReg->LeftTerminus(pairOffset, rkThis, i, rhEdge);
  }

  // Flushes the remaining run.
  //
  facRunReg->Transition(splitIdx, predIdx, rkThis, sCount, sumR);
}


/**
   @brief Splits runs sorted by FacRunReg coroutines.

   @param splitIdx is the index node index.

   @param predIdx is the predictor index.

   @param sum is the sum of response values for this index node.

   @param _sCount outputs the sample count of the argmax LHS.

   @param _lhIdxCount outputs the index count of the argmax LHS.

   @param maxGini outputs the max Gini value.

   @return top index of LHS.

   BHeap sorts factors by mean-Y over node.  Gini scoring can be done by run, as
   all factors within a run have the same predictor pseudo-value.  Individual run
   members must be noted, however, so that node LHS can be identifed later.
*/
int SPReg::SplitRuns(int splitIdx, int predIdx, double sum, int &sCount, int &lhIdxCount, double &maxGini) {
  int sCountTot = sCount; // Captures entry value of full sample count.
  //int idxCountTot = lhIdxCount; // Diagnostic only:  see caller, and below.
  int sCountL = 0;
  int idxCount = 0;
  double sumL = 0.0;
  int lhTop = -1; // Top index of lh ords in 'facOrd' (q.v.).
  int pairOffset = facRunReg->PairOffset(splitIdx, predIdx);
  int depth = facRunReg->DePop(splitIdx, predIdx);
  for (int slot = 0; slot < depth - 1; slot++) {
    sumL += facRunReg->Accum(pairOffset, slot, sCountL, idxCount);
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
  // ASSERTIONS:
  (void) facRunReg->Accum(pairOffset, depth - 1, sCountL, idxCount);
  //if (sCountL != sCountTot)
  //cout << "Incomplete sample coverage" << endl;
  //if (idxCount != idxCountTot)
  //cout << "Incomplete index coverage:  " << idxCount << " != " << idxCountTot << endl;
  return lhTop;
}

