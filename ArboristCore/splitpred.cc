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
#include "train.h"
#include "splitpred.h"
#include "splitsig.h"
#include "facrun.h"
#include "predictor.h"
#include "samplepred.h"
#include "response.h"
#include "callback.h"

SplitPred *SplitPred::splitPred = 0;
bool *SplitPred::splitFlags = 0;
bool *SplitPred::runFlags = 0;
int SplitPred::levelMax = -1;
int SplitPred::nPred = -1;
int SplitPred::nPredNum = -1;
int SplitPred::nPredFac = -1;
int SplitPred::nFacTot = -1;

int SPCtg::ctgWidth = -1;
double *SPCtgNum::ctgSumR = 0;
double SPCtgNum::minDenom = 1.0e-5;

/**
   @brief Lights off base class initializations.

   @param _levelMax is the current level-max value.

   @return void.
 */
void SplitPred::Factory(int _levelMax) {
  levelMax = _levelMax;
  nPred = Predictor::NPred();
  nPredNum = Predictor::NPredNum();
  nPredFac = Predictor::NPredFac();
  nFacTot = Predictor::NCardTot();

  // Ideally, prefer sufficient slots to allocate once per tree.
  //
  splitFlags = new bool[levelMax * nPred];
  runFlags = new bool[2 * levelMax * nPred]; // Double buffer, alternating between levels.
  RestageMap::Factory(levelMax);
}

/**
   @brief Local initializations and base-class factory.

   @param _levelMax is the current level-max value.

   @return void.
 */
void SPReg::Factory(int _levelMax) {
  SplitPred::Factory(_levelMax);
  splitPred = new SPReg();
  SPRegFac::Factory();
}

/**
   @brief Reallocation of level-based vectors.

   @param _levelMax is the new level-max value.

   @return void.
 */
void SplitPred::ReFactory(int _levelMax) {
  int lmPrev = levelMax;
  levelMax = _levelMax;
  bool *sfTemp = new bool[levelMax * nPred];
  for (int i = 0; i < lmPrev; i++)
    sfTemp[i] = splitFlags[i];
  delete [] splitFlags;
  delete [] runFlags;

  splitFlags = sfTemp;
  runFlags = new bool[2 * levelMax * nPred];
  // TODO:  Move to subroutine, along with TreeInit() version.
  for (int i = 0; i < 2 * levelMax * nPred; i++)
    runFlags[i] = false;
}

/**
   @brief Resets the runFlags[] vector at each new tree.

   @return void.
 */
void SplitPred::TreeInit() {
  for (int i = 0; i < 2 * levelMax * nPred; i++)
    runFlags[i] = false;
}

/**
   @brief Caches latest level-max value and lights off subclass reallocation.

   @param _levelMax is the current level-max value.

   @return void.
 */
void SPReg::ReFactory(int _levelMax) {
  levelMax = _levelMax;

  SPRegFac::ReFactory();
}

/**
   @brief Finalizer.

   @return void.
 */
void SPReg::DeFactory() {
  SplitPred::DeFactory();
  SPRegFac::DeFactory();
}

/**
   @brief Lights off initializers for categorical tree.

   @param _levelMax is the latest level-max value.

   @param _ctgWidth is the response cardinality.

   @return void.
 */
void SPCtg::Factory(int _levelMax, int _ctgWidth) {
  SplitPred::Factory(_levelMax);
  splitPred = new SPCtg();

  ctgWidth = _ctgWidth;
  SPCtgNum::Factory();
  SPCtgFac::Factory();
}

/**
   @brief Reallocation of level-based vectors.

   @param _levelMax is the latest level-max value.

   @return void.
 */
void SPCtg::ReFactory(int _levelMax) {
  levelMax = _levelMax;

  SPCtgNum::ReFactory();
  SPCtgFac::ReFactory();
}

/**
   @brief Base class finalizer.

   @return void.
 */
void SplitPred::DeFactory() {
  delete splitPred;
  delete [] splitFlags;
  delete [] runFlags;
  splitPred = 0;
  splitFlags = 0;
  runFlags = 0;
  levelMax = -1;
  RestageMap::DeFactory();
}

/**
   @brief Invokes base and subclass finalizers.

   @return void.
 */
void SPCtg::DeFactory() {
  SplitPred::DeFactory();
  SPCtgNum::DeFactory();
  SPCtgFac::DeFactory();
}

/**
   @brief Lights off FacRun initializations if factor-valued predictors present.

   @return void.
 */
void SPRegFac::Factory() {
  if (nFacTot > 0)
    FacRun::Factory(levelMax, nPredFac, nFacTot, Predictor::PredFacFirst());
}

/**
   @brief If factor predictors present, reallocates FacRun using revised level-max value.

   @return void.
 */
void SPRegFac::ReFactory() {
  if (nFacTot > 0)
    FacRun::ReFactory(levelMax);
}

/**
   @brief Lights off FacRunCtg initialization if factor-valued predictors present.

   @return void.
 */
void SPCtgFac::Factory() {
  if (nPredFac > 0) {
    FacRunCtg::Factory(levelMax, nPred, nPredFac, nFacTot, Predictor::PredFacFirst(), ctgWidth);
  }
}

/**
   @brief If factor predictors present, reallocates FacRun using revised level-max value.

   @return void.
 */
void SPCtgFac::ReFactory() {
  if (nFacTot > 0)
    FacRunCtg::ReFactory(levelMax);
}

/**
   @brief If factor predictors, finalizes subclass.
 */
void SPCtgFac::DeFactory() {
  if (nFacTot > 0) {
    FacRunCtg::DeFactory();
  }  
}

/**
   @brief Allocates summation checkerboard if numerical predictors present.
 */
void SPCtgNum::Factory() {
  if (nPredNum > 0) {
    ctgSumR = new double[ctgWidth * levelMax * nPredNum];
  }
}

/**
   @brief Reallocates summation checkerboard if numerical predictors present.
 */
void SPCtgNum::ReFactory() {
  if (nPredNum > 0) {
    delete [] ctgSumR;

    ctgSumR = new double[ctgWidth * levelMax * nPredNum];
  }
}

/**
   @brief Finalizer.
 */
void SPCtgNum::DeFactory() {
  ctgWidth = -1;
  if (nPredNum > 0) {
    delete [] ctgSumR;
    ctgSumR = 0;
  }
}

/**
   @brief Finalizer.
 */
void SPRegFac::DeFactory() {
  if (nFacTot > 0)
    FacRun::DeFactory();
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
   @brief Unsets run bit for split/pred pair at current level and resets for descendents at next level.

   @param splitIdx is the split index.

   @param predIdx is the predictor index.

   @param splitL is the LHS index in the next level.

   @param splitR is the RHS index in the next level.

   @param level is the zero-based level.

   @return void.
*/
// Previous level always wiped, so final level remains dirty:  tree reset required.
//
void SplitPred::TransmitRun(int splitIdx, int predIdx, int splitL, int splitR, int level) {
  SetPredRun(splitIdx, predIdx, level, false);
  if (splitL >= 0)
    SetPredRun(splitL, predIdx, level+1, true);
  if (splitR >= 0)
    SetPredRun(splitR, predIdx, level+1, true);
}

/**
   @brief Sets the specified run bit.

   @param splitIdx is the split index.

   @param predIdx is the predictor index.

   @param is the zero-based level.

   @param val is the bit value to set.

   @return void.
*/
void SplitPred::SetPredRun(int splitIdx, int predIdx, int level, bool val) {
  bool *base = runFlags + (nPred * levelMax) * (level & 1);
  base[levelMax * predIdx + splitIdx] = val;
}

/**
   @brief Determines whether this split/pred pair is marked as a run.

   @param splitIdx is the split index.

   @param predIdx is the predictor index.

   @param level is the current level.

   @return true iff this is a run.
*/
bool SplitPred::PredRun(int splitIdx, int predIdx, int level) {
  bool *base = runFlags + (nPred * levelMax) * (level & 1);
  return base[predIdx * levelMax + splitIdx];
}

/**
   @brief Determines whether this split/pred pair is splitable.

   @param splitIdx is the split index.

   @param predIdx is the predictor index.

   @param splitCount is the count of live index nodes.

   @param index is the current level

   @return true iff the  pair is neither in the pred-prob rejection region nor a run.
*/
bool SplitPred::Splitable(int splitIdx, int predIdx, int splitCount, int level) {
  return splitFlags[splitCount * predIdx + splitIdx] && !PredRun(splitIdx, predIdx, level);
}

/**
   @brief Gini-based splitting method.

   @param predIdx is the predictor index.

   @param splitCount is the number of live index nodes.

   @param level is the current level.

   @return void.
 */
// Looping over splitPred[] indices, 'i'.  Populates non-root levels by 'y' values
// sorted according to predictor order.  Target locations given by the tree-defining
// permutation of the corresponding row number.
//
void SPRegNum::SplitGini(int predIdx, int splitCount, int level) {
  SamplePred *samplePred = SamplePred::BufferOff(predIdx, level);
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    if (!Splitable(splitIdx, predIdx, splitCount, level))
      continue;

  // Copies each sampled 'y' value into the appropriate node at each level according
  // to the 'sampleIdx' scheme described below.
  // 'y' is fetched in order of increasing predictor rank.  Values at some rank indices
  // are fetched more than once, others not at all, depending on the random sampling which
  // defines the tree.  The actual ranks are not needed, so remain implicit.
  //
  // Walks samples backward from the end of nodes so that ties are not split.
  //
    int start, end, sCount;
    double sum, maxGini;
    IndexNode::SplitFields(splitIdx, start, end, sCount, sum, maxGini);

    int lhSup = -1;
    int rkRight = samplePred[end].rank;
    double sumR = samplePred[end].yVal;
    // numL >= 1:  counts up to and including this index.
    int numL = sCount - samplePred[end].rowRun;
    int lhSampCt = 0;
    for (int i = end-1; i >= start; i--) {
      // These fields should be 'double' or wider in order to handle "larger" numbers
      // of rows, etc.  There is no noticeable effect on timing when hard-coded to replace
      // double.
      //
      int numR = sCount - numL;
      double sumL = sum - sumR;
      double idxGini = (sumL * sumL) / numL + (sumR * sumR) / numR;
      int rkThis, rowRun;
      double yVal;
      SamplePred::RegFields(samplePred, i, yVal, rkThis, rowRun);
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
      SplitSig::WriteNum(splitIdx, predIdx, level, lhSampCt, lhSup + 1 - start, maxGini);
    }
  }
}


/**
   @brief Gini-based splitting method.

   @param predIdx is the predictor index.

   @param splitCount is the number of live index nodes.

   @param level is the current level.

   @return void.
 */
void SPCtgNum::SplitGini(int predIdx, int splitCount, int level) {
  SamplePred *samplePred = SamplePred::BufferOff(predIdx, level);
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    if (!Splitable(splitIdx, predIdx, splitCount, level))
      continue;

    int start, end, sCount;
    double sum, maxGini;
    IndexNode::SplitFields(splitIdx, start, end, sCount, sum, maxGini);
    double numeratorL = ResponseCtg::SumSquares(splitIdx);
    double numeratorR = 0.0;
    double sumR = 0.0;

    int lhSup = -1;
    int rkRight = samplePred[end].rank;
    int numL = sCount;
    int lhSampCt = 0;
    for (int i = end; i >= start; i--) {
      int rkThis, rowRun, yCtg;
      double yVal;
      SamplePred::CtgFields(samplePred, i, yVal, rkThis, rowRun, yCtg);
      double sumL = sum - sumR;
      double idxGini = (sumL > minDenom && sumR > minDenom) ? numeratorL / sumL + numeratorR / sumR : 0.0;
      // Far-right element does not enter test:  sumR == 0.0, so idxGini = 0.0.
      if (idxGini > maxGini && rkThis != rkRight) {
	lhSampCt = numL;
	lhSup = i;
	maxGini = idxGini;
      }
      numL -= rowRun;

      // Suggested by Andy Liaw's version.  Numerical stability?
      double sumRCtg = CtgSumRight(predIdx, splitIdx, yCtg, yVal);
      double sumLCtg = ResponseCtg::CtgSum(splitIdx, yCtg) - sumRCtg; // Sum to left, inclusive.

      // Numerator, denominator wraparound values for next iteration
      numeratorR += yVal * (yVal + 2.0 * sumRCtg); // Gini computation uses preupdated value.
      numeratorL += yVal * (yVal - 2.0 * sumLCtg);

      // Jittering should greatly reduce possibility of ties.
      // Only bothers to compute giniLocal if both denominators > 1.0e-5
      sumR += yVal;
      rkRight = rkThis;
    }
    if (lhSup >= 0) {
      SplitSig::WriteNum(splitIdx, predIdx, level, lhSampCt, lhSup + 1 - start, maxGini);
    }
    // postcond:  0 <= idx < lhStart
  }
}

/**
   @brief Splits the current level's index nodes.

   @param splitCount is the number of live index nodes.
   
   @param level is the current level.

   @return void.
 */
void SplitPred::Level(int splitCount, int level) {
  ProbSplitable(splitCount);
  splitPred->LevelReset(splitCount);
  if (level == 0)
    splitPred->LevelZero();
  else
    splitPred->RestageAndSplit(splitCount, level);
}

/**
   @brief Initializes the auxilliary data structures associated with all predictors
   for every node live at this level.

   @param splitCount is the number of live index nodes.

   @return void.
*/

// N.B.:  The numeric/regression case uses no auxilliary structures.
//
void SPReg::LevelReset(int splitCount) {
  if (nFacTot > 0)
    FacRun::LevelReset(splitCount);
}

/**
   @brief As above, but categorical response.

   @param splitCount is the number of live index nodes.

   @return void.
*/
void SPCtg::LevelReset(int splitCount) {
  if (nPredNum > 0) {
    SPCtgNum::LevelResetSumR(splitCount);
  }
  if (nFacTot > 0)
    FacRunCtg::LevelReset(splitCount);
}

/**
   @brief Resets the accumulated-sum checkerboard.

   @param splitCount is the number of live index nodes.

   @return void.
 */
void SPCtgNum::LevelResetSumR(int splitCount) {
  for (int i = 0; i < nPredNum * ctgWidth * levelMax; i++)
    ctgSumR[i] = 0.0;
}

/**
   @brief Dispatches blocks of similarly-typed predictors to their respective splitting methods.
   @return void.
 */
void SPReg::LevelZero() {
    int predIdx;

    int predNumFirst = Predictor::PredNumFirst();
    int predNumSup = Predictor::PredNumSup();
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
	SPRegNum::SplitGini(predIdx, 1, 0);
      }
    }

   int predFacFirst = Predictor::PredFacFirst();
   int predFacSup = Predictor::PredFacSup();
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	SPRegFac::SplitGini(predIdx, 1, 0);
      }
    }
}

/**
   @brief As above, but categorical response.
 */
void SPCtg::LevelZero() {
    int predIdx;
    int predNumFirst = Predictor::PredNumFirst();
    int predNumSup = Predictor::PredNumSup();
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
	SPCtgNum::SplitGini(predIdx, 1, 0);
      }
    }

   int predFacFirst = Predictor::PredFacFirst();
   int predFacSup = Predictor::PredFacSup();
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	SPCtgFac::SplitGini(predIdx, 1, 0);
      }
    }
}

/**
   @brief Restages, then splits, blocks of similarly-typed predictors.

   @param splitCount is the number of live index nodes.

   @param level is the current level.

   @return void.
 */
void SPReg::RestageAndSplit(int splitCount, int level) {
    int predIdx;

    int predNumFirst = Predictor::PredNumFirst();
    int predNumSup = Predictor::PredNumSup();
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
	RestageMap::Restage(predIdx, level);
	SPRegNum::SplitGini(predIdx, splitCount, level);
      }
    }

   int predFacFirst = Predictor::PredFacFirst();
   int predFacSup = Predictor::PredFacSup();
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	RestageMap::Restage(predIdx, level);
	SPRegFac::SplitGini(predIdx, splitCount, level);
      }
    }
}

/**
   @brief As above, but categorical response.
 */
void SPCtg::RestageAndSplit(int splitCount, int level) {
  int predIdx;

  int predNumFirst = Predictor::PredNumFirst();
  int predNumSup = Predictor::PredNumSup();
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
	RestageMap::Restage(predIdx, level);
	SPCtgNum::SplitGini(predIdx, splitCount, level);
      }
    }

   int predFacFirst = Predictor::PredFacFirst();
   int predFacSup = Predictor::PredFacSup();
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	RestageMap::Restage(predIdx, level);
	SPCtgFac::SplitGini(predIdx, splitCount, level);
      }
    }
}

/**
   @brief Gini-based splitting method.

   @param predIdx is the predictor index.

   @param splitCount is the number of live index nodes.

   @param level is the current level.

   @return void.
 */
void SPCtgFac::SplitGini(int predIdx, int splitCount, int level) {
  SamplePred *samplePred = SamplePred::BufferOff(predIdx, level);
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    if (!Splitable(splitIdx, predIdx, splitCount, level))
      continue;

    int start, end, dummy;
    double sum, maxGini;
    IndexNode::SplitFields(splitIdx, start, end, dummy, sum, maxGini);

    int top = BuildRuns(samplePred, splitIdx, predIdx, start, end);
    int argMax = SplitRuns(splitIdx, predIdx, splitCount, sum, top, maxGini);

    // Reconstructs LHS sample and index counts from 'argMax'.
    //
    if (argMax > 0) {
      int pairOffset = FacRun::PairOffset(splitIdx, predIdx);
      int sCount = 0;
      int lhIdxCount = 0;
      // TODO:  Check 'slot' range.  Subsets may be narrower
      // than [0,top].
      int lhTop = -1;
      for (int slot = 0; slot <= top; slot++) {
	// If bit # 'slot' set in 'argMax', then the factor value, 'rk', at this
	// position is copied to the the next vacant position, 'lhTop'.
	// Over-writing is not a concern, as 'lhTop' <= 'slot'.
	//
	if ((argMax & (1 << slot)) > 0) {
	  FacRun::Pack(pairOffset, ++lhTop, slot);
	  (void) FacRun::Accum(pairOffset, lhTop, sCount, lhIdxCount);
	}
      }
      SplitSig::WriteFac(splitIdx, predIdx, level, sCount, lhIdxCount, maxGini, lhTop);
    }
  }
}

/**
   @brief Builds runs of ranked predictors for checkerboard processing.

   @param samplePred contains the level's index information.

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
int SPCtgFac::BuildRuns(const SamplePred samplePred[], int splitIdx, int predIdx, int start, int end) {
  int rkThis = -1;
  int sCount = 0;
  double sumR = 0.0;

  int pairOffset = FacRun::PairOffset(splitIdx, predIdx);
  int top = 0; // Top index of compressed rank vector.
  for (int i = end; i >= start; i--) {
    int rkRight = rkThis;
    int rowRun, yCtg;
    double yVal; // Sum of y-values for samples at this entry.
    SamplePred::CtgFields(samplePred, i, yVal, rkThis, rowRun, yCtg);
    bool rhEdge;
    if (rkThis == rkRight) { // No transition:  counters accumulate.
      sumR += yVal;
      sCount += rowRun;
      rhEdge = false;
    }
    else {
      if (rkRight >= 0) // Flushes run to the right.
	FacRunCtg::Transition(pairOffset, top++, rkRight, sCount, sumR);

      // New run:  node reset and bounds initialized.
      //
      sumR = yVal;
      sCount = rowRun;
      rhEdge = true;
    }
    // Always moving left:
    FacRunCtg::LeftTerminus(splitIdx, predIdx, rkThis, i, yCtg, yVal, rhEdge);
  }

  // Flushes the remaining runs.
  //
  FacRunCtg::Transition(pairOffset, top++, rkThis, sCount, sumR);

  return top;
}

/**
   @brief Splits blocks of runs.

   @param splitIdx is the index node index.

   @param predIdx is the predictor index.

   @param splitCount is the number of live index nodes.

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
int SPCtgFac::SplitRuns(int splitIdx, int predIdx, int splitCount, double sum, int &top, double &maxGini) {
  top = FacRunCtg::Shrink(splitIdx, predIdx, splitCount, top);
  int fullSet = (1 << top) - 1;

  // Iterates over all nontrivial subsets of factors in the node.
  // 'Top' value of zero falls out as no-op.
  //
  int argMax = -1;
  for (int subset = 1; subset <= fullSet; subset++) {
    double sumL = 0.0;
    double numerL = 0.0;
    double numerR = 0.0;
    for (int yCtg = 0; yCtg < ctgWidth; yCtg++) {
      double sumCtg = 0.0;
      for (int slot = 0; slot  < top; slot++) {
	if ((subset & (1 << slot)) > 0) {
	  sumCtg += FacRunCtg::SlotSum(splitIdx, predIdx, slot, yCtg);
	}
      }
      double totSum = ResponseCtg::CtgSum(splitIdx, yCtg);
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

   @param splitCount is the number of live index nodes.

   @param level is the current level.

   @return void.
 */
void SPRegFac::SplitGini(int predIdx, int splitCount, int level) {
  SamplePred *samplePred = SamplePred::BufferOff(predIdx, level);
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    if (!Splitable(splitIdx, predIdx, splitCount, level))
      continue;

    int start, end, sCount;
    double sum, maxGini;
    IndexNode::SplitFields(splitIdx, start, end, sCount, sum, maxGini);
    BuildRuns(samplePred, splitIdx, predIdx, start, end);

    int lhIdxCount = end - start + 1; // Initialization for diagnostics, only.
    int lhTop = SplitRuns(splitIdx, predIdx, sum, sCount, lhIdxCount, maxGini);
    if (lhTop >= 0) {
      SplitSig::WriteFac(splitIdx, predIdx, level, sCount, lhIdxCount, maxGini, lhTop);
    }
  }
}

/**
 @brief Builds runs and maintains using FacRunReg coroutines.

 @param samplePred holds the restaged information.

 @param splitIdx is the index node index.

 @param predIdx is the predictor index.

 @param start is the starting index.

 @param end is the ending index.

 @return void. 
*/
void SPRegFac::BuildRuns(const SamplePred samplePred[], int splitIdx, int predIdx, int start, int end) {
  int rkThis = -1;
  int sCount = 0;
  double sumR = 0.0; // Wide node:  see comments above.
  int pairOffset = FacRun::PairOffset(splitIdx, predIdx);
  for (int i = end; i >= start; i--) {
    int rkRight = rkThis;
    double yVal;
    int rowRun;
    SamplePred::RegFields(samplePred, i, yVal, rkThis, rowRun);

    bool rhEdge;
    if (rkThis == rkRight) { // No transition:  counters accumulate.
      sumR += yVal;
      sCount += rowRun;
      rhEdge = false;
    }
    else { // Transition
      if (rkRight >= 0) { // Flushes right-lying run.
	FacRunReg::Transition(splitIdx, predIdx, rkRight, sCount, sumR);
      }
      // New run:  node reset and bounds initialized.
      //
      sumR = yVal;
      sCount = rowRun;
      rhEdge = true;
    }
    // Always moving left:
    FacRun::LeftTerminus(pairOffset, rkThis, i, rhEdge);
  }

  // Flushes the remaining run.
  //
  FacRunReg::Transition(splitIdx, predIdx, rkThis, sCount, sumR);
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
int SPRegFac::SplitRuns(int splitIdx, int predIdx, double sum, int &sCount, int &lhIdxCount, double &maxGini) {
  int sCountTot = sCount; // Captures entry value of full sample count.
  int idxCountTot = lhIdxCount; // Diagnostic only:  see caller, and below.
  int sCountL = 0;
  int idxCount = 0;
  double sumL = 0.0;
  int lhTop = -1; // Top index of lh ords in 'facOrd' (q.v.).
  int pairOffset = FacRun::PairOffset(splitIdx, predIdx);
  int depth = FacRunReg::DePop(splitIdx, predIdx);
  for (int slot = 0; slot < depth - 1; slot++) {
    sumL += FacRun::Accum(pairOffset, slot, sCountL, idxCount);
    int numR = sCountTot - sCountL;
    //if (numR == 0) // ASSERTION.  Will lead to division by zero.
	//cout << "All samples seen but " << depth - slot - 1 <<  " heap slots remain" << endl;

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
  (void) FacRun::Accum(pairOffset, depth - 1, sCountL, idxCount);
  //if (sCountL != sCountTot)
  //cout << "Incomplete sample coverage" << endl;
  //if (idxCount != idxCountTot)
  //cout << "Incomplete index coverage:  " << idxCount << " != " << idxCountTot << endl;
  return lhTop;
}

