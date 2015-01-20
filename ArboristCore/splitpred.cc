// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <iostream>
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

// Computes the number of levels, 'levels', in the tree.
// Computes the maximum node index, 'levelMax', as a function of 'minHeight'.
// 'minHeight' is the smallest node size for which a root node is to be split.
// Hence nodes rooted at indices 'levelMax' or higher are not split.
//
// Precond: nSamp > 0; minHeight >= 3, minHeight <= nSamp.
// Postcond:  levels > 0; stCount > 0; stCount < nSamp (weak).
//
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

void SPReg::Factory(int _levelMax) {
  SplitPred::Factory(_levelMax);
  splitPred = new SPReg();

  // Could diminish this somewhat (~ 30%) by replacing 'nSamp' with maximum bag-count
  // value observed over all trees.  Would require precomputation of all sample sizes, though.
  // On a per-tree basis, however, can index in increments of local bag-count.
  //
  SPRegFac::Factory();
}

// TODO:  Check the allocation paramaters.
//
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

void SplitPred::TreeInit() {
  for (int i = 0; i < 2 * levelMax * nPred; i++)
    runFlags[i] = false;
}

void SPReg::ReFactory(int _levelMax) {
  levelMax = _levelMax;

  SPRegFac::ReFactory();
}

void SPReg::DeFactory() {
  SplitPred::DeFactory();
  SPRegFac::DeFactory();
}

void SPCtg::Factory(int _levelMax, int _ctgWidth) {
  SplitPred::Factory(_levelMax);
  splitPred = new SPCtg();

  ctgWidth = _ctgWidth;
  SPCtgNum::Factory();
  SPCtgFac::Factory();
}

void SPCtg::ReFactory(int _levelMax) {
  levelMax = _levelMax;

  SPCtgNum::ReFactory();
  SPCtgFac::ReFactory();
}

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

void SPCtg::DeFactory() {
  SplitPred::DeFactory();
  SPCtgNum::DeFactory();
  SPCtgFac::DeFactory();
}

void SPRegFac::Factory() {
  if (nFacTot > 0)
    FacRun::Factory(levelMax, nPredFac, nFacTot);
}

// N.B.:  Assumes 'levelMax' has been reset further upstream.
//
void SPRegFac::ReFactory() {
  if (nFacTot > 0)
    FacRun::ReFactory(levelMax);
}

void SPCtgFac::Factory() {
  if (nPredFac > 0) {
    FacRunCtg::Factory(levelMax, nPredFac, nFacTot, ctgWidth);
  }
}

// N.B.  Assumes 'levelMax' has been reset further upstream.
//
void SPCtgFac::ReFactory() {
  if (nFacTot > 0)
    FacRunCtg::ReFactory(levelMax);
}

void SPCtgFac::DeFactory() {
  if (nFacTot > 0) {
    FacRunCtg::DeFactory();
  }  
}

void SPCtgFac::TreeInit() {
  if (nFacTot > 0)
    FacRunCtg::TreeInit();
}

void SPCtgFac::ClearTree() {
  if (nFacTot > 0)
    FacRunCtg::ClearTree();
}

void SPCtgNum::Factory() {
  if (nPredNum > 0) {
    ctgSumR = new double[ctgWidth * levelMax * nPredNum];
  }
}

void SPCtgNum::ReFactory() {
  if (nPredNum > 0) {
    delete [] ctgSumR;

    ctgSumR = new double[ctgWidth * levelMax * nPredNum];
  }
}

void SPCtgNum::DeFactory() {
  ctgWidth = -1;
  if (nPredNum > 0) {
    delete [] ctgSumR;
    ctgSumR = 0;
  }
}

void SPRegFac::DeFactory() {
  if (nFacTot > 0)
    FacRun::DeFactory();
}

// For now, invoked once per level.  Can block in larger chunks if this
// proves too slow, but local scope is nice.
//
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

// Unsets the run bit for the split/pred pair at this 'level' value and resets it
// for the progenitor pairs at the next level.
//
// Previous level always wiped, so final level remains dirty:  tree reset required.
//
void SplitPred::TransmitRun(int splitIdx, int predIdx, int splitL, int splitR, int level) {
  SetPredRun(splitIdx, predIdx, level, false);
  if (splitL >= 0)
    SetPredRun(splitL, predIdx, level+1, true);
  if (splitR >= 0)
    SetPredRun(splitR, predIdx, level+1, true);
}

// Sets the run bit to 'val' for the split/pred pair at level 'level'.
//
void SplitPred::SetPredRun(int splitIdx, int predIdx, int level, bool val) {
  bool *base = runFlags + (nPred * levelMax) * (level & 1);
  base[levelMax * predIdx + splitIdx] = val;
}

// Determines whether the split/pred pair is marked as a run this 'level'.
//
bool SplitPred::PredRun(int splitIdx, int predIdx, int level) {
  bool *base = runFlags + (nPred * levelMax) * (level & 1);
  return base[predIdx * levelMax + splitIdx];
}

// Returns true iff the split/pred pair is neither in the pred-prob rejection region nor
// defines a run at this 'level'.
//
bool SplitPred::Splitable(int splitIdx, int predIdx, int splitCount, int level) {
  return splitFlags[splitCount * predIdx + splitIdx] && !PredRun(splitIdx, predIdx, level);
}

// Walks the (#preds x # splitCount) split results and saves the state for the maxima at this
// level.  State is passed to the index tree via 'levelSplitSig[]', which recieves a
// reference to the information in the node's SplitSig.
//
// Factor LHS data resides with nodes so is overwritten at the next
// level.  As factor-valued splits for current level are final at this point,
// it makes sense to register these now for use by the decision tree.
//
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
    if (numL != 0) // ASSERTION
      cout << "Row runs do not sum to sample count:  " << numL << endl;

    if (lhSup >= 0) {
      SplitSig::WriteNum(splitIdx, predIdx, level, lhSampCt, lhSup + 1 - start, maxGini);
    }
  }
}

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

void SplitPred::Level(int splitCount, int level) {
  ProbSplitable(splitCount);
  splitPred->LevelReset(splitCount);
  if (level == 0)
    splitPred->LevelZero();
  else
    splitPred->RestageAndSplit(splitCount, level);
}

// Initializes the auxilliary data structures associated with all predictors
// for every node live at this level.
//
// N.B.:  The numeric/regression case uses no auxilliary structures.
//
void SPReg::LevelReset(int splitCount) {
  if (nFacTot > 0)
    FacRun::LevelReset(splitCount);
}

void SPCtg::LevelReset(int splitCount) {
  if (nPredNum > 0) {
    SPCtgNum::LevelResetSumR(splitCount);
  }
  if (nFacTot > 0)
    FacRunCtg::LevelReset(splitCount);
}

void SPCtgNum::LevelResetSumR(int splitCount) {
  for (int i = 0; i < nPredNum * ctgWidth * levelMax; i++)
    ctgSumR[i] = 0.0;
}

// N.B.:  Relies on Predictor's having ordered like-typed predictors in blocks.
//
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

// N.B.:  Relies on Predictor's having ordered like-typed predictors in blocks.
//
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

// N.B.:  Relies on Predictor's having ordered like-typed predictors in blocks.
//
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

void SPCtgFac::SplitGini(int predIdx, int splitCount, int level) {
  SamplePred *samplePred = SamplePred::BufferOff(predIdx, level);
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    if (!Splitable(splitIdx, predIdx, splitCount, level))
      continue;

    int start, end, dummy;
    double sum, maxGini;
    IndexNode::SplitFields(splitIdx, start, end, dummy, sum, maxGini);

    int top = BuildRuns(samplePred, splitIdx, predIdx, start, end);
    int argMax = SplitRuns(splitIdx, predIdx, sum, top, maxGini);

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

// Retains local sumR values until a transition is noted.  On each transition, pushes
// pair consisting of local factor value (rank) and mean-Y onto node's heap.
// Pushes one more time at conclusion, to catch each node's final factor/mean-Y pair.
//
// N.B.:  Only the Transition() method is specific to FacRunCtg.  All other actions
// involving the heap can be implemented via FacRun.
//
// Returns depth of run vector.
//
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


// Nodes are now represented compactly as a collection of runs.
// For each node, subsets of these collections are examined, looking for the
// Gini argmax beginning from the pre-bias.
//
// Iterates over nontrivial subsets, coded by integers as bit patterns.  If the
// full factor set is not present, then all 'facCount' factors may participate
// in the split.  A practical limit of 2^10 trials is employed.  Hence a node
// with more than 11 distinct factors requires random sampling:  selects 1024
// full-width sequences with bits set ~Bernoulli(0.5).
//
int SPCtgFac::SplitRuns(int splitIdx, int predIdx, double sum, int &top, double &maxGini) {
  int pairOffset = FacRun::PairOffset(splitIdx, predIdx);
  top = FacRunCtg::Shrink(pairOffset, top);
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

//
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

// Must sort runs on the fly, ordering by mean.
//
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

// Causes a SplitSig to be filled in, when appropriate.
//
// BHeap sorts factors by mean-Y over node.  Gini scoring can be done by run, as
// all factors within a run have the same predictor pseudo-value.  Individual run
// members must be noted, however, so that node LHS can be identifed later.
//
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
    if (numR == 0) // ASSERTION.  Will lead to division by zero.
      cout << "All samples seen but " << depth - slot - 1 <<  " heap slots remain" << endl;

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
  if (sCountL != sCountTot)
    cout << "Incomplete sample coverage" << endl;
  if (idxCount != idxCountTot)
    cout << "Incomplete index coverage:  " << idxCount << " != " << idxCountTot << endl;
  return lhTop;
}

