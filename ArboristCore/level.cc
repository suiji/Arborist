/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <iostream>
using namespace std;

#include "train.h"
#include "level.h"
#include "splitsig.h"
#include "facrun.h"
#include "predictor.h"

// Computes the number of levels, 'levels', in the tree.
// Computes the maximum node index, 'infNonAccum', as a function of 'minHeight'.
// 'minHeight' is the smallest node size for which a root node is to be split.
// Hence nodes rooted at indices 'infNonAccum' or higher are not split.
//
// Precond: nSamp > 0; minHeight >= 3, minHeight <= nSamp.
// Postcond:  levels > 0; stCount > 0; stCount < nSamp (weak).
//

int Level::bagCount = -1;
int Level::stCount = -1;
int *Level::treeIdx = 0;

SampleOrd *LevelReg::sampleOrdWS = 0;
SampleOrdCtg *LevelCtg::sampleOrdCtgWS = 0;

int *LevelCtgFac::wideOffset = 0; // Set on simulation.
double *LevelCtgFac::rvWide = 0; // Reset on level
int LevelCtgFac::totalWide = -1; // Set on simulation.

double *LevelCtgNum::ctgSumR = 0;
int LevelCtgNum::ctgWidth = -1;

int LevelCtgFac::ctgWidth = -1;

void Level::Factory(int _nSamp, int _stCount) {
  nSamp = _nSamp;
  stCount = _stCount;
}

void Level::ReFactory(int _stCount) {
  stCount = _stCount;
}

void Level::FactoryReg(int _nSamp, int _stCount) {
  Factory(_nSamp, _stCount);
  int nPred = Predictor::NPred();

  sampleOrdWS = new SampleOrd[2 * nPred * _nSamp];
  // Could diminish this somewhat (~ 30%) by replacing 'nSamp' with maximum bag-count
  // value observed over all trees.  Would require precomputation of all sample sizes, though.
  // On a per-tree basis, however, can index in increments of local bag-count.
  //
  LevelRegNum::Factory();
  LevelRegFac::Factory();
}

void Level::ReFactoryReg(int _stCount) {
  ReFactory(_stCount);
  LevelRegNum::ReFactory();
  LevelRegFac::ReFactory();
}

void Level::DeFactoryReg() {
  DeFactory();
  delete [] sampleOrdWS;
  sampleOrdWS = 0;

  LevelRegNum::DeFactory();
  LevelRegFac::DeFactory();
}

int Level::FactoryCtg(int _nSamp, int _stCount, int _ctgWidth) {
  int nPred = Predictor::NPred();
  Factory(_nSamp, _stCount);

  sampleOrdCtgWS = new SampleOrdCtg[2 * nPred * _nSamp];
  LevelCtgNum::Factory(_ctgWidth);
  int auxSize = LevelCtgFac::Factory(_ctgWidth);

  return auxSize;  
}

void Level::ReFactoryCtg(int _stCount) {
  ReFactory(_stCount);
  LevelCtgNum::ReFactory();
  LevelCtgFac::ReFactory();
}

void Level::DeFactory() {
  stCount = -1;
}

void Level::DeFactoryCtg() {
  DeFactory();
  delete [] sampleOrdCtgWS;
  sampleOrdCtgWS = 0;

  LevelCtgNum::DeFactory();
  LevelCtgFac::DeFactory();
}

void LevelRegNum::Factory() {
  int nPredNum = Predictor::NPredNum();
  if (nPredNum > 0) {
    //predRegNum = new LevelRegNum[stCount * nPredNum];
  }
}

// N.B.:  Assumes 'stCount' has been reset further upstream.
void LevelRegNum::ReFactory() {
  int nPredNum = Predictor::NPredNum();
  if (nPredNum > 0) {
    //    delete [] predRegNum;
    //predRegNum = new LevelRegNum[stCount * nPredNum];
  }
}

void LevelRegFac::Factory() {
  int nPredFac = Predictor::NPredFac();
  if (nPredFac > 0) {
    //    predRegFac = new LevelRegFac[stCount * nPredFac];
    FacRun::Factory(stCount);
  }
}

// N.B.:  Assumes 'stCount' has been reset further upstream.
//
void LevelRegFac::ReFactory() {
  int nPredFac = Predictor::NPredFac();
  if (Predictor::nPredFac > 0) {
    //    delete [] predRegFac;
    //predRegFac = new LevelRegFac[stCount * nPredFac];
    FacRun::ReFactory(stCount);
  }
}

// Returns size of auxiliary vector, if needed.
//
int LevelCtgFac::Factory(const int _ctgWidth) {
  ctgWidth = _ctgWidth;
  int nPredFac = Predictor::NPredFac();
  if (Predictor::nPredFac > 0) {
    //predRegFacCtg = new LevelCtgFac[stCount * nPredFac];
    wideOffset = new int[nPredFac];
    SetWide();
    FacRunCtg::Factory(stCount, ctgWidth);
  }
  return totalWide;
}

// N.B.  Assumes 'stCount' has been reset further upstream.
//
void LevelCtgFac::ReFactory() {
  int nPredFac = Predictor::NPredFac();
  if (nPredFac > 0) {
    //    delete [] predRegFacCtg;

    //predRegFacCtg = new LevelCtgFac[stCount * nPredFac];
    FacRunCtg::ReFactory(stCount);
  }
}

void LevelCtgFac::DeFactory() {
  if (Predictor::NPredFac() > 0) {
    delete [] wideOffset;
    //    delete [] predRegFacCtg;
    wideOffset = 0;
    //    predRegFacCtg = 0;
    FacRunCtg::DeFactory();
  }  
}

void Level::TreeInit(bagCount) {
  targIdx = new int[bagCount * nPred];
}

void Level::ClearTree() {
  delete[] targIdx;
  targIdx = 0;
}

void LevelCtgFac::TreeInit(double *_rvWide) {
  rvWide = _rvWide;
}

void LevelCtgFac::ClearTree() {
  rvWide = 0;
}

void LevelCtgFac::SetWide() {
  int wideOff = 0;
  for (int facIdx = 0; facIdx < Predictor::NPredFac(); facIdx++) {
    int width = Predictor::FacWidth(facIdx);
    if (width > maxWidthDirect) {
      wideOffset[facIdx] = wideOff;
      wideOff += width;
    }
    else
      wideOffset[facIdx] = -1;
  }
  totalWide = wideOff;
}


void LevelCtgNum::Factory(const int _ctgWidth) {
  ctgWidth = _ctgWidth;
  int nPredNum = Predictor::NPredNum();
  if (nPredNum > 0) {
    //    predRegNumCtg = new LevelCtgNum[stCount * nPredNum];
    ctgSumR = new double[ctgWidth * stCount * nPredNum];
  }
}

void LevelCtgNum::ReFactory() {
  int nPredNum = Predictor::NPredNum();
  if (nPredNum > 0) {
    //delete [] predRegNumCtg;
    delete [] ctgSumR;

    //    predRegNumCtg = new LevelCtgNum[stCount * nPredNum];
    ctgSumR = new double[ctgWidth * stCount * nPredNum];
  }
}

void LevelCtgNum::DeFactory() {
  ctgWidth = -1;
  if (Predictor::NPredNum() > 0) {
    delete [] ctgSumR;
    //delete [] predRegNumCtg;
    ctgSumR = 0;
    // predRegNumCtg = 0;
  }
}

void LevelRegNum::DeFactory() {
  if (Predictor::NPredNum() > 0) {
    //delete [] predRegNum;
    //predRegNum = 0;
  }
}

void LevelRegFac::DeFactory() {
  if (Predictor::NPredFac() > 0) {
    FacRun::DeFactory();
  }  
}

// Walks the (#preds x # accum) split results and saves the state for the maxima at this
// level.  State is passed to the decision tree via 'levelSplitSig[]', which recieves a
// reference to the information in the node's SplitSig.
//
// Factor LHS data resides with nodes so is overwritten at the next
// level.  As factor-valued splits for current level are final at this point,
// it makes sense to register these now for use by the decision tree.
//

// Looping over orders, 'i'.  Populates non-root levels by 'y' values, sorted
// according to predictor order.  Target locations given by the tree-defining
// permutation of the corresponding row number.
//
void LevelRegNum::Split(int predIdx, const Node *node, int liveCount, int level) {
  SplitSigNum *splitBase = SplitSigNum::SplitBase(predIdx);
  SampleOrd *sampleOrd = sampleOrdWS + SampleOff(predIdx, level);

  for (int liveIdx = 0; liveIdx < liveCount; liveIdx++) {
    if (!Train::Splitable(predIdx, liveIdx))
      continue;

  // Copies each sampled 'y' value into the appropriate node at each level according
  // to the 'sampleIdx' scheme described below.
  // 'y' is fetched in order of increasing predictor rank.  Values at some rank indices
  // are fetched more than once, others not at all, depending on the random sampling which
  // defines the tree.  The actual ranks are not needed, so remain implicit.
  //
  // Walks samples backward from the end of nodes so that ties are not split.
  //
    Node lAcc = node[liveIdx];
    int start = lAcc.lhStart;
    int end =  start + lAcc.idxCount - 1;
    int sCount = lAcc.sCount;
    double sumR = 0.0;
    SplitSigNum ssn;

    ssn.lhEdge = -1;
    ssn.Gini = lAcc.preBias;
    // TODO:  Initialize state so that initial (rightmost) iteration is avoided:
    // This would eliminate the test "numR == 0", below.
    for (int i = end; i >= start; i--) {
      int numL = sCount; // >= 1:  counts up to and including this index.
      sCount  = numL - sampleOrd[i].rowRun;
      FPTYPE yVal = sampleOrd[i].yVal;

       // ASSERTION:
      //      if (lhEdge < 0)
      //cout << predIdx << " [" << accumIdx << "] edge:  " << lhEdge << endl;

      // These nodes should be 'double' or wider in order to handle "larger" numbers
      // of rows, etc.  There is no noticeable effect on timing when hard-coded to replace
      // FPTYPE.
      //
      int numR = lAcc.sCount - numL;
      double sumL = lAcc.sum - sumR;
      FPTYPE idxGini = numR == 0 ? 0.0 : (sumL * sumL) / numL + (sumR * sumR) / numR;

      // Passing this test results in a splitting state which persists until the node
      // has been walked completely.  The state may be revised if a new maximum is attained,
      // but left-branch samples are recorded beginning with the first nonzero 'lhEdge' value
      // encountered.  The consumer ultimately ignores "old" splitting information, as it
      // uses only the leftmost 'lhEdge' encountered.
      //
      // The rightmost index never passes the test, so there is no danger of splitting at the
      // right end of a node.
      //
      if (idxGini > ssn.Gini) {
	if (sampleOrd[i].rank != sampleOrd[i+1].rank) {
	  ssn.sCount = numL; // Computable on replay.
	  ssn.lhEdge = i - start;
	  ssn.Gini = idxGini;
	  ssn.spLow = sampleOrd[i].rank; // Computable on replay.
	  ssn.spHigh = sampleOrd[i+1].rank; // " " 
	}
	//cout << accumIdx << ": " << idx << " Gini: " << idxGini << ", " << sumL << ", " << numL << ", " << sumR << ", " << numR << endl;
      }
      sumR += yVal;
      // postcond:  0 <= idx < lhStart
    }
    if (ssn.Gini > lAcc.preBias)
      splitBase[liveIdx] = ssn;
  }
}

void LevelCtgNum::Split(int predIdx, const NodeCtg *node, int liveCount, int level) {
  double *ctgSubAccum = ctgSumR + predIdx * stCount * ctgWidth;
  SplitSigNum *splitBase = SplitSigNum::SplitBase(predIdx);
  SampleOrdCtg *sampleOrdCtg = sampleOrdCtgWS + SampleOff(predIdx, level);

  for (int liveIdx = 0; liveIdx < liveCount; liveIdx++) {
    if (!Train::Splitable(predIdx, liveIdx))
      continue;
    NodeCtg lAcc = node[liveIdx];
    int start = lAcc.lhStart;
    int end = start + lAcc.idxCount - 1;
    int numL  = lAcc.sCount;
    double sum = lAcc.sum;
    double numeratorL = lAcc.sumSquares;
    double numeratorR = 0.0;
    double sumR = 0.0;
    SplitSigNum ssn;
    ssn.Gini = lAcc.preBias;

    // TODO:  Ensure far right end does not enter Gini value test.
    for (int i = end; i >= start; i-- ) {
      double sumL = sum - sumR;
      FPTYPE idxGini = (sumL > minDenom && sumR > minDenom) ? numeratorL / sumL + numeratorR / sumR : 0.0;
      // Far-right element does not enter test:  sumR == 0.0, so idxGini = 0.0.
      if (idxGini > ssn.Gini) {
	if (sampleOrdCtg[i].rank !=  sampleOrdCtg[i+1].rank) {
	  ssn.lhEdge = i - start;
	  ssn.sCount = numL;
	  ssn.spLow = sampleOrdCtg[i].rank;
	  ssn.spHigh = sampleOrdCtg[i+1].rank;
	  ssn.Gini = idxGini;
	}
      }
      numL -= sampleOrdCtg[i].rowRun;
      FPTYPE yVal = sampleOrdCtg[i].yVal;
      int sampleIdx = sampleOrdCtg[i].sampleIdx;
      int yCtg = sampleOrdCtg[i].ctg;
      int ctgOff = liveIdx * ctgWidth + yCtg;

    // Suggested by Liaw's version.  Numerical stability?
      double sumRCtg = ctgSubAccum[ctgOff]; // Sum of proxy values at category, strictly to right.
      double sumLCtg = NodeCtg::ctgSum[ctgOff] - sumRCtg; // Sum to left, inclusive.

      // Numerator, denominator wraparound values for next iteration
      numeratorR += yVal * (yVal + 2.0 * sumRCtg); // Gini computation uses preupdated value.
      numeratorL += yVal * (yVal - 2.0 * sumLCtg);

      // Numerators do not use updated sumR/sumL values, so update is delayed until now:
      ctgSubAccum[ctgOff] = sumRCtg + yVal;

      // ASSERTION:
      //      if (lhEdge < 0)
      //cout << predIdx << " [" << accumIdx << "] edge:  " << lhEdge << endl;

      // Jittering should greatly reduce possibility of ties.
      // Only bothers to compute giniLocal if both denominators > 1.0e-5
      sumR += yVal;
    }
    if (ssn.Gini > lAcc.preBias) {
      splitBase[liveIdx] = ssn;
    }
    // postcond:  0 <= idx < lhStart
  }
}

void Level::Level(int liveCount, int level) {
  SplitSig::LevelReset(liveCount);
  predNode->LevelReset(liveCount);
  if (level == 0)
    predNode->LevelZero(liveCount);
  else
    predNode->RestageAndSplit(liveCount, level);
}

// Initializes the auxilliary data structures associated with all predictors
// for every node live at this level.
//
// N.B.:  The numeric/regression case uses no auxilliary structures.
//
void LevelReg::LevelReset(int liveCount) {
  if (Predictor::NPredFac() > 0)
    FacRun::LevelReset(liveCount);
}

void LevelCtg::LevelReset(int liveCount) {
  for (int i = 0; i < Predictor::NPredNum() * ctgWidth * liveCount; i++)
    ctgSumR[i] = 0.0;

  if (Predictor::NPredFac() > 0)
    FacRunCtg::LevelReset(liveCount);
}
/*
  for (int facIdx = 0; facIdx < Predictor::NPredFac(); facIdx++) {
    for (int off = 0; off < liveCount; off++) {
      // Clears each FacRun associated with this index pair.  This enables
      // unused FacRuns to be excluded from the count of active factors in
      // the node.
      //
      FacRunCtg::Reset(facIdx, off);
    }
  }
  */

// N.B.:  Relies on Predictor's having ordered like-typed predictors in blocks.
//
void LevelReg::LevelZero(int liveCount) {
    int predIdx;
    int predNumFirst = Predictor::PredNumFirst();
    int predNumSup = Predictor::PredNumSup();
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
	DataOrd::PredByRank(predIdx, sample, sampleOrdWS + SampleOff(predIdx, 0));
	LevelRegNum::Split(predIdx, node, liveCount, 0);
      }
    }

   int predFacFirst = Predictor::PredFacFirst();
   int predFacSup = Predictor::PredFacSup();

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	DataOrd::PredByRank(predIdx, sample, sampleOrdWS + SampleOff(predIdx, 0));
	LevelRegFac::Split(predIdx, node, liveCount, 0);
      }
    }
}

// N.B.:  Relies on Predictor's having ordered like-typed predictors in blocks.
//
void LevelCtg::LevelZero(int liveCount) {
    int predIdx;
    int predNumFirst = Predictor::PredNumFirst();
    int predNumSup = Predictor::PredNumSup();
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
	DataOrd::PredByRank(predIdx, sampleCtg, sampleOrdCtgWS + SampleOff(predIdx, 0));
	LevelRegNum::Split(predIdx, node, liveCount, 0);
      }
    }

   int predFacFirst = Predictor::PredFacFirst();
   int predFacSup = Predictor::PredFacSup();

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	DataOrd::PredByRank(predIdx, sampleCtg, sampleOrdCtgWS + SampleOff(predIdx, 0));
	LevelRegFac::Split(predIdx, node, liveCount, 0);
      }
    }
}

// N.B.:  Relies on Predictor's having ordered like-typed predictors in blocks.
//
void LevelReg::RestageAndSplit(int liveCount, int level) {
    int predIdx;
    int predNumFirst = Predictor::PredNumFirst();
    int predNumSup = Predictor::PredNumSup();
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
	NodeCache::Restage(predIdx, level-1);
	LevelRegNum::Split(predIdx, node, liveCount, level);
      }
    }

   int predFacFirst = Predictor::PredFacFirst();
   int predFacSup = Predictor::PredFacSup();

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	NodeCache::Restage(predIdx, level-1);
	LevelRegFac::Split(predIdx, node, liveCount, level);
      }
    }
}

void LevelCtg::RestageAndSplit(int liveCount, int level) {
  int predIdx;
  int predNumFirst = Predictor::PredNumFirst();
  int predNumSup = Predictor::PredNumSup();

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
	NodeCache::Restage(predIdx, level-1);
	LevelCtgNum::Split(predIdx, node, liveCount, level);
      }
    }

   int predFacFirst = Predictor::PredFacFirst();
   int predFacSup = Predictor::PredFacSup();

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	NodeCache::Restage(predIdx, level-1);
	LevelCtgFac::Split(predIdx, node, liveCount, level);
      }
    }
}

// Reassembles the next level's sampleOrd[] for unit-stride access, by node.
// At least one of 'lhNext', 'rhNext' is guaranteed to be live.
//
void LevelReg::Restage(int predIdx, int level, int startIdx, int idxCount, int lhNext, int rhNext) {
  SampleOrd *source = sampleOrdWS + SampleOff(predIdx, level);
  SampleOrd *targ = sampleOrdWS + SampleOff(predIdx, level+1);

  int *targOff = targIdx + predIdx * bagCount;
  int end = startIdx + idxCount - 1;
  for (int i = startIdx; i <= end; i++) {
    int sIdx = source[i].sampleIdx;
    targOff[i] = sample2Node[sIdx]; // Irregular read
  }

  if (lhNext >= 0 && rhNext >= 0) {
    int lhIdx = NodeStart(lhNext);
    int rhIdx = NodeStart(rhNext);
    // Target nodes should all be either lh or rh.
    for (int i = startIdx; i <= end; i++) {
      int targAccum = targOff[i];
      if (targAccum == lhNext)
	targ[lhIdx++] = source[i];
      else
	targ[rhIdx++] = source[i];
    }
  }
  else if (lhNext >= 0) {
    int lhIdx = NodeStart(lhNext);
    for (int i = startIdx; i <= end; i++) {
      int targAccum = targOff[i];
      if (targAccum >= 0)
	targ[lhIdx++] = source[i];
    }
  }
  else {
    int rhIdx = NodeStart(rhNext);
    for (int i = startIdx; i <= end; i++) {
      int targAccum = targOff[i];
      if (targAccum >= 0)
	targ[rhIdx++] = source[i];
    }
  }
}

void LevelCtg::Restage(int predIdx, int level, int startIdx, int idxCount, int lhNext, int rhNext) {
  int sourcePos = (lev & 1) > 0 ? 1 : 0;
  SampleOrdCtg *source = sampleOrdCtgWS + SampleOff(predIdx, level);
  SampleOrdCtg *targ = sampleOrdCtgWS + SampleOff(predIdx, level+1);

  int *targOff = targIdx + predIdx * bagCount;
  int end = startIdx + idxCount - 1;
  for (int i = startIdx; i <= end; i++) {
    int sIdx = source[i].sampleIdx;
    targOff[i] = sample2Node[sIdx]; // Irregular read
  }

  if (lhNext >= 0 && rhNext >= 0) {
    int lhIdx = NodeStart(lhNext);
    int rhIdx = NodeStart(rhNext);
    // Target nodes should all be either lh or rh.
    for (int i = startIdx; i <= end; i++) {
      int targAccum = targOff[i];
      if (targAccum == lhNext)
	targ[lhIdx++] = source[i];
      else
	targ[rhIdx++] = source[i];
    }
  }
  else if (lhNext >= 0) {
    int lhIdx = NodeStart(lhNext);
    for (int i = startIdx; i <= end; i++) {
      int targAccum = targOff[i];
      if (targAccum >= 0)
	targ[lhIdx++] = source[i];
    }
  }
  else {
    int rhIdx = NodeStart(rhNext);
    for (int i = startIdx; i <= end; i++) {
      int targAccum = targOff[i];
      if (targAccum >= 0)
	targ[rhIdx++] = source[i];
    }
  }
}

// Resets LHS/RHS nodes for all samples associated with caller's node.
//
// Returns sum of samples associated with the block.
//
double LevelReg::SampleReplay(int predIdx, int level, int start, int count, int id) {
  SampleOrd *tOrd = sampleOrdWS + (2 * predIdx + ((level & 1) > 0 ? 1 : 0)) * nSamp;
  double sum = 0.0;

  for (int i = start; i < start + count; i++) {
    int sIdx = tOrd[i].sampleIdx;
    sum += tOrd[i].yVal;
    sample2Node[sIdx] = id;
  }

  return sum;
}

double LevelCtg::SampleReplay(int predIdx, int level, int start, int count, int id) {
  SampleOrdCtg *tOrd = sampleOrdCtgWS + (2 * predIdx + ((level & 1) > 0 ? 1 : 0)) * nSamp;
  double sum = 0.0;
  for (int i = start; i < start + count; i++) {
    int sIdx = tOrd[i].sampleIdx;
    sum += tOrd[i].yVal;
    sample2Node[sIdx] = id;
  }

  return sum;
}

void LevelCtgFac::Split(int predIdx, const NodeCtg *node, int liveCount, int level) {
  int facIdx = Predictor::FacIdx(predIdx);

  // Retain local sumR values until a transition is noted.  On each transition, push pair consisting of
  // local factor value (rank) and mean-Y onto node's heap.  Perform this push one more time at
  // conclusion, to catch each node's final factor/mean-Y pair.
  // For each node, computes Gini on the fly using each factor's mean-Y value and noting the factors
  // at which successive maxima are found.
  //
  SampleOrdCtg *sampleOrdCtg = sampleOrdCtgWS + SampleOff(predIdx, level);
  for (int liveIdx = 0; liveIdx < liveCount; liveIdx++) {
    if (!Train::Splitable(predIdx, liveIdx))
      continue;

    NodeCtg lAcc = node[liveIdx];
    int start = lAcc.lhStart;
    int end = start + lAcc.idxCount - 1;
    int ord = -1;
    int sCount = 0;
    double sumR = 0.0;

    SplitSigFac *splitBase = SplitSigFac::SplitBase(facIdx);
    double sumLet = 0.0;
    for (int i = end; i >= start; i--) {
      int ordR = ord;
      SampleOrdCtg tOrd = sampleOrdCtg[i];
      int sampleIdx = tOrd.sampleIdx;
      int yCtg = tOrd.ctg;
      FPTYPE yVal = tOrd.yVal; // Sum of y-values for samples at this entry.
      ord = tOrd.rank;  // Rank beginning run of identical predictor values.
      int runSize = tOrd.rowRun; // Total count of rows up to and including this.

      // Can index bh[] by live #.  Caller should make a reasonable number available, reallocating
      // if a high watermark hit.
      //
      if (ord == ordR) { // No transition:  counters accumulate.
	sumR += yVal;
	sCount += runSize;
	FacRunCtg::Terminus(facIdx, liveIdx, ord, i, yCtg, yVal);
      }
      else {
	if (ordR >= 0) { // Run to the right flushed by setting sum, count totals from node.
	  FacRun::Transition(facIdx, liveIdx, ordR, sCount, sumR);
	  FacRun::Insert(facIdx, liveIdx, ordR, sCount, sumR);
	}
	// New run:  node reset and bounds initialized.
	//
	sumR = yVal;
	sCount = runSize;
	FacRunCtg::Terminus(facIdx, liveIdx, ord, i, yCtg, yVal, true);
      }
    }
    // Flushes the remaining runs.
    //
    FacRun::Transition(facIdx, liveIdx, ord, sCount, sumR);
    FacRun::Insert(facIdx, liveIdx, ord, sCount, sumR);

    // It is not necessary to sort the FacRuns.  The heap structure does provide a quick
    // way to compute depth, however, and should have trivial effect on performance.
    // 
    // Nodes are now represented compactly as a collection of runs.
    // For each node, subsets of these collections are examined, looking for the
    // Gini argmax beginning from the pre-bias.
    //
    double levelSum = lAcc.sum;
    double maxGini = lAcc.preBias;

    // Iterates over nontrivial subsets, coded by integers as bit patterns.  If the
    // full factor set is not present, then all 'facCount' factors may participate
    // in the split.  A practical limit of 2^10 trials is employed.  Hence a node
    // with more than 11 distinct factors requires random sampling:  selects 1024
    // full-width sequences with bits set ~Bernoulli(0.5).
    //
    int *facOrd;
    int depth = FacRun::DePop(facIdx, liveIdx, facOrd);
    if (depth <= 1)
      return;

    int keyVal = -1;
    if (depth > maxWidthDirect) {
      int off = wideOffset[facIdx];
      // Samples 2^(maxWidthDirect-1) random subsets.
      for (int sample = 0; sample < (1 << (maxWidthDirect-1)); sample++, off += depth) {
	int hiSlots = 0;
	for (int slot = 0; slot < depth; slot++) {
	  if (rvWide[off+slot] > 0.5)
	    hiSlots++;
	}
	if (hiSlots == depth) // Prescreens accepted slots for full set.
	  continue;

	double sumL = 0.0;
	double numerL = 0.0;
	double numerR = 0.0;
	for (int ctg = 0; ctg < ctgWidth; ctg++) {
	  double sumCtg = 0.0;
	  for (int slot = 0; slot < depth; slot++) {
	    if (rvWide[off+slot] > 0.5) { // Accepts slots with probablity 0.5
	      double *ctgBase = FacRunCtg::Sum(facIdx, liveIdx, facOrd[slot]);
	      sumCtg += ctgBase[ctg];
	    }
	  }
	  double totSum = NodeCtg::ctgSum[liveIdx * ctgWidth + ctg];
	  sumL += sumCtg;
	  numerL += sumCtg * sumCtg;
	  numerR += (totSum - sumCtg) * (totSum - sumCtg);
	}
	double sumR = levelSum - sumL;
	double idxGini = (sumL <= 1.0e-8 || sumR <= 1.0e-5) ? 0.0 : numerR / sumR + numerL / sumL;
	if (idxGini > maxGini) {
	  maxGini = idxGini;
	  keyVal = off;
	}
      }
    }
    else {
      // Iterates over all nontrivial subsets of factors in the node.
      //
      int fullSet = (1 << (depth - 1)) - 1;
      for (int subset = 1; subset <= fullSet; subset++) {
	double sumL = 0.0;
	double numerL = 0.0;
	double numerR = 0.0;
	for (int ctg = 0; ctg < ctgWidth; ctg++) {
	  double sumCtg = 0.0;
	  for (int slot = 0; slot  < depth - 1; slot++) {
	    if ((subset & (1 << slot)) > 0) {
	      double *ctgBase = FacRunCtg::Sum(facIdx, liveIdx, facOrd[slot]);
	      sumCtg += ctgBase[ctg];
	    }
	  }
	  double totSum = NodeCtg::ctgSum[liveIdx * ctgWidth + ctg];
	  sumL += sumCtg;
	  numerL += sumCtg * sumCtg;
	  numerR += (totSum - sumCtg) * (totSum - sumCtg);
	}
	double sumR = levelSum - sumL;
	double idxGini = (sumL <= 1.0e-8 || sumR <= 1.0e-5) ? 0.0 : numerR / sumR + numerL / sumL;
	if (idxGini > maxGini) {
	  maxGini = idxGini;
	  keyVal = subset;
	}
      }
    }

    // TODO:  Subsume into Replay method, as in RbSNP.
    // Reconstructs LHS from 'keyVal'.
    //
    int countL = 0;
    int bufferLeft = -1; // Total # samples up to and including the current ordinal.
    if (keyVal >= 0) { // == 0 useful only if wide
      SplitSigFac::Clear(facIdx, liveIdx);
      if (depth > maxWidthDirect) {
	for (int slot = 0; slot < depth; slot++) {
	  if (rvWide[keyVal + slot] > 0.5) {
	    int lhFac = facOrd[slot];
	    FacRun *fRun = FacRun::Ref(facIdx, liveIdx, lhFac);
	    countL += fRun->sCount;
	    bufferLeft += 1 + fRun->end - fRun->start;
	    SplitSigFac::SingleBit(facIdx, liveIdx, lhFac);
	  }
	}
      }
      else {
	// Reconstructs LHS positions and counts from 'subset' value.
	int subset = keyVal;
	for (int slot = 0; slot  < depth - 1; slot++) {
	  if ((subset & (1 << slot)) > 0) {
	    int lhFac = facOrd[slot];
	    FacRun *fRun = FacRun::Ref(facIdx, liveIdx, lhFac);
	    countL += fRun->sCount;
	    bufferLeft += 1 + fRun->end - fRun->start;
	    SplitSigFac::SingleBit(facIdx, liveIdx, lhFac);
	  }
	}
      }
      SplitSigFac ssf;
      ssf.sCount = countL;
      ssf.lhEdge = bufferLeft;
      ssf.Gini = maxGini;
      ssf.subset = keyVal;
      splitBase[liveIdx] = ssf;
    }
  }
}


// Streamlined version of Split() for level zero, which uses a single node.  Various
// node-based objects can therefore be cached as locals, as the node in use
// never varies.  Init(), moreover, need not be called here.
//
void LevelRegFac::Split(int predIdx, const Node *node, int liveCount, int level) {
  int facIdx = Predictor::FacIdx(predIdx);
  SampleOrd *sampleOrd = sampleOrdWS + SampleOff(predIdx, level);

  for (int liveIdx = 0; liveIdx < liveCount; liveIdx++) {
    if (!Train::Splitable(predIdx, liveIdx))
      continue;

    Node lAcc = node[liveIdx];
    int start = lAcc.lhStart;
    int end = start + lAcc.idxCount -1;

    int ord = -1;
    int sCount = 0;
    double sumR = 0.0; // Wide node:  see comments above.

    for (int i = end; i >= start; i--) {
      int ordR = ord;
      ord = sampleOrd[i].rank;
      FPTYPE yVal = sampleOrd[i].yVal;
      int runSize = sampleOrd[i].rowRun;

      // Can index bh[] by live #.  Caller should make a reasonable number available, reallocating
      // if a high watermark hit.
      //
      if (ord == ordR) { // No transition:  counters accumulate.
	sumR += yVal;
	sCount += runSize;
	FacRun::Terminus(facIdx, liveIdx, ord, i);
      }
      else { // Transition
	if (ordR >= 0) { // Run to the right flushed by setting sum, count totals from node.
	  FacRun::Transition(facIdx, liveIdx, ordR, sCount, sumR);
	  FacRun::Insert(facIdx, liveIdx, ordR, sCount, sumR);
	}

	// New run:  node reset and bounds initialized.
	//
	sumR = yVal;
	sCount = runSize;
	FacRun::Terminus(facIdx, liveIdx, ord, i, true);
      }
    }

    // Flushes the remaining run.
    //
    FacRun::Transition(facIdx, liveIdx, ord, sCount, sumR);
    FacRun::Insert(facIdx, liveIdx, ord, sCount, sumR);

    // BHeap sorts factors by mean-Y over node.  Gini scoring can be done in blocks, as
    // all factors within a block have the same predictor pseudo-value.  Individual block
    // members must be noted, however, so that node LHS can be identifed later.
    //
    int numL = 0;
    double levelSum = lAcc.sum;
    int levelCount = lAcc.sCount;
    double sumL = 0.0;
    int bufferLeft = -1;
    int lhsWidth = 0;
    SplitSigFac ssf;
    ssf.Gini = lAcc.preBias;

    int *facOrd;
    int depth = FacRun::DePop(facIdx, liveIdx, facOrd);
    for (int fc = 0; fc < depth - 1; fc++) {
      FacRun *fRun = FacRun::Ref(facIdx, liveIdx, facOrd[fc]);
      numL += fRun->sCount;
      sumL += fRun->sum;
      int numR = levelCount - numL;
      double sumR = levelSum - sumL;
      bufferLeft += 1 + fRun->end - fRun->start; // Accumulates total of run populations.

      FPTYPE idxGini = numR == 0 ? 0.0 : (sumL * sumL) / numL + (sumR * sumR) / numR;
    //  cout << predIdx << " (" << "0" << "):  " << idxGini << endl;

      if (idxGini > ssf.Gini) {
	ssf.sCount = numL;
	ssf.Gini = idxGini;
      // Would-be edge of LHS:  buffer of sample indices never actually left-justified.
	ssf.lhEdge = bufferLeft;
	lhsWidth = fc + 1;
      }
    }
    if (lhsWidth > 0) {
      SplitSigFac *splitBase = SplitSigFac::SplitBase(facIdx);
      splitBase[liveIdx] = ssf;
      SplitSigFac::FacBits(facIdx, liveIdx, lhsWidth, facOrd);
    }
  }
}
