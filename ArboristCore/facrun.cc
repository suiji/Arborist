// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file facrun.cc

   @brief Methods for maintaining runs of factor-valued predictors during splitting.

   @author Mark Seligman
 */

#include "facrun.h"
#include "samplepred.h"
#include "pretree.h"
#include "callback.h"

// Testing only:
//#include <iostream>
using namespace std;

int *BHeap::vacant = 0;
BHPair *BHeap::bhPair = 0;

FacRun *FacRun::levelFR = 0;
int *FacRun::levelFac = 0;
int FacRun::nCardTot = -1;
int FacRun::nPredFac = -1;
int FacRun::predFacFirst = -1;
int FacRun::levelMax = 0;

double *FacRunCtg::facCtgSum = 0;
double *FacRunCtg::rvWide = 0;
int *FacRunCtg::wideOffset = 0;
int FacRunCtg::ctgWidth = -1;
int FacRunCtg::totalWide = -1; // Set on simulation.

/**
   @brief Fires off initializations.

   @param _levelMax is the current level size, increase of which precipitates reallocations.

   @param _nPredFac is the number of factor-valued predictors.

   @param _cardTot is the sum of cardinalities of all factor-valued predictors.

   @param _predFacFirst is the index of the first factor-valued predictor.

   @return void.
 */
void FacRun::Factory(int _levelMax, int _nPredFac, int _cardTot, int _predFacFirst) {
  nCardTot = _cardTot;
  nPredFac = _nPredFac;
  levelMax = _levelMax;
  predFacFirst = _predFacFirst;
  int vacCount = levelMax * nPredFac;
  BHeap::vacant = new int[vacCount];
  for (int i = 0; i < vacCount; i++)
    BHeap::vacant[i] = 0;

  BHeap::bhPair = new BHPair[levelMax * nCardTot];
  levelFR = new FacRun[levelMax * nCardTot];
  levelFac = new int[levelMax * nCardTot];
}

/**
   @brief Reallocates data structures dependent upon level-max value.

   @param _levelMax is the current level-max value.

   @return void.
*/
void FacRun::ReFactory(int _levelMax) {
  levelMax = _levelMax;
  delete [] BHeap::vacant;
  delete [] BHeap::bhPair;
  delete [] levelFR;
  delete [] levelFac;

  int vacCount = levelMax * nPredFac;
  BHeap::vacant = new int[vacCount];
  for (int i = 0; i < vacCount; i++)
    BHeap::vacant[i] = 0;

  BHeap::bhPair = new BHPair[levelMax * nCardTot];
  levelFR = new FacRun[levelMax * nCardTot];
  levelFac = new int[levelMax * nCardTot];
}

/**
   @brief Deallacations.
   
   @return void.
*/
void FacRun::DeFactory() {
  delete [] BHeap::vacant;
  delete [] BHeap::bhPair;
  delete [] levelFR;
  delete [] levelFac;

  BHeap::vacant = 0;
  BHeap::bhPair = 0;
  levelFR = 0;
  levelFac = 0;
}

/**
   @brief Resets all fields for FacRuns potentially used in the upcoming level.

   Exposes the internals of method PairOffset() for efficient traversal.

   @param splitCount is the number of splits in the current level.

   @return void.
*/
void FacRun::LevelReset(int splitCount) {
  for (int predIdx = Predictor::PredFacFirst(); predIdx < Predictor::PredFacSup(); predIdx++) {
    int facCard = Predictor::FacCard(predIdx);
    int predOff = Predictor::FacOffset(predIdx) * levelMax;
    for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {      
      FacRun *base = levelFR + predOff + splitIdx * facCard ;
      for (int fac = 0; fac < facCard; fac++) {
	FacRun *fr = base + fac;
	fr->start = fr->end = fr->sCount = -1;
	fr->sum = 0.0;
      }
    }
  }
}

/**
   @brief Resets the sum vector and replenishes 'rvWide' with new random variates.

   @pram splitCount is the count of splits in the current level.

   @return void.
*/
void FacRunCtg::LevelReset(int splitCount) {
  FacRun::LevelReset(splitCount);

  for (int i = 0; i < splitCount * nCardTot * ctgWidth; i++)
      facCtgSum[i] = 0.0;
  if (totalWide > 0) {
    int levelWide = splitCount * totalWide;
    CallBack::RUnif(levelWide, rvWide);
  }
}

/**
   @brief Shrinks the contents of the rank vector to 'maxWidthDirect' or less
   by randomly deleting elements. N.B.:  caller ensures that this predictor is wide.

   Uses Bernoulli scheme i/o sampling.  Initializes to random
   spot in vector and walks in a circular fashion, so as to minimize
   bias.  Stops when either the entire vector has been walked or when
   'maxWidthDirect' indices are selected.  Bernoulli has wide variance,
   so there may be undercounting.

   @param splitIdx is the split index.

   @param predIdx is the predictor index.

   @param splitCount is the count of splits in the current level.

   @param depth is the number of elements in the rank vector.

   @param facOrd is the input and output rank vector.

   @return size of rank vector, with output parameter.
*/
int FacRunCtg::Shrink(int splitIdx, int predIdx, int splitCount, int depth, int facOrd[]) {
  // The first rv for this pair is used to locate an arbitrary position
  // in [0, depth-1].  The remaining rv's are used to select up to
  // 'maxWidthDirect'-many indices out of 'depth' to retain.
  //
  // The indices are walked beginning from the arbitrary position to the
  // top, then looping around from zero, until up to 'maxWidthDirect' are
  // selected.  Unselected indices are marked with a negative value and
  // shrunk out in a separate pass.
  //
  int rvOffset = WideOffset(splitIdx, predIdx, splitCount);
  int startIdx = rvWide[rvOffset] * (depth - 1);
  double *rvBase = &rvWide[rvOffset + 1];
  
  int selected = 0;
  double thresh = double(maxWidthDirect) / depth;
  for (int idx = startIdx; idx < depth; idx++) { // Loop to top.
    if (selected == maxWidthDirect)
      break;
    if (rvBase[idx] <= thresh)
      selected++;
    else
      facOrd[idx] = -1;
  }
  for (int idx = 0; idx < startIdx; idx++) { // Loop from bottom.
    if (selected == maxWidthDirect)
      break;
    if (rvBase[idx] <= thresh)
      selected++;
    else
      facOrd[idx] = -1;
  }

  // Shrinks the index vector by moving only positive indices to the
  // next unfilled postion.
  //
  int j = 0; // Destination index of copy.
  for (int idx = 0; idx < depth; idx++) { // Source index of copy.
    int slot = facOrd[idx];
    if (slot >= 0)
      facOrd[j++] = slot;
  }

  return selected;
}

/**
 @brief Sets the RV offsets for the wide-cardinality factors.  Uses one slot
 for each factor value, plus one for entry index.

 @return high watermark of workspace offsets.
*/
int FacRunCtg::SetWideOffset() {
  int wideOff = 0;
  int predIdx;
  for (predIdx = 0; predIdx < predFacFirst; predIdx++)
    wideOffset[predIdx] = -1;

  for (predIdx = predFacFirst; predIdx < predFacFirst + nPredFac; predIdx++) {
    int width = Predictor::FacCard(predIdx);
    if (width > maxWidthDirect) {
      wideOffset[predIdx] = wideOff;
      wideOff += width + 1;
    }
    else
      wideOffset[predIdx] = -1;
  }

  return wideOff;
}

/**
   @brief Invokes base class factory and lights off class specific initializations.

   @param _levelMax is the current level size, increase of which precipitates reallocations.

   @param _nPred is the number of predictors.

   @param _nPredFac is the number of factor-valued predictors.

   @param _nCardTot is the sum of cardinalities of all factor-valued predictors.

   @param _predFacFirst is the index of the first factor-valued predictor.

   @param _ctgWidth is the response cardinality.

   @return void.
 */
void FacRunCtg::Factory(int _levelMax, int _nPred, int _nPredFac, int _nCardTot, int _predFacFirst, int _ctgWidth) {
  ctgWidth = _ctgWidth;
  FacRun::Factory(_levelMax, _nPredFac, _nCardTot, _predFacFirst);
  facCtgSum = new double[_levelMax * _nCardTot * ctgWidth];
  wideOffset = new int[_nPred];
  totalWide = SetWideOffset();
  rvWide = new double[_levelMax * totalWide];
}

/**
   @brief Reallocates data structures dependent on level-max.

   @param _levelMax is the current level-max value.

   @return void.
 */
void FacRunCtg::ReFactory(int _levelMax) {
  FacRun::ReFactory(_levelMax);

  delete [] facCtgSum;
  facCtgSum = new double[_levelMax * nCardTot * ctgWidth];

  delete [] rvWide;
  rvWide = new double[_levelMax * totalWide];
}

/**
   @brief Deallocation of class-specific data structures as well as base class.

   @return void.
 */
void FacRunCtg::DeFactory() {
  delete [] facCtgSum;
  delete [] wideOffset;
  delete [] rvWide;
  facCtgSum = 0;
  wideOffset = 0;
  rvWide = 0;
  ctgWidth = -1;
  totalWide = -1;

  FacRun::DeFactory();
}

/**
   @brief The LHS factors are recovered from the pretree, where they were set when the
   nonterminal was registered.

   @param splitIdx is the split index of the pair.

   @param predIdx is the predictor index of the pair.

   @param level is the zero-based level number under construction.

   @param bitStart is the beginning of the bit encoding for the LH subset.

   @param ptLH is the pretree index of the corresponding left-hand node.

   @param ptRH is the pretree index of the corresponding right-hand node.

   @return sum of response values associated with the left-hand subnode.
*/
double FacRun::Replay(int splitIdx, int predIdx, int level, int bitStart, int ptLH, int ptRH) {
  double lhSum = 0.0;
  FacRun *base = levelFR + PairOffset(splitIdx, predIdx);
  for (int fac = 0; fac < Predictor::FacCard(predIdx); fac++) {
    FacRun *fRun = base + fac;
    if (PreTree::BitVal(bitStart + fac)) {
      lhSum += SamplePred::Replay(predIdx, level, fRun->start, fRun->end, ptLH);
    }
    else if (fRun->sCount > 0) {
      (void) SamplePred::Replay(predIdx, level, fRun->start, fRun->end, ptRH);
    }
  }

  return lhSum;
}
