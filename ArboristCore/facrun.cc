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
#include "callback.h"

// Testing only:
//#include <iostream>
using namespace std;

/* FacRun values only during a single level, from argmax pass one (splitting)
   through argmax pass two.  Unless the implementation changes to support
   splitting/argmax on multiple trees concurrently, then, static
   factories should suffice.
*/

int FacRun::nCardTot = -1;
int FacRun::nPredFac = -1;
int FacRun::predFacFirst = -1;

int *FacRunCtg::wideOffset = 0;
int FacRunCtg::ctgWidth = -1;
int FacRunCtg::totalWide = -1; // Set on simulation.

/**
   @brief Fires off initializations.

   @param _nPredFac is the number of factor-valued predictors.

   @param _cardTot is the sum of cardinalities of all factor-valued predictors.

   @param _predFacFirst is the index of the first factor-valued predictor.

   @return void.
 */
void FacRun::Immutables(int _nPredFac, int _cardTot, int _predFacFirst) {
  nCardTot = _cardTot;
  nPredFac = _nPredFac;
  predFacFirst = _predFacFirst;
}

void FacRun::DeImmutables() {
  nCardTot = -1;
  nPredFac = -1;
  predFacFirst = -1;
}


/**
   @brief Constructor.

   @return void.
 */
FacRun::FacRun() {}


/**
   @brief Constructor.
 */
FacRunReg::FacRunReg() {
  LevelInit(1);
}


void FacRunReg::LevelInit(int splitCount) {
  FacRun::LevelInit(splitCount);
  bHeap = new BHeap(splitCount);
}


void FacRunReg::LevelClear() {
  FacRun::LevelClear();
  delete bHeap;
  bHeap = 0;
}


BHeap::BHeap(int _splitCount) {
  splitCount = _splitCount;

  int vacCount = splitCount * FacRun::nPredFac;
  vacant = new unsigned int[vacCount];
  for (int i = 0; i < vacCount; i++)
    vacant[i] = 0;
  bhPair = new BHPair[splitCount * FacRun::nCardTot];
}


BHeap::~BHeap() {
  delete [] vacant;
  delete [] bhPair;
}


/**
   @brief Resets all fields for FacRuns potentially used in the upcoming level.

   Exposes the internals of method PairOffset() for efficient traversal.

   @param splitCount is the number of splits in the current level.

   @return void.
*/
void FacRun::LevelInit(int _splitCount) {
  splitCount = _splitCount;
  levelFR = new FRNode[splitCount * nCardTot];
  levelFac = new int[splitCount * nCardTot];
  for (int predIdx = predFacFirst; predIdx < predFacFirst + nPredFac; predIdx++) {
    int facCard = Predictor::FacCard(predIdx);
    int predOff = Predictor::FacOffset(predIdx) * splitCount;
    for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {      
      FRNode *base = levelFR + predOff + splitIdx * facCard;
      for (int fac = 0; fac < facCard; fac++) {
	FRNode *fr = base + fac;
	fr->start = fr->end = fr->sCount = -1;
	fr->sum = 0.0;
      }
    }
  }
}


void FacRun::LevelClear() {
  delete [] levelFR;
  delete [] levelFac;
  levelFR = 0;
  levelFac = 0;
}


/**
   @brief Constructor.
 */
FacRunCtg::FacRunCtg() {
  LevelInit(1);
}


/**
   @brief Resets the sum vector and replenishes 'rvWide' with new random variates.

   @pram splitCount is the count of splits in the current level.

   @return void.
*/
void FacRunCtg::LevelInit(int _splitCount) {
  FacRun::LevelInit(_splitCount);

  facCtgSum = new double[splitCount * nCardTot * ctgWidth];
  for (int i = 0; i < splitCount * nCardTot * ctgWidth; i++)
      facCtgSum[i] = 0.0;
  if (totalWide > 0) {
    rvWide = new double[splitCount * totalWide];
    int levelWide = splitCount * totalWide;
    CallBack::RUnif(levelWide, rvWide);
  }
}


void FacRunCtg::LevelClear() {
  FacRun::LevelClear();
  delete [] facCtgSum;
  if (totalWide > 0) {
    delete [] rvWide;
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
unsigned int FacRunCtg::Shrink(int splitIdx, int predIdx, unsigned int depth, int facOrd[]) {
  // The first rv for this pair is used to locate an arbitrary position
  // in [0, depth-1].  The remaining rv's are used to select up to
  // 'maxWidthDirect'-many indices out of 'depth' to retain.
  //
  // The indices are walked beginning from the arbitrary position to the
  // top, then looping around from zero, until up to 'maxWidthDirect' are
  // selected.  Unselected indices are marked with a negative value and
  // shrunken out in a separate pass.
  //
  int rvOffset = WideOffset(splitIdx, predIdx);
  int startIdx = rvWide[rvOffset] * (depth - 1);
  double *rvBase = &rvWide[rvOffset + 1];
  
  unsigned int selected = 0;
  double thresh = double(maxWidthDirect) / depth;
  for (unsigned int idx = startIdx; idx < depth; idx++) { // Loop to top.
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
  for (unsigned int idx = 0; idx < depth; idx++) { // Source index of copy.
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
  for (int predIdx = predFacFirst; predIdx < predFacFirst + nPredFac; predIdx++) {
    int facIdx = predIdx - predFacFirst;
    int width = Predictor::FacCard(predIdx);
    if (width > maxWidthDirect) {
      wideOffset[facIdx] = wideOff;
      wideOff += width + 1;
    }
    else
      wideOffset[facIdx] = -1;
  }

  return wideOff;
}


/**
   @brief Invokes base class factory and lights off class specific initializations.

   @param _nPred is the number of predictors.

   @param _nPredFac is the number of factor-valued predictors.

   @param _nCardTot is the sum of cardinalities of all factor-valued predictors.

   @param _predFacFirst is the index of the first factor-valued predictor.

   @param _ctgWidth is the response cardinality.

   @return void.
 */
void FacRunCtg::Immutables(int _nPred, int _nPredFac, int _nCardTot, int _predFacFirst, int _ctgWidth) {
  FacRun::Immutables(_nPredFac, _nCardTot, _predFacFirst);
  ctgWidth = _ctgWidth;
  wideOffset = new int[nPredFac];
  totalWide = SetWideOffset();
}


/**
   @brief Restoration of class immutables to static default values.

   @return void.
 */
void FacRunCtg::DeImmutables() {
  ctgWidth = -1;
  totalWide = -1;
  delete [] wideOffset;
  wideOffset = 0;
  
  FacRun::DeImmutables();
}

