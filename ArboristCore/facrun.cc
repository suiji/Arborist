// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "facrun.h"
#include "samplepred.h"
#include "pretree.h"
#include "callback.h"

#include <iostream>
using namespace std;

int *BHeap::vacant = 0;
BHPair *BHeap::bhPair = 0;

FacRun *FacRun::levelFR = 0;
int *FacRun::levelFac = 0;
int FacRun::nCardTot = -1;
int FacRun::nPredFac = -1;
int FacRun::levelMax = 0;

double *FacRunCtg::facCtgSum = 0;
double *FacRunCtg::rvWide = 0;
int FacRunCtg::ctgWidth = -1;
int *FacRunCtg::wideOffset = 0; // Set on simulation.
int FacRunCtg::totalWide = -1; // Set on simulation.

void FacRun::Factory(int _levelMax, int _nPredFac, int _cardTot) {
  nCardTot = _cardTot;
  nPredFac = _nPredFac;
  levelMax = _levelMax;
  int vacCount = levelMax * nPredFac;
  BHeap::vacant = new int[vacCount];
  for (int i = 0; i < vacCount; i++)
    BHeap::vacant[i] = 0;

  BHeap::bhPair = new BHPair[levelMax * nCardTot];
  levelFR = new FacRun[levelMax * nCardTot];
  levelFac = new int[levelMax * nCardTot];
}

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

// Resets all fields for FacRuns potentially used in the upcoming level.
//
// Exposes the internals of method PairOffset() for efficient traversal.
//
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

// Shrinks the contents of facOrd[] to specified 'height' by randomly
// deleting elements.
//
// TODO:  Use Bernoulli scheme i/o sampling.  Initialize to random
// spot in vector and walk in a circular fashion.  Stop when either
// the entire vector has been walked or only 'height' - many remain.
// Bernoulli has wide variance, so there may be undercounting.
//
int FacRunCtg::Shrink(int depth, int facOrd[]) {
  int toGo = depth - maxWidthDirect;
  int gone = 0;

  // N.B.:  Assumes that rvWide[] has enough elements remaining to
  // terminate without error.
  while (gone < toGo) {
    int idx = depth * *rvWide++;
    if (facOrd[idx] >= 0) {
      gone++;
      facOrd[idx] = -1;
    }
  }

  int j = 0; // Destination index of copy.
  for (int i = 0; i < depth; i++) { // Source index of copy.
    int slot = facOrd[i];
    if (slot >= 0)
      facOrd[j++] = slot;
  }

  return depth - gone; // Should equal j.
}

int FacRunCtg::SetWide() {
  int wideOff = 0;
  for (int facIdx = 0; facIdx < nPredFac; facIdx++) {
    int width = Predictor::FacCard(facIdx);
    if (width > maxWidthDirect) {
      wideOffset[facIdx] = wideOff;
      wideOff += width;
    }
    else
      wideOffset[facIdx] = -1;
  }

  return wideOff;
}

void FacRunCtg::TreeInit() {
  rvWide = new double[totalWide];
  CallBack::RUnif(totalWide, rvWide);
}

void FacRunCtg::ClearTree() {
  delete [] rvWide;
  rvWide = 0;
}

void FacRunCtg::Factory(int _levelMax, int _nPredFac, int _nCardTot, int _ctgWidth) {
  ctgWidth = _ctgWidth;
  FacRun::Factory(_levelMax, _nPredFac, _nCardTot);
  facCtgSum = new double[_levelMax * _nCardTot * ctgWidth];
  wideOffset = new int[_nPredFac];
  totalWide = SetWide();
}

void FacRunCtg::ReFactory(int _levelMax) {
  FacRun::ReFactory(_levelMax);

  delete [] facCtgSum;
  facCtgSum = new double[_levelMax * nCardTot * ctgWidth];
}

void FacRunCtg::DeFactory() {
  delete [] facCtgSum;
  delete [] wideOffset;
  facCtgSum = 0;
  wideOffset = 0;
  ctgWidth = -1;
  totalWide = -1;

  FacRun::DeFactory();
}

// The LHS factors are recovered from the pretree, where they were set when the
// nonterminal was registered.
//
double FacRun::Replay(int splitIdx, int predIdx, int level, int bitStart, int ptLH, int ptRH) {
  double lhSum = 0.0;
  FacRun *base = levelFR + PairOffset(splitIdx, predIdx);
  for (int fac = 0; fac < Predictor::FacCard(predIdx); fac++) {
    FacRun *fRun = base + fac;
    if (PreTree::BitVal(bitStart + fac)) {
      lhSum += SamplePred::Replay(predIdx, level, fRun->start, fRun->end, ptLH);
      //cout << "Level " << level << " true " << ptLH <<  " onto " << bitStart << " + " << fac << ":  " << fRun->start << " to " << fRun->end << endl;
    }
    else if (fRun->sCount > 0) {
      (void) SamplePred::Replay(predIdx, level, fRun->start, fRun->end, ptRH);
      //cout << "Level " << level << " false " << ptRH << " onto " << bitStart << " + " << fac << ":  " << fRun->start << " to " << fRun->end << endl;
    }
  }

  return lhSum;
}
