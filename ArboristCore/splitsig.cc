// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "splitsig.h"
#include "train.h"
#include "predictor.h"
#include "index.h"
#include "facrun.h"
#include "pretree.h"
#include "samplepred.h"

#include <iostream>
using namespace std;

SplitSig *SplitSig::levelSS = 0;
int SplitSig::nPred = -1;

// TODO:  Economize on width (nPred) here et seq.
//
void SplitSig::Factory(int _levelMax, int _nPred) {
  nPred = _nPred;
  levelSS = new SplitSig[_levelMax * nPred];
}

void SplitSig::ReFactory(int _levelMax) {
  delete [] levelSS;

  levelSS = new SplitSig[_levelMax * nPred];
  TreeInit(_levelMax);
}

void SplitSig::DeFactory() {
  delete [] levelSS;
  levelSS = 0;

  nPred = -1;
}

// Returns SplitSig with minimal information, in a form suitable for decision
// tree construction.
//
void SplitSig::NonTerminalNum(int level, int lhStart, int ptId) {
  int rkLow, rkHigh;
  SamplePred::SplitRanks(predIdx, level, lhStart + lhIdxCount - 1, rkLow, rkHigh);
  double splitVal = Predictor::SplitVal(predIdx, rkLow, rkHigh);
  PreTree::NonTerminalGeneric(ptId, info, splitVal, predIdx);
}

// Causes pretree to set LHS bits by unpacking dense 'facOrd' vector of bit offsets.
// Returns starting bit offset in pretree, for later use by Replay().
//
void SplitSig::NonTerminalFac(int splitIdx, int ptId) {
  int bitOff = PreTree::TreeBitOffset();
  //  cout << "Factor split top " << fac.lhTop << "  (" << predIdx << "):  " << lhIdxCount << " true indices to bit offset " << bitOff << ": " << endl;
  for (int slot = 0; slot <= fac.lhTop; slot++) {
    PreTree::SingleBit(FacRun::FacVal(splitIdx, predIdx, slot));
  }
  PreTree::NonTerminalFac(ptId, info, predIdx);

  fac.bitOff = bitOff;
}

// Once again, replaces elegance with efficiency.
//
double SplitSig::Replay(int splitIdx, int ptLH, int ptRH) {
  if (Predictor::FacIdx(predIdx) >= 0)
    return FacRun::Replay(splitIdx, predIdx, level, fac.bitOff, ptLH, ptRH);
  else
    return NodeCache::ReplayNum(splitIdx, predIdx, level, lhIdxCount);
}


// Returns absolute predictor index of SplitSig with highest information
// content greater than 'minInfo',if any.
//
SplitSig* SplitSig::ArgMax(int splitIdx, int _level, double preBias, double minInfo) {
  SplitSig *argMax = 0;
  double maxInfo = preBias + minInfo;

  // TODO:  Randomize predictor walk to break ties nondeterministically.
  //
  SplitSig *ssBase = Lookup(splitIdx);
  for (int predIdx = 0; predIdx < nPred; predIdx++) {
    SplitSig *candSS = &ssBase[predIdx];
    if (candSS->level == _level && candSS->info > maxInfo) {
      argMax = candSS;
      maxInfo = candSS->info;
    }
  }

  if (argMax != 0) {
    argMax->info -= preBias;
  }

  return argMax;
}

// TODO:  Implement internally ut avoid host calls from coprocessor.
//
double SplitSig::MinInfo() {
  return Train::MinInfo(info);
}

// Resets the level field for all SplitSigs and initializes 'predIdx' field.
//
void SplitSig::TreeInit(int _levelMax) {
  int i = 0;
  for (int splitIdx = 0; splitIdx < _levelMax; splitIdx++) {
    for (int predIdx = 0; predIdx < nPred; predIdx++) {
      levelSS[i].level = -1;
      levelSS[i].predIdx = predIdx;
      i++;
    }
  }
}
