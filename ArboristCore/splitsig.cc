// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitsig.cc

   @brief Methods to construct and transmit SplitSig objects, which record the reults of predictor argmax methods.

   @author Mark Seligman
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

/**
   @brief Lights initializer for level workspace.

   @param _levelMax is the current level-max value.

   @param _nPred is the number of predictors.

   @return void.
 */
void SplitSig::Factory(int _levelMax, int _nPred) {
  nPred = _nPred;
  levelSS = new SplitSig[_levelMax * nPred];
}

/**
   @brief Reallocates level workspace.

   @param _levelMax is the new level-max value.

   @return void.
*/
void SplitSig::ReFactory(int _levelMax) {
  delete [] levelSS;

  levelSS = new SplitSig[_levelMax * nPred];
  TreeInit(_levelMax);
}


/**
   @brief Finalizer.
*/  
void SplitSig::DeFactory() {
  delete [] levelSS;
  levelSS = 0;

  nPred = -1;
}

/**
   @brief Records splitting information in pretree for numerical predictor.

   @param level is the current level.

   @param lhStart is the starting index of the LHS.

   @param ptId is the pretree index.

   @return void.
*/
void SplitSig::NonTerminalNum(int level, int lhStart, int ptId) {
  int rkLow, rkHigh;
  SamplePred::SplitRanks(predIdx, level, lhStart + lhIdxCount - 1, rkLow, rkHigh);
  double splitVal = Predictor::SplitVal(predIdx, rkLow, rkHigh);
  PreTree::NonTerminalGeneric(ptId, info, splitVal, predIdx);
}

/**
   @brief Records splitting information in pretree for factor-valued predictor.

   @param splitIdx is the index node index.

   @param ptId is the pretree index.

   @return void.
 */
void SplitSig::NonTerminalFac(int splitIdx, int ptId) {
  int bitOff = PreTree::TreeBitOffset();
  for (int slot = 0; slot <= fac.lhTop; slot++) {
    PreTree::SingleBit(FacRun::FacVal(splitIdx, predIdx, slot));
  }
  PreTree::NonTerminalFac(ptId, info, predIdx);

  fac.bitOff = bitOff;
}

/**
   @brief Dispatches replay method according to predictor type.

   @param spiltIdx is the index node index.

   @param ptLH is the pretree index of the LHS.

   @param ptRH is the pretree index of the RHS.

   @return sum of LHS reponse values.

   Not virtual:  once again replaces elegance with efficiency.
*/
double SplitSig::Replay(int splitIdx, int ptLH, int ptRH) {
  if (Predictor::FacIdx(predIdx) >= 0)
    return FacRun::Replay(splitIdx, predIdx, level, fac.bitOff, ptLH, ptRH);
  else
    return NodeCache::ReplayNum(splitIdx, predIdx, level, lhIdxCount);
}


/**
 @brief  Walks level's split signatures to find maximal information content.

 @param splitIdx is the index node index.

 @param _level is the current level.

 @param preBias is an information threshold derived from the index node.

 @param minInfo is an additional threshold derived from the pretree.

 @return SplitSig with maximal information content, if any, exceeding threshold.
*/
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

/**
   @brief Derives an information threshold.

   @return information threshold

   TODO:  Implement internally ut avoid host calls from coprocessor.
*/
double SplitSig::MinInfo() {
  return Train::MinInfo(info);
}

/**
 @brief Resets all level and predIdx fields, per tree.

 @param _levelMax is the current level-max value.

 @return void.
*/
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
