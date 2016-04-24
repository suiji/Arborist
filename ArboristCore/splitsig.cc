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
#include "samplepred.h"
#include "pretree.h"
#include "splitpred.h"
#include "run.h"

#include <cfloat>

//#include <iostream>
using namespace std;

/* Split signature values only live during a single level, from argmax
   pass one (splitting) through argmax pass two.
*/

unsigned int SplitSig::nPred = 0;
double SSNode::minRatio = 0.0;

// TODO:  Economize on width (nPred) here et seq.
//

/**
   @brief Sets immutable static values.

   @param _nPred is the number of predictors.

   @param _minRatio is an inf information content for splitting.  Must
   be non-negative, as otherwise ArgMax cannot distinguish splitting
   candidates from unset SSNodes, which have initial 'info' == 0.

   @return void.
 */
void SplitSig::Immutables(unsigned int _nPred, double _minRatio) {
  nPred = _nPred;
  SSNode::minRatio = _minRatio;
}


/**
   @brief Finalizer.
*/  
void SplitSig::DeImmutables() {
  nPred = 0;
  SSNode::minRatio = 0.0;
}


/**
   @brief Sets splitting fields for a splitting predictor.

   @param _spPair is the SplitPred pair precipitating the split.

   @param _sCount is the count of samples in the LHS.

   @param _lhIdxCount is count of indices associated with the LHS.

   @param _info is the splitting information value, currently Gini.

   @return void.
 */
void SplitSig::Write(const SPPair *_spPair, unsigned int _sCount, unsigned int _lhIdxCount, double _info) {
  SSNode ssn;
  ssn.runId = _spPair->RSet();
  ssn.sCount = _sCount;
  ssn.lhIdxCount = _lhIdxCount;
  ssn.info = _info;

  int splitIdx; // Dummy.
  _spPair->Coords(splitIdx, ssn.predIdx);
  Lookup(splitIdx, ssn.predIdx) = ssn;
}


SSNode::SSNode() : info(-DBL_MAX) {
}


/**
   @brief Dispatches nonterminal method based on predictor type.

   With LH and RH PreTree indices known, the sample indices associated with
   this split node can be looked up and remapped.  Replay() assigns actual
   index values, irrespective of whether the pre-tree nodes at these indices
   are terminal or non-terminal.

   @param ptId is the pretree index.

   @param lhStart is the start index of the LHS.

   @return void.

   Sacrifices elegance for efficiency, as coprocessor may not support virtual calls.
*/
double SSNode::NonTerminal(SamplePred *samplePred, PreTree *preTree, SplitPred *splitPred, int level, int start, int end, unsigned int ptId, unsigned int &ptLH, unsigned int &ptRH) {
  return runId >= 0 ? NonTerminalRun(samplePred, preTree, splitPred->Runs(), level, start, end, ptId, ptLH, ptRH) : NonTerminalNum(samplePred, preTree, level, start, end, ptId, ptLH, ptRH);
}


/**
   @brief Writes PreTree nonterminal node for multi-run (factor) predictor.

   @return sum of left-hand subnode's response values.
 */
double SSNode::NonTerminalRun(SamplePred *samplePred, PreTree *preTree, Run *run, int level, int start, int end, unsigned int ptId, unsigned int &ptLH, unsigned int &ptRH) {
  preTree->NonTerminalFac(info, predIdx, ptId, ptLH, ptRH);

  // Replays entire index extent of node with RH pretree index then,
  // where appropriate, overwrites by replaying with LH index in the
  // loop to follow.
  (void) preTree->Replay(samplePred, predIdx, level, start, end, ptRH);

  double lhSum = 0.0;
  for (int outSlot = 0; outSlot < run->RunsLH(runId); outSlot++) {
    int runStart, runEnd;
    unsigned int rank = run->RunBounds(runId, outSlot, runStart, runEnd);
    preTree->LHBit(ptId, rank);
    lhSum += preTree->Replay(samplePred, predIdx, level, runStart, runEnd, ptLH);
  }

  return lhSum;
}


/**
   @brief Writes PreTree nonterminal node for numerical predictor.

   @return sum of LH subnode's sample values.
 */
double SSNode::NonTerminalNum(SamplePred *samplePred, PreTree *preTree, int level, int start, int end, unsigned int ptId, unsigned int &ptLH, unsigned int &ptRH) {
  unsigned int rkLow, rkHigh;
  samplePred->SplitRanks(predIdx, level, start + lhIdxCount - 1, rkLow, rkHigh);
  preTree->NonTerminalNum(info, predIdx, rkLow, rkHigh, ptId, ptLH, ptRH);
  
  double lhSum = preTree->Replay(samplePred, predIdx, level, start, start + lhIdxCount - 1, ptLH);
  (void) preTree->Replay(samplePred, predIdx, level, start + lhIdxCount, end, ptRH);

  return lhSum;
}


/**
   @brief Walks predictors associated with a given split index to find which,
   if any, maximizes information gain above split's threshold.

   @param splitIdx is the current split index.

   @param gainMax begins as the minimal information gain suitable for spltting this
   index node.

   @return void.
 */
SSNode *SplitSig::ArgMax(int splitIdx, double gainMax) const {
  SSNode *argMax = 0;

  // TODO: Break ties nondeterministically.
  //
  int predOff = splitIdx;
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++, predOff += splitCount) {
    SSNode *candSS = &levelSS[predOff];
    if (candSS->info > gainMax) {
      argMax = candSS;
      gainMax = candSS->info;
    }
  }

  return argMax;
}


/**
 @brief Allocates level's splitting signatures and initializes 'info'
 content to zero.

 @param _splitCount is the number of splits in the current level.

 @return void.
*/
void SplitSig::LevelInit(int _splitCount) {
  splitCount = _splitCount;
  levelSS = new SSNode[nPred * splitCount];
}


/**
   @brief Deallocates level's signatures.

   @return void.
 */
void SplitSig::LevelClear() {
  delete [] levelSS;
  levelSS = 0;
}
