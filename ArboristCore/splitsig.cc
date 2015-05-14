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
#include "predictor.h"
#include "samplepred.h"
#include "pretree.h"
#include "splitpred.h"

//#include <iostream>
using namespace std;

/* Split signature values only live during a single level, from argmax
   pass one (splitting) through argmax pass two.
*/

int SplitSig::nPred = -1;
double SSNode::minRatio = 0.0;

// TODO:  Economize on width (nPred) here et seq.
//

/**
   @brief Sets immutable static values.

   @param _nPred is the number of predictors.

   @param _minRatio is an inf information content for splitting.

   @return void.
 */
void SplitSig::Immutables(int _nPred, double _minRatio) {
  nPred = _nPred;
  SSNode::minRatio = _minRatio;
}


/**
   @brief Finalizer.
*/  
void SplitSig::DeImmutables() {
  nPred = -1;
  SSNode::minRatio = 0.0;
}


/**
   @brief Dispatches nonterminal method based on predictor type.

   With LH and RH PreTree indices known, the sample indices associated with
   this split node can be looked up and remapped.  Replay() assigns actual
   index values, irrespective of whether the pre-tree nodes at these indices
   are terminal or non-terminal.


   @param splitIdx is the index node index.

   @param ptId is the pretree index.

   @param lhStart is the start index of the LHS.

   @return void.

   Sacrifices elegance for efficiency, as coprocessor may not support virtual calls.
*/
double SSNode::NonTerminal(SamplePred *samplePred, PreTree *preTree, SplitPred *splitPred, int level, int start, int end, int ptId, int &ptLH, int &ptRH) {
  preTree->TerminalOffspring(ptId, ptLH, ptRH);
  double splitVal, lhSum;
  int facCard = Predictor::FacCard(predIdx);
  if (facCard > 0) {
    lhSum = NonTerminalFac(samplePred, preTree, splitPred, level, start, end, ptLH, ptRH, facCard, splitVal);
  }
  else {
    lhSum = NonTerminalNum(samplePred, preTree, level, start, end, ptLH, ptRH, splitVal);
  }
  preTree->NonTerminal(ptId, info, splitVal, predIdx);

  return lhSum;
}


/**
   @brief Writes PreTree nonterminal node for factor predictor.
 */
double SSNode::NonTerminalFac(SamplePred *samplePred, PreTree *preTree, SplitPred *splitPred, int level, int start, int end, int ptLH, int ptRH, int facCard, double &splitVal) {
  double lhSum = 0.0;
  int bitOff = preTree->TreeBitOffset();
  preTree->Replay(samplePred, predIdx, level, start, end, ptRH);
  for (int slot = 0; slot <= facLHTop; slot++) {
    int runStart, runEnd;
    int fac = splitPred->RunBounds(splitIdx, predIdx, slot, runStart, runEnd);
    preTree->LHBit(bitOff + fac);
    lhSum += preTree->Replay(samplePred, predIdx, level, runStart, runEnd, ptLH);
  }
  splitVal = bitOff;
  preTree->BumpOff(facCard);

  return lhSum;
}


/**
   @brief Writes PreTree nonterminal node for numerical predictor.
 */
double SSNode::NonTerminalNum(SamplePred *samplePred, PreTree *preTree, int level, int start, int end, int ptLH, int ptRH, double &splitVal) {
  int rkHigh, rkLow;
  samplePred->SplitRanks(predIdx, level, start + lhIdxCount - 1, rkLow, rkHigh);
  splitVal = Predictor::SplitVal(predIdx, rkLow, rkHigh);

  (void) preTree->Replay(samplePred, predIdx, level, start + lhIdxCount, end, ptRH);
  return preTree->Replay(samplePred, predIdx, level, start, start + lhIdxCount - 1, ptLH);
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
  for (int predIdx = 0; predIdx < nPred; predIdx++, predOff += splitCount) {
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
  for (int i = 0; i < nPred * splitCount; i++) {
    levelSS[i].info = 0.0;
  }
}


/**
   @brief Deallocates level's signatures.

   @return void.
 */
void SplitSig::LevelClear() {
  delete [] levelSS;
  levelSS = 0;
}
