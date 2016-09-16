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
#include "runset.h"

#include <cfloat>

//#include <iostream>
//using namespace std;

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

   @param _sCount is the count of samples in the LHS.

   @param _lhIdxCount is count of indices associated with the LHS.

   @param _info is the splitting information value, currently Gini.

   @return void.
 */
void SplitSig::Write(unsigned int _levelIdx, unsigned int _predIdx, unsigned int _setIdx, unsigned int _bufIdx, const SplitNux &nux) {
  SSNode ssn;
  ssn.predIdx = _predIdx;
  ssn.setIdx = _setIdx;
  ssn.bufIdx = _bufIdx;
  nux.Ref(ssn.lhIdxCount, ssn.sCount, ssn.info);

  Lookup(_levelIdx, ssn.predIdx) = ssn;
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
void SSNode::NonTerminal(SamplePred *samplePred, PreTree *preTree, Run *run, int start, unsigned int ptId, unsigned int &ptLH, unsigned int &ptRH) {
  return run->IsRun(setIdx) ? NonTerminalRun(preTree, run, ptId, ptLH, ptRH) : NonTerminalNum(samplePred, preTree, start, ptId, ptLH, ptRH);
}


double SSNode::Replay(SamplePred *samplePred, PreTree *preTree, Run *run, int start, double sum, unsigned int ptId, unsigned int ptLH, unsigned int ptRH) {
  return run->IsRun(setIdx) ? ReplayRun(samplePred, preTree, sum, ptId, ptLH, ptRH, run) : ReplayNum(samplePred, preTree, start, ptLH);
}



/**
   @brief Writes PreTree nonterminal node for multi-run (factor) predictor.

   @return sum of left-hand subnode's response values.
 */
void SSNode::NonTerminalRun(PreTree *preTree, Run *run, unsigned int ptId, unsigned int &ptLH, unsigned int &ptRH) {
  preTree->NonTerminalFac(info, predIdx, ptId, run->ExposeRH(setIdx), ptLH, ptRH);
}


/**
   @brief Writes PreTree nonterminal node for numerical predictor.

   @return sum of LH subnode's sample values.
 */
void SSNode::NonTerminalNum(SamplePred *samplePred, PreTree *preTree, int start, unsigned int ptId, unsigned int &ptLH, unsigned int &ptRH) {
  unsigned int rkLow, rkHigh;
  samplePred->SplitRanks(predIdx, bufIdx, start + lhIdxCount - 1, rkLow, rkHigh);
  preTree->NonTerminalNum(info, predIdx, rkLow, rkHigh, ptId, ptLH, ptRH);
}


/**
   @brief Writes PreTree nonterminal node for multi-run (factor) predictor.

   @return sum of left-hand subnode's response values.
 */
double SSNode::ReplayRun(SamplePred *samplePred, PreTree *preTree, double sum, unsigned int ptId, unsigned int ptLH, unsigned int ptRH, Run *run) {
  // Preplay() has overwritten all live sample indices with one or the other
  // descendant.  Now the complementary descendant index is applied as appropriate.
  // appropriate.
  //
  if (run->ExposeRH(setIdx)) { // Must walk both LH and RH runs.
    double rhSum = 0.0;
    for (unsigned int outSlot = 0; outSlot < run->RunCount(setIdx); outSlot++) {
      unsigned int runStart, runEnd;
      unsigned int rank = run->RunBounds(setIdx, outSlot, runStart, runEnd);
      if (outSlot < run->RunsLH(setIdx)) {
        preTree->LHBit(ptId, rank);
      }
      else {
        rhSum += preTree->Replay(samplePred, predIdx, bufIdx, runStart, runEnd, ptRH);
      }
    }
    return sum - rhSum;
  }
  else { // Suffices just to walk LH runs.
    double lhSum = 0.0;
    for (unsigned int outSlot = 0; outSlot < run->RunsLH(setIdx); outSlot++) {
      unsigned int runStart, runEnd;
      unsigned int rank = run->RunBounds(setIdx, outSlot, runStart, runEnd);
      preTree->LHBit(ptId, rank);
      lhSum += preTree->Replay(samplePred, predIdx, bufIdx, runStart, runEnd, ptLH);
    }
    return lhSum;
  }
}


/**
   @brief Writes LH successor id at all sample indices preceding the cut.  Preplay()
   has already preinitialized all samples with the RH id.

   @return sum of LH subnode's sample values.
 */
double SSNode::ReplayNum(SamplePred *samplePred, PreTree *preTree, int start, unsigned int ptLH) {
  return  preTree->Replay(samplePred, predIdx, bufIdx, start, start + lhIdxCount - 1, ptLH);
}


/**
   @brief Walks predictors associated with a given split index to find which,
   if any, maximizes information gain above split's threshold.

   @param levelIdx is the current split index.

   @param gainMax begins as the minimal information gain suitable for spltting this
   index node.

   @return void.
 */
SSNode *SplitSig::ArgMax(unsigned int levelIdx, double gainMax) const {
  SSNode *argMax = 0;

  // TODO: Break ties nondeterministically.
  //
  unsigned int predOff = levelIdx;
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
