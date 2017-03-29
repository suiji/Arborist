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
#include "bottom.h"
#include "pretree.h"
#include "runset.h"
#include "index.h"

#include <cfloat>

//#include <iostream>
//using namespace std;

/* Split signature values only live during a single level, from argmax
   pass one (splitting) through argmax pass two.
*/

double SSNode::minRatio = 0.0;

// TODO:  Economize on width (nPred) here et seq.
//

/**
   @brief Sets immutable static values.

   @param _minRatio is an inf information content for splitting.  Must
   be non-negative, as otherwise ArgMax cannot distinguish splitting
   candidates from unset SSNodes, which have initial 'info' == 0.

   @return void.
 */
void SplitSig::Immutables(double _minRatio) {
  SSNode::minRatio = _minRatio;
}


/**
   @brief Finalizer.
*/  
void SplitSig::DeImmutables() {
  SSNode::minRatio = 0.0;
}


/**
   @brief Sets splitting fields for a splitting predictor.

   @param _sCount is the count of samples in the LHS.

   @param _idxCount is count of indices associated with the LHS.

   @param _info is the splitting information value, currently Gini.

   @return void.
 */
void SplitSig::Write(unsigned int _levelIdx, unsigned int _predIdx, unsigned int _setIdx, unsigned int _bufIdx, const NuxLH &nux) {
  SSNode ssn;
  ssn.predIdx = _predIdx;
  ssn.setIdx = _setIdx;
  ssn.bufIdx = _bufIdx;
  nux.Ref(ssn.idxStart, ssn.lhExtent, ssn.sCount, ssn.info, ssn.rankRange, ssn.lhImplicit);

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

   @return true iff left-hand of split is explicit.

   Sacrifices elegance for efficiency, as coprocessor may not support virtual calls.
*/
bool SSNode::NonTerminal(Bottom *bottom, PreTree *preTree, Run *run, unsigned int extent, unsigned int ptId, double &sumExpl) {
  return run->IsRun(setIdx) ? NonTerminalRun(bottom, preTree, run, extent, ptId, sumExpl) : NonTerminalNum(bottom, preTree, extent, ptId, sumExpl);
}


/**
   @brief Writes PreTree nonterminal node for multi-run (factor) predictor.

   @return true iff LH is implicit.
 */
bool SSNode::NonTerminalRun(Bottom *bottom, PreTree *preTree, Run *run, unsigned int extent, unsigned int ptId, double &sumExpl) {
  preTree->NonTerminalFac(info, predIdx, ptId);

  sumExpl = ReplayRun(bottom, preTree, ptId, run);
  return !run->ImplicitLeft(setIdx);
}


/**
   @brief Writes PreTree nonterminal node for multi-run (factor) predictor.

   @return sum of left-hand subnode's response values.
 */
double SSNode::ReplayRun(Bottom *bottom, PreTree *preTree, unsigned int ptId, const Run *run) {
  double sumExpl = 0.0;
  if (run->ImplicitLeft(setIdx)) {// LH runs hold bits, RH hold replay indices.
    for (unsigned int outSlot = 0; outSlot < run->RunCount(setIdx); outSlot++) {
      if (outSlot < run->RunsLH(setIdx)) {
        preTree->LHBit(ptId, run->Rank(setIdx, outSlot));
      }
      else {
	unsigned int runStart, runExtent;
	run->RunBounds(setIdx, outSlot, runStart, runExtent);
        sumExpl += bottom->BlockReplay(predIdx, bufIdx, runStart, runExtent);
      }
    }
  }
  else { // LH runs hold bits as well as replay indices.
    for (unsigned int outSlot = 0; outSlot < run->RunsLH(setIdx); outSlot++) {
      preTree->LHBit(ptId, run->Rank(setIdx, outSlot));
      unsigned int runStart, runExtent;
      run->RunBounds(setIdx, outSlot, runStart, runExtent);
      sumExpl += bottom->BlockReplay(predIdx, bufIdx, runStart, runExtent);
    }
  }

  return sumExpl;
}


/**
   @brief Writes PreTree nonterminal node for numerical predictor.

   @return true iff LH is explicit.
 */
bool SSNode::NonTerminalNum(Bottom *bottom, PreTree *preTree, unsigned int extent, unsigned int ptId, double &sumExpl) {
  preTree->NonTerminalNum(info, predIdx, rankRange, ptId);

  sumExpl = ReplayNum(bottom, extent);
  return lhImplicit == 0;
}


/**
   @brief Writes successor id over appropriate side of the cut.  Replay()
   has already preinitialized all samples with the complementary id.

   @param sum is the response sum of the predecessor node.

   @param extent is the size of the predecessor node's index set.

   @return sum of explicit successor node's sample values.
 */
double SSNode::ReplayNum(Bottom *bottom, unsigned int extent) {
  return bottom->BlockReplay(predIdx, bufIdx, lhImplicit == 0 ? idxStart : idxStart - lhImplicit + lhExtent, lhImplicit == 0 ? lhExtent : extent - lhExtent);
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
void SplitSig::LevelInit(unsigned int _splitCount) {
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
