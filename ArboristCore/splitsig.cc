// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitsig.cc

   @brief Methods to construct and transmit SplitSig objects, which record
   the reults of predictor argmax methods.

   @author Mark Seligman
 */


#include "splitsig.h"
#include "pretree.h"
#include "runset.h"
#include "index.h"
#include "samplepred.h"

#include <cfloat>


double SSNode::minRatio = 0.0;


void SplitSig::Immutables(double minRatio) {
  SSNode::minRatio = minRatio;
}


void SplitSig::DeImmutables() {
  SSNode::minRatio = 0.0;
}


void SplitSig::Write(unsigned int levelIdx,
		     unsigned int predIdx,
		     unsigned int setIdx,
		     unsigned int bufIdx,
		     const NuxLH &nux) {
  SSNode ssn;
  ssn.predIdx = predIdx;
  ssn.setIdx = setIdx;
  ssn.bufIdx = bufIdx;
  nux.Ref(ssn.idxStart, ssn.lhExtent, ssn.sCount, ssn.info, ssn.rankRange, ssn.lhImplicit);

  Lookup(levelIdx, ssn.predIdx) = ssn;
}


SSNode::SSNode() : info(-DBL_MAX) {
}


bool SSNode::NonTerminal(IndexLevel *index,
			 PreTree *preTree,
			 IndexSet *iSet,
			 Run *run) const {
  return run->IsRun(setIdx) ? BranchRun(index, iSet, preTree, run) : BranchNum(index, iSet, preTree);
}


bool SSNode::BranchRun(IndexLevel *index,
		       IndexSet *iSet,
		       PreTree *preTree,
		       Run *run) const {
  // TODO:  Recast as run->Replay(index, iSet, preTree, predIdx, bufIdx, setIdx)
  preTree->BranchFac(info, predIdx, iSet->PTId());
  ReplayRun(index, iSet, preTree, run);

  return  !run->ImplicitLeft(setIdx);
}


void SSNode::ReplayRun(IndexLevel *index,
		       IndexSet *iSet,
		       PreTree *preTree,
		       const Run *run) const {
  if (run->ImplicitLeft(setIdx)) {// LH runs hold bits, RH hold replay indices.
    for (unsigned int outSlot = 0; outSlot < run->RunCount(setIdx); outSlot++) {
      if (outSlot < run->RunsLH(setIdx)) {
        preTree->LHBit(iSet->PTId(), run->Rank(setIdx, outSlot));
      }
      else {
	unsigned int runStart, runExtent;
	run->RunBounds(setIdx, outSlot, runStart, runExtent);
        index->BlockReplay(iSet, predIdx, bufIdx, runStart, runExtent);
      }
    }
  }
  else { // LH runs hold bits as well as replay indices.
    for (unsigned int outSlot = 0; outSlot < run->RunsLH(setIdx); outSlot++) {
      preTree->LHBit(iSet->PTId(), run->Rank(setIdx, outSlot));
      unsigned int runStart, runExtent;
      run->RunBounds(setIdx, outSlot, runStart, runExtent);
      index->BlockReplay(iSet, predIdx, bufIdx, runStart, runExtent);
    }
  }
}


bool SSNode::BranchNum(IndexLevel *index,
		       IndexSet *iSet,
		       PreTree *preTree) const {
  preTree->BranchNum(info, predIdx, rankRange, iSet->PTId());
  ReplayNum(index, iSet);

  return lhImplicit == 0;
}


void SSNode::ReplayNum(IndexLevel *index,
		       IndexSet *iSet) const {
  index->BlockReplay(iSet, predIdx, bufIdx, lhImplicit == 0 ? idxStart : idxStart - lhImplicit + lhExtent, lhImplicit == 0 ? lhExtent : iSet->Extent() - lhExtent);
}


void SSNode::ArgMax(const SplitSig *splitSig,
		    unsigned int splitIdx) {
  Update(splitSig->ArgMax(splitIdx, info));
}


SSNode *SplitSig::ArgMax(unsigned int levelIdx,
			 double gainMax) const {
  SSNode *argMax = nullptr;

  // TODO: Break ties nondeterministically.
  //
  unsigned int predOff = levelIdx;
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++, predOff += splitCount) {
    SSNode *candSS = &levelSS[predOff];
    if (candSS->GainMax(gainMax)) {
      argMax = candSS;
    }
  }

  return argMax;
}


void SplitSig::LevelInit(unsigned int splitCount) {
  levelSS = new SSNode[nPred * splitCount];
  this->splitCount = splitCount;
}


void SplitSig::LevelClear() {
  delete [] levelSS;
  levelSS = nullptr;
}
