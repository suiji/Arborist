// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file frontier.cc

   @brief Maintains the sample-index representation of the frontier, typically by level.

   @author Mark Seligman
 */

#include "frontier.h"
#include "indexset.h"
#include "trainframe.h"
#include "sample.h"
#include "sampler.h"
#include "train.h"
#include "splitfrontier.h"
#include "deffrontier.h"
#include "path.h"
#include "ompthread.h"
#include "branchsense.h"

#include <numeric>


unsigned int Frontier::totLevels = 0;

void Frontier::immutables(unsigned int totLevels) {
  Frontier::totLevels = totLevels;
}


void Frontier::deImmutables() {
  totLevels = 0;
}


unique_ptr<PreTree> Frontier::oneTree(const TrainFrame* frame,
                                      Sampler* sampler) {
  sampler->rootSample(frame);
  unique_ptr<Frontier> frontier(make_unique<Frontier>(frame, sampler->getSample()));
  return frontier->levels(sampler->getSample());
}


Frontier::Frontier(const TrainFrame* frame_,
                   const Sample* sample) :
  frame(frame_),
  indexSet(vector<IndexSet>(1)),
  bagCount(sample->getBagCount()),
  nCtg(sample->getNCtg()),
  defMap(make_unique<DefFrontier>(frame, this)),
  nodeRel(false),
  pretree(make_unique<PreTree>(frame->getCardExtent(), bagCount)),
  smTerminal(SampleMap(bagCount)),
  smNonterm(SampleMap(bagCount)) {
  indexSet[0].initRoot(sample);
  smNonterm.addNode(bagCount, 0);
  iota(smNonterm.indices.begin(), smNonterm.indices.end(), 0);
}


unique_ptr<PreTree> Frontier::levels(const Sample* sample) {
  defMap->stage(sample);

  unsigned int level = 0;
  while (!indexSet.empty()) {
    unique_ptr<BranchSense> branchSense = make_unique<BranchSense>(bagCount);
    SplitFrontier::split(this, branchSense.get());
    indexSet = splitDispatch(branchSense.get());
    defMap->setPrecandidates(level);
    level++;
  }
  
  relFlush();
  pretree->cacheSampleMap(move(recoverSt2PT()));

  return move(pretree);
}


void Frontier::relFlush() {
  if (nodeRel) {
    for (IndexT relIdx = 0; relIdx < smNonterm.getEndIdx(); relIdx++) {
      defMap->setExtinct(relIdx, smNonterm.indices[relIdx]);
    }
  }
}

void Frontier::earlyExit(unsigned int level) {
  if (level + 1 == totLevels) {
    for (auto & iSet : indexSet) {
      iSet.setUnsplitable();
    }
  }
}


vector<IndexSet> Frontier::splitDispatch(const BranchSense* branchSense) {
  SampleMap smNext = surveySet(indexSet);
  nextLevel(branchSense, smNext);
  smNonterm = move(smNext);

  defMap->overlap(smNonterm.getNodeCount(), bagCount, smNonterm.getEndIdx(), nodeRel);

  return produce();
}


SampleMap Frontier::surveySet(vector<IndexSet>& indexSet) {
  terminalNodes = smTerminal.getNodeCount();
  SampleMap smNext;
  for (auto & iSet : indexSet) {
    surveySplit(iSet, smNext);
  }

  smNext.indices = vector<IndexT>(smNext.getEndIdx());

  return smNext;
}


void Frontier::surveySplit(IndexSet& iSet,
			   SampleMap& smNext) {
  if (iSet.isTerminal()) {
    registerTerminal(iSet);
  }
  else {
    registerNonterminal(iSet, smNext);
    iSet.nonterminal(smNext);
  }
}


void Frontier::registerTerminal(IndexSet& iSet) {
  iSet.setIdxNext(smTerminal.getNodeCount());
  smTerminal.addNode(iSet.getExtent(), iSet.getPTId());
}


void Frontier::registerNonterminal(IndexSet& iSet, SampleMap& smNext) {
  iSet.setIdxNext(smNext.getNodeCount());
  smNext.addNode(iSet.getExtentSucc(true), iSet.getPTIdSucc(this, true));
  smNext.addNode(iSet.getExtentSucc(false), iSet.getPTIdSucc(this, false));
}


IndexT Frontier::getPTIdSucc(IndexT ptId, bool senseTrue) const {
  return pretree->getSuccId(ptId, senseTrue);
}


void Frontier::nextLevel(const BranchSense* branchSense,
			 SampleMap& smNext) {
  bool transitional = false;
  if (!nodeRel) {
    nodeRel = IdxPath::localizes(bagCount, smNext.maxExtent);
    transitional = nodeRel;
  }

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
      updateMap(indexSet[splitIdx], branchSense, smNext, transitional);
    }
  }
}


void Frontier::updateMap(IndexSet& iSet,
			 const BranchSense* branchSense,
			 SampleMap& smNext,
			 bool transitional) {
  if (!iSet.isTerminal()) {
    updateLive(branchSense, iSet, smNext, transitional);
  }
  else {
    updateExtinct(iSet, transitional);
  }
}


void Frontier::updateExtinct(const IndexSet& iSet,
			     bool transitional) {
  IdxPath* stPath = defMap->getSubtreePath();
  IndexT* destOut = smTerminal.getWriteStart(iSet.getIdxNext());
  IndexRange range = getNontermRange(iSet);
  for (IndexT idx = range.idxStart; idx != range.getEnd(); idx++) {
    IndexT sIdx = smNonterm.indices[idx];
    *destOut++ = sIdx;
    if (nodeRel && !transitional)  // relExtinct()
      defMap->setExtinct(idx, sIdx);
    else // pathUpdateTerminals();
      stPath->setExtinct(sIdx);
  }
}


void Frontier::updateLive(const BranchSense* branchSense,
			  const IndexSet& iSet,
			  SampleMap& smNext,
			  bool transitional) {
  IdxPath* stPath = defMap->getSubtreePath();
  IndexT nodeIdx = iSet.getIdxNext();
  IndexT baseTrue = smNext.range[nodeIdx].getStart();
  IndexT destTrue = baseTrue;
  IndexT baseFalse = smNext.range[nodeIdx+1].getStart();
  IndexT destFalse = baseFalse;
  IndexRange range = getNontermRange(iSet);
  bool implicitTrue = !iSet.encodesTrue();
  if (nodeRel && !transitional) {
    for (IndexT idx = range.idxStart; idx != range.getEnd(); idx++) {
      bool sense = branchSense->senseTrue(idx, implicitTrue);
      IndexT destIdx = sense ? destTrue++ : destFalse++;
      IndexT sIdx = smNonterm.indices[idx];
      smNext.indices[destIdx] = sIdx;
      defMap->setLive(idx, destIdx, sIdx, iSet.getPathSucc(sense), sense ? baseTrue : baseFalse);
    }
  }    
  else {
    for (IndexT idx = range.idxStart; idx != range.getEnd(); idx++) {
      IndexT sIdx = smNonterm.indices[idx];
      bool sense = branchSense->senseTrue(sIdx, implicitTrue);
      IndexT destIdx = sense ? destTrue++ : destFalse++;
      smNext.indices[destIdx] = sIdx;
      stPath->setSuccessor(sIdx, iSet.getPathSucc(sense), true); // pathUpdateNode()
      if (transitional)
        stPath->setLive(sIdx, iSet.getPathSucc(sense), destIdx);
    }
  }
}


void Frontier::updateSimple(const SplitFrontier* sf,
			    const vector<SplitNux>& nuxMax,
			    BranchSense* branchSense) {
  IndexT splitIdx = 0;
  for (auto nux : nuxMax) {
    if (!nux.noNux()) {
      // splitUpdate() updates the runSet accumulators, so must
      // be invoked prior to updating the pretree's criterion state.
      indexSet[splitIdx].update(sf->splitUpdate(nux, branchSense));
      pretree->addCriterion(sf, nux);
    }
    splitIdx++;
  }
}


void Frontier::updateCompound(const SplitFrontier* sf,
			      const vector<vector<SplitNux>>& nuxMax) {
  pretree->consumeCompound(sf, nuxMax);
}


vector<IndexSet> Frontier::produce() {
  IndexT splitNext = smNonterm.getNodeCount();
  vector<IndexSet> indexNext(splitNext);
  for (auto & iSet : indexSet) {
    iSet.succHands(this, indexNext);
  }
  return indexNext;
}


void Frontier::reachingPath(const IndexSet& iSet,
                            IndexT parIdx) const {
  defMap->reachingPath(iSet, parIdx, getNontermRange(iSet).getStart());
}


vector<double> Frontier::sumsAndSquares(vector<vector<double> >& ctgSum) {
  vector<double> sumSquares(indexSet.size());

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
      ctgSum[splitIdx] = indexSet[splitIdx].sumsAndSquares(sumSquares[splitIdx]);
    }
  }
  return sumSquares;
}


const vector<IndexT> Frontier::recoverSt2PT() const {
  vector<IndexT> st2PT(smTerminal.indices.size()); // bagCount
  IndexT leafIdx = 0;
  for (auto range : smTerminal.range) {
    IndexT ptIdx = smTerminal.ptIdx[leafIdx];
    for (IndexT idx = range.getStart(); idx != range.getEnd(); idx++) {
      IndexT stIdx = smTerminal.indices[idx];
      st2PT[stIdx] = ptIdx;
    }
    leafIdx++;
  }

  return st2PT;
}
