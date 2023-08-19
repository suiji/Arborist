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
#include "algparam.h"
#include "obsfrontier.h"
#include "indexset.h"
#include "predictorframe.h"
#include "sampledobs.h"
#include "splitfrontier.h"
#include "interlevel.h"
#include "ompthread.h"
#include "branchsense.h"
#include "sampler.h"

unsigned int Frontier::totLevels = 0;

void Frontier::immutables(unsigned int totLevels) {
  Frontier::totLevels = totLevels;
}


void Frontier::deInit() {
  totLevels = 0;
}


unique_ptr<PreTree> Frontier::oneTree(const PredictorFrame* frame,
				      const Train* train,
				      SampledObs* sampledObs) {
  Frontier frontier(frame, train, sampledObs);
  sampledObs->sampleRoot(frame, train->getNodeScorer());
  SampleMap smNonTerm = frontier.produceRoot();
  return frontier.splitByLevel(smNonTerm);
}


Frontier::Frontier(const PredictorFrame* frame_,
		   const Train* train,
		   SampledObs* sampledObs_) :
  frame(frame_),
  scorer(train->getNodeScorer()),
  sampledObs(sampledObs_),
  bagCount(sampledObs->getBagCount()),
  nCtg(sampledObs->getNCtg()),
  interLevel(make_unique<InterLevel>(frame, sampledObs, this)),
  pretree(make_unique<PreTree>(frame, bagCount)),
  smTerminal(SampleMap(bagCount)) {
}


SampleMap Frontier::produceRoot() {
  pretree->offspring(0, true);
  frontierNodes.emplace_back(sampledObs);

  SampleMap smNonterm = SampleMap(bagCount);
  smNonterm.addNode(bagCount, 0);
  iota(smNonterm.sampleIndex.begin(), smNonterm.sampleIndex.end(), 0);

  return smNonterm;
}


unique_ptr<PreTree> Frontier::splitByLevel(SampleMap& smNonterm) {
  while (!frontierNodes.empty()) {
    smNonterm = splitDispatch(smNonterm);
    vector<IndexSet> frontierNext = produceLevel();
    interLevel->overlap(frontierNodes, frontierNext, smNonterm.getEndIdx());
    frontierNodes = std::move(frontierNext);
  }
  pretree->setTerminals(std::move(smTerminal));

  return std::move(pretree);
}


SampleMap Frontier::splitDispatch(const SampleMap& smNonterm) {
  // The current frontier can be scored as soon as its nodes are in
  // place.
  scorer->frontierPreamble(this);

  earlyExit(interLevel->getLevel());
  CandType cand = interLevel->repartition(this);
  splitFrontier = SplitFactoryT::factory(this);

  BranchSense branchSense(bagCount);
  splitFrontier->split(cand, branchSense);

  SampleMap smNext = surveySplits();
  ObsFrontier* cellFrontier = interLevel->getFront();
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitIdx = 0; splitIdx < frontierNodes.size(); splitIdx++) {
      IndexSet iSet(getNode(splitIdx));
      cellFrontier->updateMap(iSet, branchSense, smNonterm, smTerminal, smNext);
      pretree->setScore(iSet, scorer->score(smNonterm, iSet));
    }
  }

  return smNext;
}


void Frontier::earlyExit(unsigned int level) {
  if (level + 1 == totLevels) {
    for (auto & iSet : frontierNodes) {
      iSet.setUnsplitable();
    }
  }
}


vector<IndexSet> Frontier::produceLevel() {
  vector<IndexSet> frontierNext;
  for (auto iSet : frontierNodes) {
    if (!iSet.isTerminal()) {
      frontierNext.emplace_back(this, iSet, true);
      frontierNext.emplace_back(this, iSet, false);
    }
  }

  return frontierNext;
}


SampleMap Frontier::surveySplits() {
  SampleMap smNext;
  for (auto & iSet : frontierNodes) {
    registerSplit(iSet, smNext);
  }

  smNext.sampleIndex = vector<IndexT>(smNext.getEndIdx());

  return smNext;
}


void Frontier::registerSplit(IndexSet& iSet,
			     SampleMap& smNext) {
  if (iSet.isTerminal()) {
    registerTerminal(iSet);
  }
  else {
    registerNonterminal(iSet, smNext);
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


void Frontier::updateSimple(const vector<SplitNux>& nuxMax,
			    BranchSense& branchSense) {
  IndexT splitIdx = 0;
  for (auto nux : nuxMax) {
    if (!nux.noNux()) {
      // splitUpdate() updates the runSet accumulators, so must
      // be invoked prior to updating the pretree's criterion state.
      frontierNodes[splitIdx].update(splitFrontier->splitUpdate(nux, branchSense));
      pretree->addCriterion(splitFrontier.get(), nux);
    }
    splitIdx++;
  }
}


void Frontier::updateCompound(const vector<vector<SplitNux>>& nuxMax,
			      BranchSense& branchSense) {
  pretree->consumeCompound(splitFrontier.get(), nuxMax);
}


vector<double> Frontier::sumsAndSquares(vector<vector<double> >& ctgSum) {
  vector<double> sumSquares(frontierNodes.size());

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitIdx = 0; splitIdx < frontierNodes.size(); splitIdx++) {
      ctgSum[splitIdx] = frontierNodes[splitIdx].sumsAndSquares(sumSquares[splitIdx]);
    }
  }
  return sumSquares;
}


SplitNux Frontier::candMax(IndexT splitIdx,
			   const vector<SplitNux>& candV) const {
  return frontierNodes[splitIdx].candMax(candV);
}


