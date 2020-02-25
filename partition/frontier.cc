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
#include "summaryframe.h"
#include "sample.h"
#include "train.h"
#include "obspart.h"
#include "splitfrontier.h"
#include "algparam.h"
#include "defmap.h"
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


Frontier::~Frontier() {
}


unique_ptr<PreTree> Frontier::oneTree(const Train* train,
				      const SummaryFrame* frame,
                                      const Sample *sample) {
  unique_ptr<Frontier> frontier(make_unique<Frontier>(frame, sample));
  return frontier->levels(sample, frame);
}


Frontier::Frontier(const SummaryFrame* frame,
                   const Sample* sample) :
  indexSet(vector<IndexSet>(1)),
  bagCount(sample->getBagCount()),
  nCtg(sample->getNCtg()),
  defMap(make_unique<DefMap>(frame, bagCount)),
  nodeRel(false),
  idxLive(bagCount),
  relBase(vector<IndexT>(1)),
  rel2ST(vector<IndexT>(bagCount)),
  st2Split(vector<IndexT>(bagCount)),
  st2PT(vector<IndexT>(bagCount)),
  branchSense(make_unique<BranchSense>(bagCount)),
  pretree(make_unique<PreTree>(frame->getCardExtent(), bagCount)) {
  indexSet[0].initRoot(sample);
  relBase[0] = 0;
  iota(rel2ST.begin(), rel2ST.end(), 0);
  fill(st2Split.begin(), st2Split.end(), 0);
  fill(st2PT.begin(), st2PT.end(), 0);
}


unique_ptr<PreTree> Frontier::levels(const Sample* sample,
                                     const SummaryFrame* frame) {
  unique_ptr<SplitFrontier> splitFrontier = SFType::factory(frame, this, sample, SampleNux::getNCtg());
  defMap->rootDef(splitFrontier->stage(sample), bagCount);
  
  unsigned int level = 0;
  while (!indexSet.empty()) {
    splitFrontier->restageAndSplit(indexSet, defMap.get(), branchSense.get(), pretree.get());
    indexSet = splitDispatch(level++);
  }

  relFlush();
  pretree->finish(st2PT);
  return move(pretree);
}


vector<IndexSet> Frontier::splitDispatch(unsigned int level) {
  levelTerminal = (level + 1 == totLevels);

  SplitSurvey survey = nextLevel();
  for (auto & iSet : indexSet) {
    iSet.dispatch(this);
  }

  reindex(survey);
  relBase = move(succBase);

  return produce(survey.splitNext);
}


SplitSurvey Frontier::nextLevel() {//const SplitSurvey& survey) {
  SplitSurvey survey = surveyNodes(indexSet);
  succBase = vector<IndexT>(survey.succCount(indexSet.size()));
  fill(succBase.begin(), succBase.end(), idxLive); // Previous level's extent

  succLive = 0;
  succExtinct = survey.splitNext; // Pseudo-indexing for extinct sets.
  liveBase = 0;
  extinctBase = survey.idxLive;
  idxLive = survey.idxLive;

  return survey;
}


SplitSurvey Frontier::surveyNodes(vector<IndexSet>& indexSet) {
  SplitSurvey survey;
  for (auto iSet : indexSet) {
    iSet.surveySplit(levelTerminal, survey);
  }

  return survey;
}


IndexT Frontier::idxSucc(IndexT extent,
                         IndexT& offOut,
                         bool extinct) {
  IndexT succIdx;
  if (extinct || levelTerminal) { // Pseudo split caches settings.
    succIdx = succExtinct++;
    offOut = extinctBase;
    extinctBase += extent;
  }
  else {
    succIdx = succLive++;
    offOut = liveBase;
    liveBase += extent;
  }
  succBase[succIdx] = offOut;

  return succIdx;
}


void Frontier::reindex(const SplitSurvey& survey) {
  if (nodeRel) {
    nodeReindex();
  }
  else {
    nodeRel = IdxPath::localizes(bagCount, survey.idxMax);
    if (nodeRel) {
      transitionReindex(survey.splitNext);
    }
    else {
      stReindex(survey.splitNext);
    }
  }
}


void Frontier::nodeReindex() {
  vector<IndexT> succST(idxLive);
  rel2PT = vector<IndexT>(idxLive);

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1) 
    for (OMPBound splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
      indexSet[splitIdx].reindex(branchSense.get(), this, idxLive, succST);
    }
  }
  rel2ST = move(succST);
}


IndexT Frontier::relLive(IndexT relIdx,
                            IndexT targIdx,
                            IndexT path,
                            IndexT base,
                            IndexT ptIdx) {
  IndexT stIdx = rel2ST[relIdx];
  rel2PT[targIdx] = ptIdx;
  defMap->setLive(relIdx, targIdx, stIdx, path, base);

  return stIdx;
}


void Frontier::relExtinct(IndexT relIdx, IndexT ptId) {
  IndexT stIdx = rel2ST[relIdx];
  st2PT[stIdx] = ptId;
  defMap->setExtinct(relIdx, stIdx);
}


void Frontier::stReindex(IndexT splitNext) {
  unsigned int chunkSize = 1024;
  IndexT nChunk = (bagCount + chunkSize - 1) / chunkSize;

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound chunk = 0; chunk < nChunk; chunk++) {
    stReindex(defMap->getSubtreePath(), splitNext, chunk * chunkSize, (chunk + 1) * chunkSize);
  }
  }
}


void Frontier::stReindex(IdxPath* stPath,
                         IndexT splitNext,
                         IndexT chunkStart,
                         IndexT chunkNext) {
  IndexT chunkEnd = min(chunkNext, bagCount);
  for (IndexT stIdx = chunkStart; stIdx < chunkEnd; stIdx++) {
    if (stPath->isLive(stIdx)) {
      IndexT pathSucc, ptSucc;
      IndexT splitIdx = st2Split[stIdx];
      IndexT splitSucc = indexSet[splitIdx].offspring(branchSense.get(), stIdx, pathSucc, ptSucc);
      st2Split[stIdx] = splitSucc;
      stPath->setSuccessor(stIdx, pathSucc, splitSucc < splitNext);
      st2PT[stIdx] = ptSucc;
    }
  }
}


void Frontier::transitionReindex(IndexT splitNext) {
  IdxPath *stPath = defMap->getSubtreePath();
  for (IndexT stIdx = 0; stIdx < bagCount; stIdx++) {
    if (stPath->isLive(stIdx)) {
      IndexT pathSucc, idxSucc, ptSucc;
      IndexT splitIdx = st2Split[stIdx];
      IndexT splitSucc = indexSet[splitIdx].offspring(branchSense.get(), stIdx, pathSucc, idxSucc, ptSucc);
      if (splitSucc < splitNext) {
        stPath->setLive(stIdx, pathSucc, idxSucc);
        rel2ST[idxSucc] = stIdx;
      }
      else {
        stPath->setExtinct(stIdx);
      }
      st2PT[stIdx] = ptSucc;
    }
  }
}


vector<IndexSet> Frontier::produce(IndexT splitNext) {
  defMap->overlap(splitNext, bagCount, idxLive, nodeRel);
  vector<IndexSet> indexNext(splitNext);
  for (auto & iSet : indexSet) {
    iSet.succHands(this, indexNext);
  }
  return indexNext;
}


IndexT Frontier::getPTIdSucc(IndexT ptId, bool senseTrue) const {
  return pretree->getSuccId(ptId, senseTrue);
}


void Frontier::getPTIdTF(IndexT ptId, IndexT& ptTrue, IndexT& ptFalse) const {
  pretree->getSuccTF(ptId, ptTrue, ptFalse);
}


IndexRange Frontier::getBufRange(const DefCoord& preCand) const {
  return indexSet[preCand.splitCoord.nodeIdx].getBufRange();
}


void Frontier::reachingPath(IndexT splitIdx,
                            IndexT parIdx,
                            const IndexRange& bufRange,
                            IndexT relBase,
                            unsigned int path) const {
  defMap->reachingPath(splitIdx, parIdx, bufRange, relBase, path);
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
