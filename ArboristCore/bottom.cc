// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bottom.cc

   @brief Methods involving the most recently trained tree levels.

   @author Mark Seligman
 */

#include "bottom.h"
#include "level.h"
#include "bv.h"
#include "index.h"
#include "splitpred.h"
#include "samplepred.h"
#include "sample.h"
#include "framemap.h"
#include "runset.h"
#include "rowrank.h"
#include "path.h"
#include "splitsig.h"

#include <numeric>
#include <algorithm>


Bottom::Bottom(const FrameTrain *_frameTrain,
	       const RowRank *_rowRank,
	       SplitPred *_splitPred,
	       vector<StageCount> &stageCount,
	       unsigned int _bagCount) :
  nPred(_frameTrain->NPred()),
  nPredFac(_frameTrain->NPredFac()),
  bagCount(_bagCount),
  stPath(new IdxPath(bagCount)),
  splitPrev(0), splitCount(1),
  frameTrain(_frameTrain),
  rowRank(_rowRank),
  noRank(rowRank->NoRank()),
  splitPred(_splitPred),
  run(splitPred->Runs()),
  history(vector<unsigned int>(0)),
  levelDelta(vector<unsigned char>(nPred)),
  levelFront(new Level(1, nPred,rowRank->DenseIdx(), rowRank->NPredDense(), bagCount, bagCount, false, this)),
  runCount(vector<unsigned int>(nPredFac))
{
  level.push_front(levelFront);
  levelFront->Ancestor(0, 0, bagCount);
  fill(levelDelta.begin(), levelDelta.end(), 0);
  fill(runCount.begin(), runCount.end(), 0);
  RootDef(stageCount);
}


void Bottom::RootDef(const vector<StageCount> &stageCount) {
  const unsigned int bufIdx = 0; // Initial staging buffer index.
  const unsigned int splitIdx = 0;
  for (unsigned int predIdx = 0; predIdx < stageCount.size(); predIdx++) {
    bool singleton = stageCount[predIdx].singleton;
    unsigned int expl = stageCount[predIdx].expl;
    (void) levelFront->Define(splitIdx, predIdx, bufIdx, singleton, bagCount - expl);
    SetRunCount(splitIdx, predIdx, false, singleton ? 1 : frameTrain->FacCard(predIdx));
  }
}


void Bottom::Split(SamplePred *samplePred,
		   IndexLevel *index,
		   vector<SSNode> &argMax) {
  unsigned int supUnFlush = FlushRear();
  levelFront->Candidates(index, splitPred);

  Backdate();
  Restage(samplePred);

  // Reaching levels must persist through restaging ut allow path lookup.
  //
  for (unsigned int off = level.size() -1 ; off > supUnFlush; off--) {
    delete level[off];
    level.pop_back();
  }
  splitPred->ScheduleSplits(index, levelFront);
  splitPred->Split(samplePred, argMax);
}



unsigned int Bottom::FlushRear() {
  unsigned int supUnFlush = level.size() - 1;

  // Capacity:  1 front level + 'pathMax' back levels.
  // If at capacity, every reaching definition should be flushed
  // to current level ut avoid falling off the deque.
  // Flushing prior to split assignment, rather than during, should
  // also save lookup time, as all definitions reaching from rear are
  // now at current level.
  //
  if ((level.size() > NodePath::pathMax)) {
    level.back()->Flush();
    supUnFlush--;
  }

  // Walks backward from rear, purging non-reaching definitions.
  // Stops when a level with no non-reaching nodes is encountered.
  //
  for (unsigned int off = supUnFlush; off > 0; off--) {
    if (!level[off]->NonreachPurge())
      break;
  }

  unsigned int backDef = 0;
  for (unsigned int off = supUnFlush; off > 0; off--) {
    backDef += level[off]->DefCount();
  }
  unsigned int thresh = backDef * efficiency;

  for (unsigned int off = supUnFlush; off > 0; off--) {
    if (level[off]->DefCount() <= thresh) {
      thresh -= level[off]->DefCount();
      level[off]->Flush();
      supUnFlush--;
    }
    else {
      break;
    }
  }

  return supUnFlush;
}


void Bottom::ScheduleRestage(unsigned int del,
			     unsigned int mrraIdx,
			     unsigned int predIdx,
			     unsigned bufIdx) {
  SPPair mrra = make_pair(mrraIdx, predIdx);
  RestageCoord rsCoord;
  rsCoord.Init(mrra, del, bufIdx);
  restageCoord.push_back(rsCoord);
}


Bottom::~Bottom() {
  for (auto *defLevel : level) {
    defLevel->Flush(false);
    delete defLevel;
  }
  level.clear();

  delete stPath;
}


void Bottom::Restage(SamplePred *samplePred) {
  int nodeIdx;

#pragma omp parallel default(shared) private(nodeIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (nodeIdx = 0; nodeIdx < int(restageCoord.size()); nodeIdx++) {
      Restage(samplePred, restageCoord[nodeIdx]);
    }
  }

  restageCoord.clear();
}


void Bottom::Restage(SamplePred *samplePred, RestageCoord &rsCoord) {
  unsigned int del, bufIdx;
  SPPair mrra;
  rsCoord.Ref(mrra, del, bufIdx);
  samplePred->Restage(level[del], levelFront, mrra, bufIdx);
}


unsigned int Bottom::FacStride(unsigned int predIdx,
			       unsigned int nStride,
			       bool &isFactor) const {
  return frameTrain->FacStride(predIdx, nStride, isFactor);
}


void Bottom::LevelInit(IndexLevel *index) {
  splitPred->LevelInit(index);
}


void Bottom::LevelClear() {
  splitPred->LevelClear();
}


void Bottom::Overlap(unsigned int splitNext,
		     unsigned int idxLive,
		     bool nodeRel) {
  splitPrev = splitCount;
  splitCount = splitNext;
  if (splitCount == 0) // No further splitting or restaging.
    return;

  levelFront = new Level(splitCount, nPred, rowRank->DenseIdx(), rowRank->NPredDense(), bagCount, idxLive, nodeRel, this);
  level.push_front(levelFront);

  historyPrev = move(history);
  history = move(vector<unsigned int>(splitCount * (level.size()-1)));

  deltaPrev = move(levelDelta);
  levelDelta = move(vector<unsigned char>(splitCount * nPred));

  runCount = move(vector<unsigned int>(splitCount * nPredFac));
  fill(runCount.begin(), runCount.end(), 0);

  // Recomputes paths reaching from non-front levels.
  //
  for (unsigned int i = 1; i < level.size(); i++) {
    level[i]->Paths();
  }
}


void Bottom::Backdate() const {
  if (level.size() > 2 && level[1]->NodeRel()) {
    for (auto lv = level.begin() + 2; lv != level.end(); lv++) {
      if (!(*lv)->Backdate(FrontPath(1))) {
	break;
      }
    }
  }
}

  
void Bottom::ReachingPath(unsigned int levelIdx,
			  unsigned int parIdx,
			  unsigned int start,
			  unsigned int extent,
			  unsigned int relBase,
			  unsigned int path) {
  for (unsigned int backLevel = 0; backLevel < level.size() - 1; backLevel++) {
    history[levelIdx + splitCount * backLevel] = backLevel == 0 ? parIdx : historyPrev[parIdx + splitPrev * (backLevel - 1)];
  }

  Inherit(levelIdx, parIdx);
  levelFront->Ancestor(levelIdx, start, extent);
  
  // Places <levelIdx, start> pair at appropriate position in every
  // reaching path.
  //
  for (unsigned int i = 1; i < level.size(); i++) {
    level[i]->PathInit(this, levelIdx, path, start, extent, relBase);
  }
}


void Bottom::SetLive(unsigned int ndx,
		     unsigned int targIdx,
		     unsigned int stx,
		     unsigned int path,
		     unsigned int ndBase) {
  levelFront->SetLive(ndx, path, targIdx, ndBase);

  if (!level.back()->NodeRel()) {
    stPath->SetLive(stx, path, targIdx);  // Irregular write.
  }
}


void Bottom::SetExtinct(unsigned int nodeIdx,
			unsigned int stIdx) {
  levelFront->SetExtinct(nodeIdx);
  SetExtinct(stIdx);
}


void Bottom::SetExtinct(unsigned int stIdx) {
  if (!level.back()->NodeRel()) {
    stPath->SetExtinct(stIdx);
  }
}


unsigned int Bottom::SplitCount(unsigned int del) const {
  return level[del]->SplitCount();
}


void Bottom::AddDef(unsigned int reachIdx,
		    unsigned int predIdx,
		    unsigned int bufIdx,
		    bool singleton) {
  if (levelFront->Define(reachIdx, predIdx, bufIdx, singleton)) {
    levelDelta[reachIdx * nPred + predIdx] = 0;
  }
}
  

unsigned int Bottom::History(const Level *reachLevel,
			     unsigned int splitIdx) const {
  return reachLevel == levelFront ? splitIdx : history[splitIdx + (reachLevel->Del() - 1) * splitCount];
}


/**
   Passes through to front level.
 */
unsigned int Bottom::AdjustDense(unsigned int levelIdx,
				 unsigned int predIdx,
				 unsigned int &startIdx,
				 unsigned int &extent) const {
    return levelFront->AdjustDense(levelIdx, predIdx, startIdx, extent);
}


IdxPath *Bottom::FrontPath(unsigned int del) const {
  return level[del]->FrontPath();
}


/**
   Passes through to front level.
 */
bool Bottom::Singleton(unsigned int levelIdx,
		       unsigned int predIdx) const {
  return levelFront->Singleton(levelIdx, predIdx);
}


void Bottom::SetSingleton(unsigned int splitIdx,
			  unsigned int predIdx) const {
  levelFront->SetSingleton(splitIdx, predIdx);
}


void Bottom::ReachFlush(unsigned int splitIdx,
			unsigned int predIdx) const {
  Level *reachLevel = ReachLevel(splitIdx, predIdx);
  reachLevel->FlushDef(History(reachLevel, splitIdx), predIdx);
}


double Bottom::Prebias(unsigned int splitIdx,
		       double sum,
		       unsigned int sCount) const {
  return splitPred->Prebias(splitIdx, sum, sCount);
}
