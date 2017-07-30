// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bottom.h

   @brief Definitions for the classes managing the most recently
   trained tree levels.

   @author Mark Seligman

 */

#ifndef ARBORIST_BOTTOM_H
#define ARBORIST_BOTTOM_H

#include <deque>
#include <vector>
#include <map>

#include "param.h"


/**
   @brief Coordinates referencing most-recently restaged ancester (MRRA).
 */
class RestageCoord {
  SPPair mrra; // Level-relative coordinates of reaching ancestor.
  unsigned char del; // # levels back to referencing level.
  unsigned char bufIdx; // buffer index of mrra's SamplePred.
 public:

  void inline Init(const SPPair &_mrra, unsigned int _del, unsigned int _bufIdx) {
    mrra = _mrra;
    del = _del;
    bufIdx = _bufIdx;
  }

  void inline Ref(SPPair &_mrra, unsigned int &_del, unsigned int &_bufIdx) {
    _mrra = mrra;
    _del = del;
    _bufIdx = bufIdx;
  }
};


/**
 */
class Bottom {
  const unsigned int nPred;
  const unsigned int nPredFac;
  const unsigned int bagCount;

  static constexpr double efficiency = 0.15; // Work efficiency threshold.

  class IdxPath *stPath; // IdxPath accessed by subtree.
  unsigned int splitPrev;
  unsigned int splitCount; // # nodes in the level about to split.
  const class PMTrain *pmTrain;
  const class RowRank *rowRank;
  const unsigned int noRank;
  class SplitPred *splitPred;  // constant?
  class SplitSig *splitSig;
  class Run *run;

  std::vector<unsigned int> history;
  std::vector<unsigned int> historyPrev;
  std::vector<unsigned char> levelDelta;
  std::vector<unsigned char> deltaPrev;
  class Level *levelFront; // Current level.
  std::vector<unsigned int> runCount;
  std::deque<class Level *> level;
  
  std::vector<RestageCoord> restageCoord;

  // Restaging methods.
  void Restage(RestageCoord &rsCoord);
  void IndexRestage(RestageCoord &rsCoord); // COPROC

  void Backdate() const;

  
  /**
     @brief Increments reaching levels for all pairs involving node.
   */
  inline void Inherit(unsigned int levelIdx, unsigned int par) {
    unsigned char *colCur = &levelDelta[levelIdx * nPred];
    unsigned char *colPrev = &deltaPrev[par * nPred];
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      colCur[predIdx] = colPrev[predIdx] + 1;
    }
  }


 public:
  void FrontUpdate(unsigned int sIdx, bool isLeft, unsigned int relBase, unsigned int &relIdx);
  void RootDef(unsigned int predIdx, unsigned int expl, bool singleton);
  void ScheduleRestage(unsigned int del, unsigned int mrraIdx, unsigned int predIdx, unsigned int bufIdx);
  int RestageIdx(unsigned int bottomIdx);
  void RestagePath(unsigned int startIdx, unsigned int extent, unsigned int lhOff, unsigned int rhOff, unsigned int level, unsigned int predIdx);

  
  Bottom(const class PMTrain *_pmTrain, const class RowRank *_rowRank, class SplitPred *_splitPred, class SamplePred *samplePred, unsigned int _bagCount);
  ~Bottom();
  void LevelInit(class IndexLevel *index);
  void LevelClear();
  void Split(const class SamplePred *samplePred, class IndexLevel *index, std::vector<class SSNode> &argMax);
  void Overlap(class SamplePred *samplePred, unsigned int splitNext, unsigned int idxLive, bool nodeRel);
  void ReachingPath(unsigned int levelIdx, unsigned int parIdx, unsigned int start, unsigned int extent, unsigned int ptId, unsigned int path);
  unsigned int FlushRear();
  void Restage();
  void IndexRestage(); // COPROC
  bool IsFactor(unsigned int predIdx) const;
  unsigned int FacIdx(unsigned int predIdx, bool &isFactor) const;
  void SetLive(unsigned int ndx, unsigned int targIdx, unsigned int stx, unsigned int path, unsigned int ndBase);
  void SetExtinct(unsigned int stIdx);
  void SetExtinct(unsigned int nodeIdx, unsigned int stIdx);
  double Prebias(unsigned int splitIdx, double sum, unsigned int sCount) const;

  
  /**
     @brief Accessor.  SSNode only client.
   */
  inline class Run *Runs() const {
    return run;
  }


  class IdxPath *STPath() const {
    return stPath;
  }
  

  inline unsigned int NoRank() const {
    return noRank;
  }


  unsigned int SplitCount(unsigned int del) const;
  void AddDef(unsigned int reachIdx, unsigned int predIdx, unsigned int bufIdx, bool singleton);
  bool Singleton(unsigned int levelIdx, unsigned int predIdx) const;
  void SetSingleton(unsigned int levelIdx, unsigned int predIdx) const;
  unsigned int AdjustDense(unsigned int levelIdx, unsigned int predIdx, unsigned int &startIdx, unsigned int &extent) const;
  class IdxPath *FrontPath(unsigned int del) const;
  void ReachFlush(unsigned int splitIdx, unsigned int predIdx) const;
  unsigned int History(const Level *reachLevel, unsigned int splitIdx) const;
  
  /**
     @brief Looks up the level containing the MRRA of a pair.
   */
  inline class Level *ReachLevel(unsigned int levelIdx, unsigned int predIdx) const {
    return level[levelDelta[levelIdx * nPred + predIdx]];
  }


  inline unsigned int SplitCount() const {
    return splitCount;
  }


  
  /**
     @brief Numeric run counts are constrained to be either 1, if singleton,
     or zero otherwise.  Singleton iff dense and all indices implicit or
     not dense and all indices have identical rank.
  */
  inline void SetRunCount(unsigned int splitIdx, unsigned int predIdx, bool hasImplicit, unsigned int rankCount) {
    bool dummy;
    unsigned int rCount = hasImplicit ? rankCount + 1 : rankCount;
    if (rCount == 1) {
      SetSingleton(splitIdx, predIdx);
    }
    if (IsFactor(predIdx)) {
      runCount[splitIdx * nPredFac + FacIdx(predIdx, dummy)] = rCount;
    }
  }


  inline unsigned int RunCount(unsigned int splitIdx, unsigned int predIdx) const {
    bool isFactor;
    unsigned int facIdx = FacIdx(predIdx, isFactor);
    return isFactor ? runCount[splitIdx * nPredFac + facIdx] : 0;
  }
};


#endif

