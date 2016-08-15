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


/**
   @brief Records sample's recent branching path.
 */
class SamplePath {
  unsigned char extinct; // Sticky semantics.
  unsigned char path;
 public:

  SamplePath();

  inline void PathLeft() {
    path = (path << 1) | 0;
  }


  inline void PathRight() {
    path = (path << 1) | 1;
  }


  inline void PathExtinct() {
    extinct = 1;
  }
  
  
  /**
     @brief Accessor.

     @param _path outputs the path reaching the sample, if live.  Otherwise
     the value is undefined.

     @return whether sample is live.
   */
  inline bool IsLive(unsigned int &_path) const {
    _path = path;
    return extinct == 0;
  }


  inline int Path(unsigned int del) const {
    return extinct == 0 ? path & ~(0xff << del) : -1;
  }
};


/**
   @brief Coordinate pair defining most-recently-restaged cell within
   SamplePred block.
 */
class Cell {
  unsigned int start;
  unsigned int extent;
 public:

  inline void Init(unsigned int _start, unsigned int _extent) {
    start = _start;
    extent = _extent;
  }

  
  /**
     @brief Dual field accessor.
   */
  inline void Ref(unsigned int &_start, unsigned int &_extent) {
    _start = start;
    _extent = extent;
  }
};


/**
   @brief Records node and offset reached by path from MRRA.
 */
class PathNode {
  unsigned int levelIdx; // < noIndex iff path extinct.
  unsigned int offset; // Target offset for path.
 public:

  
  /**
     @brief Sets to non-extinct path coordinates.
   */
  inline void Init(unsigned int _levelIdx, unsigned int _offset) {
    levelIdx = _levelIdx;
    offset = _offset;
  }
  

  inline void Coords(unsigned int &_levelIdx, unsigned int &_offset) const {
    _offset = offset;
    _levelIdx = levelIdx;
  }

  
  inline int Offset() const {
    return offset;
  }


  inline unsigned int Idx() const {
    return levelIdx;
  }
};


typedef std::pair<unsigned int, unsigned int> SplitPair;

/**
   @brief Split/predictor coordinate pair.
 */
typedef std::pair<unsigned int, unsigned int> SPCoord;


/**
   @brief Inherited state for most-recently-restaged ancestor.
 */
class MRRA {
  static const unsigned int defBit = 1;
  static const unsigned int bufBit = 2;
  unsigned int raw;
 public:

  inline void Init(unsigned int runCount, unsigned int bufIdx) {
    raw = (runCount << 2) | (bufIdx << 1) | 1;
  }

  inline void Ref(unsigned int &runCount, unsigned int &bufIdx) {
    runCount = raw >> 2;
    bufIdx = (raw & bufBit) >> 1;
  }

  
  inline void Consume(unsigned int &runCount, unsigned int &bufIdx) {
    Ref(runCount, bufIdx);
    raw = 0;
  }

  
  inline unsigned int RunCount() {
    return raw >> 2;
  }

  inline void RunCount(unsigned int runCount) {
    raw = (runCount << 2) | (raw & 3);
  }

  
  inline bool Defined() {
    return (raw & defBit) != 0;
  }
  

  inline bool Undefine() {
    bool wasDefined = ((raw & defBit) != 0);
    raw = 0;
    return wasDefined;
  }
};

/**
   @brief Per-level reaching definitions.
 */
class Level {
  const unsigned int nPred;
  const unsigned int splitCount;
  const unsigned int noIndex; // Inattainable node index value.
  unsigned int defCount; // # live definitions.
  unsigned char del; // Position in deque.  Increments.
  // Persistent:
  std::vector<Cell> cell; // Stage coordinates, by node.
  std::vector<unsigned int> parent; // Indexed by node.

  // More elegant and parsimonious to use std::map from pair to node,
  // but hashing much too slow.
  std::vector<MRRA> def; // Indexed by pair-offset.

  // Recomputed:
  std::vector<PathNode> pathNode; // Indexed by <node, predictor> pair.
  std::vector<unsigned int> liveCount; // Indexed by node.
 public:

  Level(unsigned int _splitCount, unsigned int _nPred, unsigned int noIndex);
  ~Level();
  //  bool Defines(unsigned int &mrraIdx, unsigned int predIdx);
  void Flush(class Bottom *bottom, bool forward = true);
  void FlushDef(class Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx);
  bool NonreachPurge();
  void Paths();
  void PathInit(unsigned int &mrraIdx, unsigned int path, unsigned int levelIdx, unsigned int start);
  void Node(unsigned int levelIdx, unsigned int start, unsigned int extent, unsigned int par);
  void CellBounds(const SplitPair &mrra, unsigned int &startIdx, unsigned int &extent);
  void RootDef(unsigned int nPred);
  void FrontDef(const class Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx, unsigned int runCount, unsigned int sourceBit);
  void OffsetClone(const SplitPair &mrra, unsigned int reachOffset[]);
  void Singletons(const unsigned int reachOffset[], const class SPNode targ[], const SplitPair &mrra, Level *levelFront);  


  /**
     @brief Will overflow if level sufficiently fat:  switch to depth-first
     in such regimes.

     @return offset strided by 'nPred'.
   */
  inline unsigned int PairOffset(unsigned int mrraIdx, unsigned int predIdx) {
    return mrraIdx * nPred + predIdx;
  }


  /**
     @brief Shifts a value by the number of back-levels to compensate for
     effects of binary branching.

     @param val is the value to shift.

     @return shifted value.
   */  
  inline unsigned int BackScale(unsigned int val) {
    return val << (unsigned int) del;
  }

  
  inline unsigned int ParentIdx(unsigned int mrraIdx) {
    return parent[mrraIdx];
  }


  /**
     @brief Accessor.  What more can be said?

     @return definition count at this level.
  */
  inline unsigned int DefCount() {
    return defCount;
  }

  inline unsigned int SplitCount() {
    return splitCount;
  }


  inline void Define(unsigned int levelIdx, unsigned int predIdx, unsigned int runCount, unsigned int bufIdx) {
    def[PairOffset(levelIdx, predIdx)].Init(runCount, bufIdx);
    defCount++;
  }


  inline void Undefine(unsigned int levelIdx, unsigned int predIdx) {
    bool wasDefined = def[PairOffset(levelIdx, predIdx)].Undefine();
    defCount -= wasDefined ? 1 : 0;
  }


  inline void Consume(unsigned int levelIdx, unsigned int predIdx, unsigned int &runCount, unsigned int &bufIdx) {
    def[PairOffset(levelIdx, predIdx)].Consume(runCount, bufIdx);
    defCount--;
  }


  inline void SetRunCount(unsigned int levelIdx, unsigned int predIdx, unsigned int runCount) {
    def[PairOffset(levelIdx, predIdx)].RunCount(runCount);
  }
  

  inline void Singleton(unsigned int levelIdx, unsigned int predIdx) {
    def[PairOffset(levelIdx, predIdx)].RunCount(1);
  }
  
  
  inline bool Singleton(unsigned int levelIdx, unsigned int predIdx, unsigned int &runCount, unsigned int &bufIdx) {
    def[PairOffset(levelIdx, predIdx)].Ref(runCount, bufIdx);
    return runCount == 1;
  }


  inline void Ref(unsigned int levelIdx, unsigned int predIdx, unsigned int &runCount, unsigned int &bufIdx) {
    def[PairOffset(levelIdx, predIdx)].Ref(runCount, bufIdx);
  }


  inline bool Defined(unsigned int levelIdx, unsigned int predIdx) {
    return def[PairOffset(levelIdx, predIdx)].Defined();
  }


  /**
     @brief Adds a new definition to front level at passed coordinates, provided
   that the node index is live.

     @return true iff path reaches to current level.
   */
  inline void AddDef(unsigned int reachIdx, unsigned int predIdx, unsigned int defRC, unsigned int destBit) {
    if (reachIdx != noIndex) {
      Define(reachIdx, predIdx, defRC, destBit);
    }
  }


  /**
     @brief Tests whether this level defines split/predictor pair passed and,
     if so, forwards reaching paths.

     @param mrraIdx inputs trial level index / outputs parent index if level
     does not define pair.

     @param predIdx is the predictor index of the pair.

     @return true iff pair defined at this level.
   */
  inline bool Forwards(class Bottom *bottom, unsigned int &mrraIdx, unsigned int predIdx) {
    if (Defined(mrraIdx, predIdx)) {
      FlushDef(bottom, mrraIdx, predIdx);
      return true;
    }
    else {
      mrraIdx = parent[mrraIdx];
      return false;
    }
  }
};


/**
   @brief Encapsulates information needed to drive splitting.
 */
class SplitCoord {
  unsigned int splitPos; // Position in containing vector.
  unsigned int levelIdx;
  unsigned int predIdx;
  unsigned int runCount;
  int setPos; // runset offset, iff nonnegative.
  unsigned char bufIdx; // Buffer containing SpiltPred block.
 public:
  void Init(unsigned int _splitPos, unsigned int _levelIdx, unsigned int _predIdx, unsigned int _bufIdx, unsigned int _runCount, int _setPos) {
    splitPos = _splitPos;
    levelIdx = _levelIdx;
    predIdx = _predIdx;
    bufIdx = _bufIdx;
    runCount = _runCount;
    setPos = _runCount > 0 ? _setPos : -1;
  }

  void Ref(unsigned int &_levelIdx, unsigned int &_predIdx, int &_setPos, unsigned int &_bufIdx) const {
    _levelIdx = levelIdx;
    _predIdx = predIdx;
    _setPos = setPos;
    _bufIdx = bufIdx;
  }

  void Split(const class SamplePred *samplePred, const class IndexNode indexNode[], class SplitPred *splitPred);


  inline bool HasRuns() {
    return setPos >= 0;
  }
};


/**
   @brief Coordinates referencing most-recently restaged ancester (MRRA).
 */
class RestageCoord {
  SplitPair mrra; // Level-relative coordinates of reaching ancestor.
  unsigned int runCount;
  unsigned char del; // # levels back to referencing level.
  unsigned char bufIdx; // buffer index of mrra's SamplePred.
 public:

  void inline Init(const SplitPair &_mrra, unsigned int _del, unsigned int _runCount, unsigned int _bufIdx) {
    mrra = _mrra;
    del = _del;
    runCount = _runCount;
    bufIdx = _bufIdx;
  }

  void inline Ref(SplitPair &_mrra, unsigned int &_del, unsigned int &_runCount, unsigned int &_bufIdx) {
    _mrra = mrra;
    _del = del;
    _runCount = runCount;
    _bufIdx = bufIdx;
  }
};


/**
 */
class Bottom {
  static constexpr unsigned int pathMax = 8 * sizeof(unsigned char);
  const unsigned int nPred;
  const unsigned int nPredFac;
  const unsigned int bagCount;
  Level *levelFront; // Current level.
  std::deque<Level *> level;

  std::vector<SplitCoord> splitCoord; // Schedule of splits.
  static constexpr double efficiency = 0.15; // Work efficiency threshold.
  
  SamplePath *samplePath;
  unsigned int frontCount; // # nodes in the level about to split.
  class BV *bvLeft;
  class BV *bvDead;
  class SamplePred *samplePred;
  class SplitPred *splitPred;  // constant?
  class SplitSig *splitSig;
  class Run *run;
  std::vector<RestageCoord> restageCoord;
  //unsigned int rhIdxNext; // GPU client only:  Starting RHS index.

 public:
  void ScheduleRestage(unsigned int mrraIdx, unsigned int predIdx, unsigned int del, unsigned int runCount, unsigned int bufIdx);
  int RestageIdx(unsigned int bottomIdx);
  void RestagePath(unsigned int startIdx, unsigned int extent, unsigned int lhOff, unsigned int rhOff, unsigned int level, unsigned int predIdx);
  unsigned int ScheduleSplit(unsigned int levelIdx, unsigned int predIdx, unsigned int runTop);
  void Split(const std::vector<SplitPair> &pairNode, const class IndexNode indexNode[]);
  void Split(const class IndexNode indexNode[], unsigned int bottomIdx, int setIdx);
  inline void Singletons(const unsigned int reachOffset[], const class SPNode targ[], const SplitPair &mrra, unsigned int del) {
    level[del]->Singletons(reachOffset, targ, mrra, levelFront);
  }
  

  
 public:
  static Bottom *FactoryReg(class SamplePred *_samplePred, unsigned int _bagCount);
  static Bottom *FactoryCtg(class SamplePred *_samplePred, class SampleNode *_sampleCtg, unsigned int _bagCount);
  
  Bottom(class SamplePred *_samplePred, class SplitPred *_splitPred, unsigned int _bagCount, unsigned int _nPred, unsigned int _nPredFac);
  ~Bottom();
  void LevelInit();
  void Split(const class IndexNode indexNode[]);
  void NewLevel(unsigned int _splitCount);
  void LevelClear();
  const std::vector<class SSNode*> Split(class Index *index, class IndexNode indexNode[]);
  void ReachingPath(unsigned int _splitIdx, unsigned int path, unsigned int levelIdx, unsigned int start, unsigned int extent);
  void SSWrite(unsigned int splitPos, unsigned int lhSampCount, unsigned lhIdxCount, double info);
  void PathLeft(unsigned int sIdx) const;
  void PathRight(unsigned int sIdx) const ;
  void PathExtinct(unsigned int sIdx) const ;
  unsigned int FlushRear();
  void DefForward(unsigned int levelIdx, unsigned int predIdx);
  void Buffers(const SplitPair &mrra, unsigned int bufIdx, SPNode *&source, unsigned int *&sIdxSource, SPNode *&targ, unsigned int *&sIdxTarg) const;
  void Restage();
  void Restage(RestageCoord &rsCoord);
  SPNode *RestageOne(unsigned int reachOffset[], const SplitPair &mrra, unsigned int bufIdx);
  SPNode *RestageIrr(unsigned int reachOffset[], const SplitPair &mrra, unsigned int bufIdx, unsigned int del);
  
  /**
     @brief Accessor.  SSNode only client.
   */
  inline class Run *Runs() {
    return run;
  }


  /**
     @brief Setter methods for sample path.
   */
  inline bool IsLive(unsigned int sIdx, unsigned int &sIdxPath) const {
    return samplePath[sIdx].IsLive(sIdxPath);
  }

  inline void PathPrefetch(unsigned int *sampleIdx, unsigned int del) const {
    __builtin_prefetch(samplePath + sampleIdx[del]);
  }

  inline int Path(unsigned int sIdx, unsigned int del) const {
    return samplePath[sIdx].Path(del);
  }


  /**
     @brief Derives pair coordinates from positional index.

     @param splitIdx is split ordinal, a position within the vector of splits.

     @param levelIdx is the level-relative node index.

     @param predIdx is the predictor index.

     @return void, with reference paramters.
   */
  inline void SplitRef(unsigned int splitIdx, unsigned int &levelIdx, unsigned int &predIdx) const {
    int dummy1;
    unsigned int dummy2;
    splitCoord[splitIdx].Ref(levelIdx, predIdx, dummy1, dummy2);
  }


  /**
     @brief Variant of above, with runset position.

     @param splitIdx is split ordinal, a position within the vector of splits.

     @param levelIdx is the level-relative node index.

     @param predIdx is the predictor index.

     @param runsetPos it the position within the runset vector.

     @return void, with reference paramters.
   */
  inline void SplitRef(unsigned int splitIdx, unsigned int &levelIdx, unsigned int &predIdx, int &runsetPos) const {
    unsigned int dummy;
    splitCoord[splitIdx].Ref(levelIdx, predIdx, runsetPos, dummy);
  }


  /**
     @brief Variant of above, with runset position.

     @param splitIdx is split ordinal, a position within the vector of splits.

     @param levelIdx is the level-relative node index.

     @param predIdx is the predictor index.

     @param runsetPos it the position within the runset vector.

     @return void, with reference paramters.
   */
  inline void SplitRef(unsigned int splitIdx, unsigned int &levelIdx, unsigned int &predIdx, int &runsetPos, unsigned int &bufIdx) const {
    splitCoord[splitIdx].Ref(levelIdx, predIdx, runsetPos, bufIdx);
  }


  inline void SetRunCount(unsigned int splitIdx, unsigned int predIdx, unsigned int runCount) {
    levelFront->SetRunCount(splitIdx, predIdx, runCount);
  }

  inline void CellBounds(unsigned int del, const SplitPair &mrra, unsigned int &startIdx, unsigned int &extent) const {
    return level[del]->CellBounds(mrra, startIdx, extent);
  }


  void OffsetClone(const SplitPair &mrra, unsigned int del, unsigned int reachOffset[]) {
    level[del]->OffsetClone(mrra, reachOffset);
  }

  bool HasRuns(unsigned int splitPos) {
    return splitCoord[splitPos].HasRuns();
  }

  
  unsigned int SplitCount(unsigned int del) {
    return level[del]->SplitCount();
  }


  /**
     @brief Flips source bit if a definition reaches to current level.

     @return void
   */
  inline void AddDef(unsigned int reachIdx, unsigned int predIdx, unsigned int defRC, unsigned int destBit) const {
    levelFront->AddDef(reachIdx, predIdx, defRC, destBit);
  }

    
};


#endif

