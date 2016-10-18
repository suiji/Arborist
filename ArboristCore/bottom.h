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
  unsigned int idxStart; // Target offset for path.
  unsigned int extent;
 public:

  
  /**
     @brief Sets to non-extinct path coordinates.
   */
  inline void Init(unsigned int _levelIdx, unsigned int _idxStart, unsigned int _extent) {
    levelIdx = _levelIdx;
    idxStart = _idxStart;
    extent = _extent;
  }
  

  inline void Coords(unsigned int &_levelIdx, unsigned int &_idxStart, unsigned int &_extent) const {
    _levelIdx = levelIdx;
    _idxStart = idxStart;
    _extent = extent;
  }

  
  inline int IdxStart() const {
    return idxStart;
  }


  inline unsigned int Idx() const {
    return levelIdx;
  }
};


/**
   @brief Split/predictor coordinate pair.
 */
typedef std::pair<unsigned int, unsigned int> SPPair;

typedef std::pair<unsigned int, unsigned int> SPCoord;


/**
   @brief Inherited state for most-recently-restaged ancestor.
 */
class MRRA {
  static const unsigned int defBit = 1;
  static const unsigned int bufBit = 2;
  unsigned int raw;
  unsigned int denseMargin;
  unsigned int denseCount; // Nonincreasing.
 public:

  inline void Init(unsigned int runCount, unsigned int bufIdx, unsigned int _denseCount) {
    raw = (runCount << 2) | (bufIdx << 1) | 1;
    denseMargin = 0;
    denseCount = _denseCount;
  }

  inline void Ref(unsigned int &runCount, unsigned int &bufIdx) const {
    runCount = raw >> 2;
    bufIdx = (raw & bufBit) >> 1;
  }


  /**
     @brief Applies dense parameters to offsets derived from index node.

     @param startIdx is input as the node offset and output as the
     margin-adjusted starting index.

     @param extent is input as the node index count and output with an
     adjustment for implicit indices.

     @return dense count.
   */
  inline unsigned int AdjustDense(unsigned int &startIdx, unsigned int &extent) const {
    startIdx -= denseMargin;
    extent -= denseCount;

    return denseCount;
  }
  

  inline bool IsDense() const {
    return denseCount > 0 || denseMargin > 0;
  }

  
  inline void SetDense(unsigned int _denseMargin, unsigned int _denseCount) {
    denseMargin = _denseMargin;
    denseCount = _denseCount;
  }


  inline void Consume(unsigned int &runCount, unsigned int &bufIdx) {
    Ref(runCount, bufIdx);
    raw = 0;
  }


  /**
     @brief Run count accessor.

     Run count values are nonnegative:

     Values greater than or equal to 2 are currently reserved for
     factor-valued predictors and denote an upper limit on the number
     of runs subsumed by the pair.

     A value of zero denotes pairs for which runs are not tracked, such
     as numerical predictors having no dense rank.

     A value of one denotes a singleton, i.e., a pair which must remain
     on the books but which will not precipitate a split and so need
     not either restage or split.  Note that the method to identify a
     singleton varies with the data type.
   */
  inline unsigned int RunCount() const {
    return raw >> 2;
  }


  inline void SetRunCount(unsigned int runCount) {
    raw = (runCount << 2) | (raw & 3);
  }

  
  inline bool Defined() const {
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
  std::vector<MRRA*> def2;
  std::vector<MRRA> liveDef;
  // More elegant and parsimonious to use std::map from pair to node,
  // but hashing much too slow.
  std::vector<MRRA> def; // Indexed by pair-offset.

  // Recomputed:
  std::vector<PathNode> pathNode; // Indexed by <node, predictor> pair.
  std::vector<unsigned int> liveCount; // Indexed by node.
 public:

  Level(unsigned int _splitCount, unsigned int _nPred, unsigned int _noIndex);
  ~Level();

  void Def2();
  void Flush(class Bottom *bottom, bool forward = true);
  void FlushDef(class Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx);
  bool NonreachPurge();
  void Paths();
  void PathInit(const class Bottom *bottom, unsigned int levelIdx, unsigned int path, unsigned int start, unsigned int extent);
  void Node(unsigned int levelIdx, unsigned int start, unsigned int extent, unsigned int par);
  void CellBounds(const SPPair &mrra, unsigned int &startIdx, unsigned int &extent);
  void FrontDef(class Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx, unsigned int runCount, unsigned int sourceBit);
  void OffsetClone(const SPPair &mrra, unsigned int reachOffset[]);
  void RunCounts(const class SPNode targ[], const SPPair &mrra, const class Bottom *bottom) const ;
  void SetRuns(const class Bottom *bottom, unsigned int levelIdx, unsigned int predIdx, unsigned int idxStart, unsigned int idxCount, const class SPNode *targ);
  void PackDense(unsigned int idxLeft, const unsigned int pathCount[], Level *levelFront, const SPPair &mrra, unsigned int reachOffset[]) const;

  /**
     @brief Will overflow if level sufficiently fat:  switch to depth-first
     in such regimes.

     @return offset strided by 'nPred'.
   */
  inline unsigned int PairOffset(unsigned int mrraIdx, unsigned int predIdx) const {
    return mrraIdx * nPred + predIdx;
  }


  /**
     @brief Shifts a value by the number of back-levels to compensate for
     effects of binary branching.

     @param val is the value to shift.

     @return shifted value.
   */  
  inline unsigned int BackScale(unsigned int val) const {
    return val << (unsigned int) del;
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


  /**
     @brief

     @param denseCount is only set directly by staging.  Otherwise it has a
     default setting of zero, which is later reset by restaging.
   */
  inline bool Define(unsigned int levelIdx, unsigned int predIdx, unsigned int runCount, unsigned int bufIdx, unsigned int denseCount = 0) {
    if (levelIdx != noIndex) {
      def[PairOffset(levelIdx, predIdx)].Init(runCount, bufIdx, denseCount);
      defCount++;
      return true;
    }
    else {
      return false;
    }
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
    def[PairOffset(levelIdx, predIdx)].SetRunCount(runCount);
  }
  

  /**
     @brief Determines whether pair consists of a single run.

     @param bufIdx outputs the buffer index.

     @return true iff a singleton.
   */
  inline bool Singleton(unsigned int levelIdx, unsigned int predIdx, unsigned int &runCount, unsigned int &bufIdx) {
    def[PairOffset(levelIdx, predIdx)].Ref(runCount, bufIdx);
    return runCount == 1;
  }


  inline bool Singleton(unsigned int levelIdx, unsigned int predIdx) {
    unsigned int ignore1, ignore2;
    return Singleton(levelIdx, predIdx, ignore1, ignore2);
  }


  inline unsigned int AdjustDense(const SPPair &mrra, unsigned int &startIdx, unsigned int &extent) const {
    return def[PairOffset(mrra.first, mrra.second)].AdjustDense(startIdx, extent);
  }

  
  inline void Ref(unsigned int levelIdx, unsigned int predIdx, unsigned int &runCount, unsigned int &bufIdx) {
    def[PairOffset(levelIdx, predIdx)].Ref(runCount, bufIdx);
  }


  inline bool Defined(unsigned int levelIdx, unsigned int predIdx) {
    return def[PairOffset(levelIdx, predIdx)].Defined();
  }


  inline bool IsDense(unsigned int levelIdx, unsigned int predIdx) const {
    return def[PairOffset(levelIdx, predIdx)].IsDense();
  }

  /**
     @brief Sets the density-associated parameters for a reached node.

     @return void.
  */
  inline void SetDense(unsigned int levelIdx, unsigned int predIdx, unsigned int denseMargin, unsigned int denseCount) {
    def[PairOffset(levelIdx, predIdx)].SetDense(denseMargin, denseCount);
  }

};


/**
   @brief Coordinates referencing most-recently restaged ancester (MRRA).
 */
class RestageCoord {
  SPPair mrra; // Level-relative coordinates of reaching ancestor.
  unsigned int runCount;
  unsigned char del; // # levels back to referencing level.
  unsigned char bufIdx; // buffer index of mrra's SamplePred.
 public:

  void inline Init(const SPPair &_mrra, unsigned int _del, unsigned int _runCount, unsigned int _bufIdx) {
    mrra = _mrra;
    del = _del;
    runCount = _runCount;
    bufIdx = _bufIdx;
  }

  void inline Ref(SPPair &_mrra, unsigned int &_del, unsigned int &_runCount, unsigned int &_bufIdx) {
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
  static constexpr unsigned int noPath = 1 << pathMax;
  const unsigned int nPred;
  const unsigned int nPredFac;
  const unsigned int bagCount;
  const unsigned int stageSize;

  unsigned int *prePath;

  std::deque<Level *> level;
  Level *levelFront; // Current level.
  std::vector<unsigned int> history;
  std::vector<unsigned int> historyPrev;
  std::vector<unsigned char> levelDelta;
  std::vector<unsigned char> deltaPrev;

  static constexpr double efficiency = 0.15; // Work efficiency threshold.
  
  SamplePath *samplePath;
  unsigned int splitPrev;
  unsigned int frontCount; // # nodes in the level about to split.
  class BV *bvLeft;
  class BV *bvDead;
  const class PMTrain *pmTrain;
  class SamplePred *samplePred;
  class SplitPred *splitPred;  // constant?
  class SplitSig *splitSig;
  class Run *run;
  std::vector<RestageCoord> restageCoord;

  SPNode *RestageDense(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx, unsigned int del);


  /**
     @brief Increments reaching levels for all pairs involving node.
   */
  inline void Inherit(unsigned int levelIdx, unsigned int par) {
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      levelDelta[levelIdx * nPred + predIdx] = 1 + deltaPrev[par * nPred + predIdx];
    }
  }
//unsigned int rhIdxNext; // GPU client only:  Starting RHS index.

 public:
  void RootDef(unsigned int predIdx, unsigned int denseCount);
  void ScheduleRestage(unsigned int del, unsigned int mrraIdx, unsigned int predIdx, unsigned int runCount, unsigned int bufIdx);
  int RestageIdx(unsigned int bottomIdx);
  void RestagePath(unsigned int startIdx, unsigned int extent, unsigned int lhOff, unsigned int rhOff, unsigned int level, unsigned int predIdx);
  bool ScheduleSplit(unsigned int levelIdx, unsigned int predIdx, unsigned int &runCount, unsigned int &bufIdx);
  void Split(const std::vector<SPPair> &pairNode, const class IndexNode indexNode[]);
  void Split(const class IndexNode indexNode[], unsigned int bottomIdx, int setIdx);

  inline void RunCounts(const class SPNode targ[], const SPPair &mrra, unsigned int del) {
    level[del]->RunCounts(targ, mrra, this);
  }
  
  static Bottom *FactoryReg(const class PMTrain *_pmTrain, const class RowRank *_rowRank, class SamplePred *_samplePred, unsigned int _bagCount);
  static Bottom *FactoryCtg(const class PMTrain *_pmTrain, const class RowRank *_rowRank, class SamplePred *_samplePred, const std::vector<class SampleNode> &_sampleCtg, unsigned int _bagCount);
  
  Bottom(const class PMTrain *_pmTrain, class SamplePred *_samplePred, class SplitPred *_splitPred, unsigned int _bagCount, unsigned int _stageSize);
  ~Bottom();
  void Overlap(unsigned int _splitCount);
  void LevelInit();

  void LevelClear();
  const std::vector<class SSNode*> Split(class Index *index, class IndexNode indexNode[]);
  void ReachingPath(unsigned int _splitIdx, unsigned int path, unsigned int levelIdx, unsigned int start, unsigned int extent);
  void SSWrite(unsigned int levelIdx, unsigned int predIdx, unsigned int setPos, unsigned int bufIdx, const class NuxLH &nux) const;
  void PathLeft(unsigned int sIdx) const;
  void PathRight(unsigned int sIdx) const ;
  void PathExtinct(unsigned int sIdx) const ;
  unsigned int FlushRear();
  void DefForward(unsigned int levelIdx, unsigned int predIdx);
  void Buffers(const SPPair &mrra, unsigned int bufIdx, SPNode *&source, unsigned int *&sIdxSource, SPNode *&targ, unsigned int *&sIdxTarg) const;
  void Restage();
  void Restage(RestageCoord &rsCoord);
  SPNode *RestageOne(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx);
  SPNode *RestageIrr(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx, unsigned int del);
  bool IsFactor(unsigned int predIdx) const;
  
  
  inline void SetRuns(unsigned int levelIdx, unsigned int predIdx, unsigned int idxStart, unsigned int idxCount, const class SPNode *targ) const {
    levelFront->SetRuns(this, levelIdx, predIdx, idxStart, idxCount, targ);
  }
  
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


  inline void SetRunCount(unsigned int splitIdx, unsigned int predIdx, unsigned int runCount) const {
    levelFront->SetRunCount(splitIdx, predIdx, runCount);
  }


  inline bool IsDense(const SPPair &mrra, unsigned int del = 0) const {
    return level[del]->IsDense(mrra.first, mrra.second);
  }


  inline void CellBounds(const SPPair &mrra, unsigned int del, unsigned int &startIdx, unsigned int &extent) const {
    level[del]->CellBounds(mrra, startIdx, extent);
  }


  void OffsetClone(const SPPair &mrra, unsigned int del, unsigned int reachOffset[]) {
    level[del]->OffsetClone(mrra, reachOffset);
  }

  
  unsigned int SplitCount(unsigned int del) {
    return level[del]->SplitCount();
  }


  /**
     @brief Flips source bit if a definition reaches to current level.

     @return void
   */
  inline void AddDef(unsigned int reachIdx, unsigned int predIdx, unsigned int defRC, unsigned int destBit) {
    if (levelFront->Define(reachIdx, predIdx, defRC, destBit)) {
      levelDelta[reachIdx * nPred + predIdx] = 0;
    }
  }
  

  /**
     @brief Locates index of ancestor several levels back.

     @param levelIdx is descendant index.

     @param del is the number of levels back.

     @return index of ancestor node.
   */
  inline unsigned int History(unsigned int levelIdx, unsigned int del) const {
    return del == 0 ? levelIdx : history[levelIdx + (del-1) * frontCount];
  }


  inline unsigned int ReachLevel(unsigned int levelIdx, unsigned int predIdx) {
    return levelDelta[levelIdx * nPred + predIdx];
  }

  
  /**
     @brief Determines whether front-level pair is a singleton.

     @return true iff the pair is a singleton.
   */
  inline bool Singleton(unsigned int levelIdx, unsigned int predIdx) const {
    return levelFront->Singleton(levelIdx, predIdx);
  }


  inline unsigned int AdjustDense(unsigned int levelIdx, unsigned int predIdx, unsigned int &startIdx, unsigned int &extent) const {
    SPPair pair = std::make_pair(levelIdx, predIdx);
    return levelFront->AdjustDense(pair, startIdx, extent);
  }
};


#endif

