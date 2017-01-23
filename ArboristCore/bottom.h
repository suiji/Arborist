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


class RelPath {
  const unsigned int idxLive; // Inattainable index.
  std::vector<unsigned int> relFront;
  std::vector<unsigned char> pathFront;
 public:

  RelPath(unsigned int _idxLive);

  
  /**
   */
  inline unsigned int IdxLive() {
    return idxLive;
  }


  /**
   */
  inline void Set(unsigned int idx, unsigned int relThis, unsigned int path) {
    relFront[idx] = relThis;
    pathFront[idx] = path;
  }
  
  
  /**
     @param relPrev is the relative index of a sample from the previous front level.

     @param relThis is the sample's relative index in the current front level.

     @param path is the path to the front.

     @return void.
   */
  inline void Frontify(unsigned int relPrev, unsigned int relThis, unsigned int path) {
    if (relPrev < idxLive) { // Otherwise extinct.
      Set(relPrev, relThis, path);
    }
  }


  /**

   @brief Updates path by bagged sample index:  front only.  Revises
   relative path information in reaching (first) back level.

   @param one2Front is the relative path map belonging to the first
   level.

   @param isLeft it true iff index node is an extant splitable LHS.

   @param levelOne is the RelPath belonging to the previous (first) level.

   @return void.
*/
  inline void Frontify(RelPath *one2Front, unsigned int sIdx, unsigned int relThis, bool isLeft) {
    unsigned int path = (pathFront[sIdx] << 1) | (isLeft ? 0 : 1);
    one2Front->Frontify(relFront[sIdx], relThis, path);
    Set(sIdx, relThis, path);
  }

  
  /**
     @brief Builds a single slot of the relative map passed.

     @return void.
   */
  inline void Frontify(RelPath *relPath, unsigned int mask, unsigned int idx, unsigned int frontPrev) const {
    if (frontPrev < idxLive) { // Otherwise extinct.
      relPath->Set(idx, relFront[frontPrev], pathFront[frontPrev] & mask);
    }
  }

  
  /**
     @brief Pushes one-to-front mapping back to this level.

     @param one2Front maps first level's coordinates to front.

     @return void.
   */
  inline void BackUpdate(const RelPath *one2Front, unsigned int mask) {
    for (unsigned int idx = 0; idx < idxLive; idx++) {
      one2Front->Frontify(this, mask, idx, relFront[idx]);
    }
  }

  
  /**
     @brief Determines a sample's coordinates with respect to the front level.

     @param relIdx is the relative index of the sample.

     @param off outputs the path offset of the sample in the front level.

     @return Relative coordinate of sample in front level.
   */
  inline unsigned int FrontCoord(unsigned int relIdx, unsigned int &off) const {
    off = pathFront[relIdx];
    return relFront[relIdx];
  }


  inline void PathMask(unsigned int pathMask) {
    for (unsigned int idx = 0; idx < idxLive; idx++) {
      pathFront[idx] &= pathMask;
    }
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

  static constexpr unsigned int pathMax = 8 * sizeof(unsigned char);
  static constexpr unsigned int noPath = 1 << pathMax;

  
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
  const unsigned int idxLive; // Total # sample indices at level.
  unsigned int defCount; // # live definitions.
  unsigned char del; // Position in deque.  Increments.

  // Persistent:
  std::vector<Cell> cell; // Stage coordinates, by node.
  std::vector<MRRA> liveDef;

  // More elegant and parsimonious to use std::map from pair to node,
  // but hashing much too slow.
  std::vector<MRRA> def; // Indexed by pair-offset.

  // Recomputed:
  RelPath *relPath;
  std::vector<PathNode> pathNode; // Indexed by <node, predictor> pair.
  std::vector<unsigned int> liveCount; // Indexed by node.
 public:

  Level(unsigned int _splitCount, unsigned int _nPred, unsigned int _noIndex, unsigned int _idxLive);
  ~Level();

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
     @brief Front path accessor.

     @return reference to front path.
   */
  inline RelPath *FrontPath() const {
    return relPath;
  }

  
  /**
     @brief Accessor for count of live sample indices.
  */
  inline unsigned int IdxLive() {
    return idxLive;
  }

  
  /**
     @brief Revises relative indexing vector.  Irregular, but data locality
     improves with depth.

     @param relPrev is the previous relative index.

     @param relThis is current relative index.

     @return void.
   */
  inline void BackUpdate(const RelPath *one2Front) {
    relPath->BackUpdate(one2Front, ~BackScale(0xff));
  }

  
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


  /**
     @brief Masks all front paths in the current level.  Only necessary for
     levels not already pre-masked.

     @return void.
   */
  inline void PathMask() const {
    relPath->PathMask(~BackScale(0xff));
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
  const unsigned int nPred;
  const unsigned int nPredFac;
  const unsigned int bagCount;

  std::vector<unsigned int> prePath;

  std::deque<Level *> level;
  Level *levelFront; // Current level.
  std::vector<unsigned int> history;
  std::vector<unsigned int> historyPrev;
  std::vector<unsigned char> levelDelta;
  std::vector<unsigned char> deltaPrev;

  static constexpr double efficiency = 0.15; // Work efficiency threshold.

  RelPath *samplePath; // RelPath accessed by sample index.
  unsigned int splitPrev;
  unsigned int frontCount; // # nodes in the level about to split.
  const class PMTrain *pmTrain;
  class SamplePred *samplePred;
  class SplitPred *splitPred;  // constant?
  class SplitSig *splitSig;
  class Run *run;
  unsigned int idxLive; // # non-extinct indices in current level.
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
  void FrontUpdate(unsigned int sIdx, unsigned int relIdx, bool isLeft) const;
  void RootDef(unsigned int predIdx, unsigned int denseCount);
  void ScheduleRestage(unsigned int del, unsigned int mrraIdx, unsigned int predIdx, unsigned int runCount, unsigned int bufIdx);
  int RestageIdx(unsigned int bottomIdx);
  void RestagePath(unsigned int startIdx, unsigned int extent, unsigned int lhOff, unsigned int rhOff, unsigned int level, unsigned int predIdx);
  bool ScheduleSplit(unsigned int levelIdx, unsigned int predIdx, unsigned int &runCount, unsigned int &bufIdx);

  inline void RunCounts(const class SPNode targ[], const SPPair &mrra, unsigned int del) {
    level[del]->RunCounts(targ, mrra, this);
  }
  
  static Bottom *FactoryReg(const class PMTrain *_pmTrain, const class RowRank *_rowRank, class SamplePred *_samplePred, unsigned int _bagCount);
  static Bottom *FactoryCtg(const class PMTrain *_pmTrain, const class RowRank *_rowRank, class SamplePred *_samplePred, const std::vector<class SampleNode> &_sampleCtg, unsigned int _bagCount);
  
  Bottom(const class PMTrain *_pmTrain, class SamplePred *_samplePred, class SplitPred *_splitPred, unsigned int _bagCount, unsigned int _stageSize);
  ~Bottom();
  void Overlap(unsigned int _splitCount, unsigned int idxLive);
  void LevelInit();

  void LevelClear();
  const std::vector<class SSNode*> Split(class Index *index, std::vector<class IndexNode> &indexNode);
  void ReachingPath(unsigned int _splitIdx, unsigned int path, unsigned int levelIdx, unsigned int start, unsigned int extent);
  void SSWrite(unsigned int levelIdx, unsigned int predIdx, unsigned int setPos, unsigned int bufIdx, const class NuxLH &nux) const;
  void BackUpdate() const;
  unsigned int FlushRear();
  void DefForward(unsigned int levelIdx, unsigned int predIdx);
  void Buffers(const SPPair &mrra, unsigned int bufIdx, SPNode *&source, unsigned int *&relIdxSource, SPNode *&targ, unsigned int *&relIdxTarg) const;
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


  inline RelPath *FrontPath(unsigned int del) const {
    return del == 0 ? samplePath : level[del]->FrontPath();
  }

  
};


#endif

