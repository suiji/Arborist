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
   @brief Records index, start and extent for path reached from MRRA.
 */
class NodePath {
  unsigned int levelIdx; // < noIndex iff path extinct.
  unsigned int idxStart; // Target offset for path.
  unsigned int extent;
  unsigned int relBase; // Dense starting position.
 public:

  static constexpr unsigned int pathMax = 8 * sizeof(unsigned char) - 1;
  static constexpr unsigned int noPath = 1 << pathMax;

  
  /**
     @brief Sets to non-extinct path coordinates.
   */
  inline void Init(unsigned int _levelIdx, unsigned int _idxStart, unsigned int _extent, unsigned int _relBase) {
    levelIdx = _levelIdx;
    idxStart = _idxStart;
    extent = _extent;
    relBase = _relBase;
  }
  

  inline void Coords(unsigned int &_levelIdx, unsigned int &_idxStart, unsigned int &_extent) const {
    _levelIdx = levelIdx;
    _idxStart = idxStart;
    _extent = extent;
  }

  
  inline unsigned int IdxStart() const {
    return idxStart;
  }


  inline unsigned int Extent() const {
    return extent;
  }
  

  inline unsigned int RelBase() const {
    return relBase;
  }


  inline unsigned int Idx() const {
    return levelIdx;
  }
};


class IdxPath {
  const unsigned int idxLive; // Inattainable index.
  static constexpr unsigned int maskExtinct = NodePath::noPath;
  static constexpr unsigned int maskLive = maskExtinct - 1;
  static constexpr unsigned int relMax = 1 << 15;
  std::vector<unsigned int> relFront;
  std::vector<unsigned char> pathFront;

  // Narrow for data locality.  Can be generalized to multiple
  // sizes to accommodate more sophisticated hierarchies.
  //
  std::vector<unsigned int16_t> offFront;
 public:

  IdxPath(unsigned int _idxLive);

  /**
     @brief When appropriate, introduces relative indexing at the cost
     of trebling span of memory accesses:  char -> char + int16.

     @return True iff relative indexing expected to be profitable.
   */
  static inline bool Relable(unsigned int bagCount, unsigned int idxMax) {
    return false;//idxMax > relMax || bagCount <= 3 * relMax ? false : true;
  }

  
  /**
   */
  inline unsigned int IdxLive() const {
    return idxLive;
  }


  inline void Set(unsigned int idx, unsigned int path) {
    pathFront[idx] = path;
  }


  /**
     @brief Marks path as extinct, sets front index to inattainable value.
     Other values undefined.

     @return void.
   */
  inline void Extinct(unsigned int idx) {
    Set(idx, maskExtinct);
    relFront[idx] = idxLive;
  }

  
  /**
   */
  inline void Set(unsigned int idx, unsigned int path, unsigned int relThis) {
    Set(idx, path);
    relFront[idx] = relThis;
  }


  inline unsigned int RelFront(unsigned int idx) {
    return relFront[idx];
  }

  
  /**
   */
  inline void Set(unsigned int idx, unsigned int path, unsigned int relThis, unsigned int offRel) {
    relFront[idx] = relThis;
    pathFront[idx] = path;
    offFront[idx] = offRel;
  }

  
  /**
     @brief Accumulates a path bit vector.

     @return shift-stamped path if live else fixed extinct mask.
   */
  inline static unsigned int PathNext(unsigned int pathPrev, bool isLive, bool isLeft) {
    return  isLive ? ((maskLive & (pathPrev << 1)) | (isLeft ? 0 : 1)) : maskExtinct;
  }
  

  /**

   @brief Updates path by index:  front only.  Flags
   extinct paths explicitly.  Relative indices maintained for front
   level only.

   @param relThis is the revised relative index.

   @param isLive denotes whether the path is extinct.

   @param isLeft it true iff index node is an extant splitable LHS.

   @return void.
*/
  inline void Live(unsigned int idx, bool isLeft, unsigned int relIdx) {
    Set(idx, PathNext(pathFront[idx], true, isLeft), relIdx);
  }


  /*
*
     @brief As above, but initializes paths for back propagation.  Relies on
     relative index to flag extinct paths.
// Simplifies to FrontifySdx when one2Front eliminated.
  inline void FrontifyRel(IdxPath *one2Front, unsigned int sIdx, unsigned int relThis, unsigned int relBase, bool isLive, bool isLeft) {
    unsigned int path = PathNext(pathFront[sIdx], isLive, isLeft);
    //    one2Front->Frontify(relFront[sIdx], relThis, path, relBase);
    Set(sIdx, relThis, path);
  }
*/
  
  /**
     @param relPrev is the relative index of a sample from the previous front level.

     @param relThis is the sample's relative index in the current front level.

     @param path is the path to the front.

     @return void.
   */
  inline void Frontify(unsigned int relPrev, unsigned int relThis, unsigned int path, unsigned int relBase) {
    if (relPrev < idxLive) { // Otherwise extinct.
      Set(relPrev, path, relThis, relThis - relBase);
    }
  }


  /**
     @brief Builds a single slot of the relative map passed.

     @return void.
   */
  inline void Frontify(IdxPath *relPath, unsigned int idx, unsigned int frontPrev) const {
    if (frontPrev < idxLive) { // Otherwise extinct.
      relPath->Set(idx, relFront[frontPrev], pathFront[frontPrev], offFront[frontPrev]);
    }
  }

  
  /**
     @brief Pushes one-to-front mapping back to this level.

     @param one2Front maps first level's coordinates to front.

     @return void.
   */
  inline void BackUpdate(const IdxPath *one2Front) {
    for (unsigned int idx = 0; idx < idxLive; idx++) {
      one2Front->Frontify(this, idx, relFront[idx]);
    }
  }

  
  /**
     @brief Determines a sample's coordinates with respect to the front level.

     @param relIdx is the relative index of the sample.

     @param path outputs the path offset of the sample in the front level.

     @return true iff sample at index lies on live path.
   */
  inline bool RelLive(unsigned int relIdx, unsigned int &path, unsigned int &offRel) {
    path = pathFront[relIdx];
    offRel = offFront[relIdx];

    return path != maskExtinct;
  }

  
  /**
     @brief Looks up the path leading to the front level.

     @param idx indexes the path vector.

     @param path outputs the path to front level.

     @return true iff path to front is not extinct.
   */
  inline bool PathFront(unsigned int idx, unsigned int &path) const {
    path = pathFront[idx];
    
    return path != maskExtinct;
  }
};


/**
   @brief Coordinates from ancestor IndexSet.
 */
class IndexAnc {
  unsigned int start;
  unsigned int extent;
 public:

  inline void Init(unsigned int _start, unsigned int _extent) {
    start = _start;
    extent = _extent;
  }

  
  /**
     @brief Dual field accessor, specific to sample indexing.
   */
  inline void Ref(unsigned int &_start, unsigned int &_extent) {
    _start = start;
    _extent = extent;
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
  const bool nodeRel;  // Subtree (absolute) or node-relative indexing.

  unsigned int defCount; // # live definitions.
  unsigned char del; // Position in deque.  Increments.

  // Persistent:
  std::vector<IndexAnc> indexAnc; // Stage coordinates, by node.
  std::vector<MRRA> liveDef;

  // More elegant and parsimonious to use std::map from pair to node,
  // but hashing much too slow.
  std::vector<MRRA> def; // Indexed by pair-offset.

  // Recomputed:
  IdxPath *relPath;
  std::vector<NodePath> nodePath;; // Indexed by <node, predictor> pair.
  std::vector<unsigned int> liveCount; // Indexed by node.
 public:

  Level(unsigned int _splitCount, unsigned int _nPred, unsigned int _noIndex, unsigned int _idxLive, bool _nodeRel);
  ~Level();

  
  void Flush(class Bottom *bottom, bool forward = true);
  void FlushDef(class Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx);
  bool NonreachPurge();
  void Paths();
  void PathInit(const class Bottom *bottom, unsigned int levelIdx, unsigned int path, unsigned int start, unsigned int extent, unsigned int relBase);
  void Bounds(const SPPair &mrra, unsigned int &startIdx, unsigned int &extent);
  void FrontDef(class Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx, unsigned int runCount, unsigned int sourceBit);
  void OffsetClone(const SPPair &mrra, unsigned int reachOffset[], unsigned int reachBase[]);
  unsigned int DiagRestage(const SPPair &mrra, unsigned int reachOffset[]);
  void RunCounts(const class SPNode targ[], const SPPair &mrra, const class Bottom *bottom) const ;
  void SetRuns(const class Bottom *bottom, unsigned int levelIdx, unsigned int predIdx, unsigned int idxStart, unsigned int idxCount, const class SPNode *targ);
  void PackDense(unsigned int idxLeft, const unsigned int pathCount[], Level *levelFront, const SPPair &mrra, unsigned int reachOffset[]) const;


  /**
     @brief Accessor for indexing mode.  Currently two-valued.
   */
  inline bool NodeRel() {
    return nodeRel;
  }

  
  /**
     @brief Front path accessor.

     @return reference to front path.
   */
  inline IdxPath *FrontPath() const {
    return relPath;
  }

  
  /**
     @brief Accessor for count of live sample indices.
  */
  inline unsigned int IdxLive() {
    return idxLive;
  }


  inline void Extinct(unsigned int idx) {
    relPath->Extinct(idx);
  }


  /**
     @brief Invoked from level 0, constructs level 1's rel-to-front mapping
     for live indices.  Essential for later transition to node-relative
     indexing mode.

     @return void.
   */
  inline void Live(unsigned int idx, bool isLeft, unsigned int targIdx) {
    relPath->Live(idx, isLeft, targIdx);
  }

  
  /**
     @brief Revises relative indexing vector.  Irregular, but data locality
     improves with depth.

     @param relPrev is the previous relative index.

     @param relThis is current relative index.

     @return void.
   */
  inline void BackUpdate(const IdxPath *one2Front) {
    relPath->BackUpdate(one2Front);
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
     @brief Establishes front-level IndexSet as future ancestor.

     @return void.
  */
  void Ancestor(unsigned int levelIdx, unsigned int start, unsigned int extent) {
    indexAnc[levelIdx].Init(start, extent);
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
  std::vector<unsigned int> st2Rel; // TO GO.Next preplay relIdx:  stIndex only.
  std::vector<unsigned int> termST; // Frontier subtree indices.
  std::vector<class TermKey> termKey; // Frontier map keys:  uninitialized.
  unsigned int termTop; // Next unused terminal index.
  bool nodeRel; // Subtree- or node-relative indexing.  Sticky, once node-.
  
  std::vector<unsigned int> prePath;

  static constexpr double efficiency = 0.15; // Work efficiency threshold.

  IdxPath *stPath; // IdxPath accessed by subtree.
  unsigned int splitPrev;
  unsigned int frontCount; // # nodes in the level about to split.
  unsigned int levelBase; // previous PreTree high watermark.
  unsigned int ptHeight; // PreTree high watermark.
  const class PMTrain *pmTrain;
  class SamplePred *samplePred;
  class SplitPred *splitPred;  // constant?
  class SplitSig *splitSig;
  class Run *run;
  unsigned int idxLive; // # non-extinct indices in current level.
  unsigned int liveBase; // Accumulates index count.
  unsigned int extinctBase; // Leading index of extinct nodes.
  std::vector<unsigned int> rel2ST; // Maps to subtree index.
  std::vector<unsigned int> succST;  // Overlaps, moves to, rel2ST.
  std::vector<unsigned int> relBase; // Node-to-relative index.
  std::vector<unsigned int> succBase; // Overlaps, then moves to relBase.
  class BV *replayExpl; // Whether sample employs explicit replay.
  std::vector<unsigned int> history;
  std::vector<unsigned int> historyPrev;
  std::vector<unsigned char> levelDelta;
  std::vector<unsigned char> deltaPrev;
  Level *levelFront; // Current level.
  std::deque<Level *> level;
  
  std::vector<RestageCoord> restageCoord;

  // Restaging methods.
  void Restage(RestageCoord &rsCoord);
  SPNode *Restage(SPPair mrra, unsigned int bufIdx, unsigned int del);
  SPNode *RestageRelDense(unsigned int reachOffset[], const unsigned int reachBase[], const SPPair &mrra, unsigned int bufIdx, unsigned int del);
  SPNode *RestageRelOne(unsigned int reachOffset[], const unsigned int reachBase[], const SPPair &mrra, unsigned int bufIdx);
  SPNode *RestageRelGen(unsigned int reachOffset[], const unsigned int reachBase[], const SPPair &mrra, unsigned int bufIdx, unsigned int del);
  SPNode *RestageSdxDense(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx, unsigned int del);
  SPNode *RestageSdxOne(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx);
  SPNode *RestageSdxGen(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx, unsigned int del);
  void BackUpdate() const;
  unsigned int DiagReplay(unsigned int offExpl, unsigned int offImpl, bool leftExpl, unsigned int lhExtent, unsigned int rhExtent, unsigned int ptExpl, unsigned int ptImpl);
  void SplitPrepare(unsigned int splitNext);
  unsigned int IdxMax() const;


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


  /**
     @brief Looks up a node's relative offset for a crescent level.

     @param ptId is the index of a PTNode in a crescent level.

     @return offset of node from base of crescent level.
   */
  inline unsigned int OffsetSucc(unsigned int ptId) const {
    return ptId - ptHeight;
  }


  inline unsigned int SuccBase(unsigned int ptId) const {
    return succBase[OffsetSucc(ptId)];
  }
 

 //unsigned int rhIdxNext; // GPU client only:  Starting RHS index.

 public:
  double NonTerminal(class PreTree *preTree, class SSNode *ssNode, unsigned int extent, unsigned int lhExtent, double sum, unsigned int &ptId);
  void FrontUpdate(unsigned int sIdx, bool isLeft, unsigned int relBase, unsigned int &relIdx);
  void RootDef(unsigned int predIdx, unsigned int denseCount);
  void ScheduleRestage(unsigned int del, unsigned int mrraIdx, unsigned int predIdx, unsigned int runCount, unsigned int bufIdx);
  int RestageIdx(unsigned int bottomIdx);
  void RestagePath(unsigned int startIdx, unsigned int extent, unsigned int lhOff, unsigned int rhOff, unsigned int level, unsigned int predIdx);
  bool ScheduleSplit(unsigned int levelIdx, unsigned int predIdx, unsigned int &runCount, unsigned int &bufIdx);

  static Bottom *FactoryReg(const class PMTrain *_pmTrain, const class RowRank *_rowRank, class SamplePred *_samplePred, unsigned int _bagCount);
  static Bottom *FactoryCtg(const class PMTrain *_pmTrain, const class RowRank *_rowRank, class SamplePred *_samplePred, const std::vector<class SampleNode> &_sampleCtg, unsigned int _bagCount);
  
  Bottom(const class PMTrain *_pmTrain, class SamplePred *_samplePred, class SplitPred *_splitPred, unsigned int _bagCount, unsigned int _stageSize);
  ~Bottom();
  void LevelInit();
  void LevelClear();
  const std::vector<class SSNode*> Split(class IndexLevel *index, std::vector<class IndexSet> &indexSet);
  void Terminal(unsigned int extent, unsigned int ptId);
  void LevelSucc(class PreTree *preTree, unsigned int splitNext, unsigned int leafNext, unsigned int idxExtent, unsigned int idxLive, bool terminal);
  void Overlap(unsigned int splitNext);
  void Successor(unsigned int extent, unsigned int ptId);
  double BlockPreplay(unsigned int predIdx, unsigned int sourceBit, unsigned int start, unsigned int extent);
  void Replay(const class PreTree *preTree, unsigned int ptId, bool leftExpl, unsigned int lhExtent, unsigned int rhExtent);
  void ReachingPath(unsigned int levelIdx, unsigned int parIdx, unsigned int start, unsigned int extent, unsigned int ptId, unsigned int path);
  void SSWrite(unsigned int levelIdx, unsigned int predIdx, unsigned int setPos, unsigned int bufIdx, const class NuxLH &nux) const;
  unsigned int FlushRear();
  void DefForward(unsigned int levelIdx, unsigned int predIdx);
  void Buffers(const SPPair &mrra, unsigned int bufIdx, SPNode *&source, unsigned int *&relIdxSource, SPNode *&targ, unsigned int *&relIdxTarg) const;
  double Replay(unsigned int predIdx, unsigned int targBit, unsigned int start, unsigned int end, unsigned int ptId);
  void Restage();
  bool IsFactor(unsigned int predIdx) const;
  void SubtreeFrontier(class PreTree *preTree) const;
  

  /**
     @brief Looks up a node's relative index base for a completed level.

     @return relative index base of node.
   */
  inline unsigned int RelBase(unsigned int ptId) const {
    return relBase[LevelOffset(ptId)];
  }


  /**
     @brief Recovers subtree-relative index from pretree and node-relative index.

     @return corresponding subtree-relative index.
   */
  inline unsigned int STIdx(unsigned int ptIdx, unsigned int relIdx) const {
    return rel2ST[RelBase(ptIdx) + relIdx];
  }

  
  /**
     @brief Produces subtree index from node-relative index.
   */
  inline unsigned int RelFront(unsigned int stIdx) const {
    return stPath->RelFront(stIdx);
  }


  /**
     @brief Computes a level-relative position for a PreTree node,
     assumed to reside at the current level.

     @param ptId is the node index.

     @return Level-relative position of node.
   */
  inline unsigned int LevelOffset(unsigned int ptId) const {
    return ptId - levelBase;
  }


  inline void RunCounts(const class SPNode targ[], const SPPair &mrra, unsigned int del) {
    level[del]->RunCounts(targ, mrra, this);
  }
  

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


  inline void Bounds(const SPPair &mrra, unsigned int del, unsigned int &startIdx, unsigned int &extent) const {
    level[del]->Bounds(mrra, startIdx, extent);
  }


  void OffsetClone(const SPPair &mrra, unsigned int del, unsigned int reachOffset[], unsigned int reachBase[] = 0) {
    level[del]->OffsetClone(mrra, reachOffset, reachBase);
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


  inline IdxPath *FrontPath(unsigned int del) const {
    return level[del]->FrontPath();
  }


  /**
     @brief Updates both subtree- and node-relative paths for a live index.

     @param relIdx is the associated node-relative index.

     @param stIdx is the associated subtree-relative index.

     @param isLeft is true iff index maps to the left branch.

     @param targIdx is the updated node-relative index.

     @return void.
   */
  inline void IdxLive(unsigned int relIdx, unsigned int stIdx, bool isLeft, unsigned int targIdx) {
    levelFront->Live(relIdx, isLeft, targIdx);

    // In addition to st-relative path, maintains an ST2Rel[]
    // mapping, useful during the overlap period.
    stPath->Live(stIdx, isLeft, targIdx);  // Irregular write.
  }


  /**
     @brief Determines whether node is live.

     @return true iff node's pretree index has been marked live.
   */
  bool IsLive(unsigned int ptId) {
    return RelBase(ptId) < idxLive;
  }

  
  /**
     @brief Terminates subtree-relative path for an extinct index.  Also
     terminates node-relative path if currently live.

     @param relIdx is a node-relative index.

     @return void.
   */
  inline void Extinct(unsigned int relIdx) {
    if (relIdx < idxLive) { // I.e., not flagged in previous level.
      levelFront->Extinct(relIdx);
    }
    unsigned int stIdx = rel2ST[relIdx];
    stPath->Extinct(stIdx);
    termST[termTop++] = stIdx;
  }
};


#endif

