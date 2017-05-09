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
  const bool nodeRel;  // Subtree- or node-relative indexing.

  unsigned int defCount; // # live definitions.
  unsigned char del; // Position in deque.  Increments.

  // Persistent:
  std::vector<IndexAnc> indexAnc; // Stage coordinates, by node.
  std::vector<MRRA> liveDef;

  // More elegant and parsimonious to use std::map from pair to node,
  // but hashing much too slow.
  std::vector<MRRA> def; // Indexed by pair-offset.

  // Recomputed:
  class IdxPath *relPath;
  std::vector<class NodePath> nodePath; // Indexed by <node, predictor> pair.
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
  void SetExtinct(unsigned int idx);
  bool Backdate(const class IdxPath *one2Front);
  void SetLive(unsigned int idx, unsigned int path, unsigned int targIdx, unsigned int ndBase);


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
  inline class IdxPath *FrontPath() const {
    return relPath;
  }

  
  /**
     @brief Accessor for count of live sample indices.
  */
  inline unsigned int IdxLive() {
    return idxLive;
  }


  /**
     @brief Will overflow if level sufficiently fat.
     TODO:  switch to depth-first in such regimes.

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
     @brief Produces mask approprate for level:  lowest 'del' bits high.

     @return bit mask value.
   */
  inline unsigned int PathMask() const {
    return BackScale(1) - 1;
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
  std::vector<unsigned int> termST; // Frontier subtree indices.
  std::vector<class TermKey> termKey; // Frontier map keys:  uninitialized.
  //unsigned int termTop; // Next unused terminal index.
  bool nodeRel; // Subtree- or node-relative indexing.  Sticky, once node-.
  
  std::vector<unsigned int> prePath;

  static constexpr double efficiency = 0.15; // Work efficiency threshold.

  IdxPath *stPath; // IdxPath accessed by subtree.
  unsigned int splitPrev;
  unsigned int splitCount; // # nodes in the level about to split.
  const class PMTrain *pmTrain;
  class SamplePred *samplePred;
  class SplitPred *splitPred;  // constant?
  class SplitSig *splitSig;
  class Run *run;
  //  unsigned int idxLive; // # non-extinct indices in current level.
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
  SPNode *RestageNdxDense(unsigned int reachOffset[], const unsigned int reachBase[], const SPPair &mrra, unsigned int bufIdx, unsigned int del);
  SPNode *RestageStxDense(unsigned int reachOffset[], const SPPair &mrra, unsigned int bufIdx, unsigned int del);
  void Backdate() const;
  void ArgMax(const class IndexLevel &index, std::vector<class SSNode*> &argMax);

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


  inline unsigned int PathMask(unsigned int del) const {
    return level[del]->PathMask();
  }


 //unsigned int rhIdxNext; // GPU client only:  Starting RHS index.

 public:
  bool NonTerminal(class PreTree *preTree, class SSNode *ssNode, unsigned int extent, unsigned int ptId, double &sumExpl);
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
  void Split(class IndexLevel &index, std::vector<class SSNode*> &argMax);
  void Terminal(unsigned int extent, unsigned int ptId);
  void Overlap(class PreTree *preTree, unsigned int splitNext, unsigned int leafNext);
  void LevelPrepare(unsigned int splitNext, unsigned int idxLive, unsigned int idxMax);
  double BlockReplay(unsigned int predIdx, unsigned int sourceBit, unsigned int start, unsigned int extent);
  void Reindex(class IndexLevel *indexLevel);
  void ReindexST(class IndexLevel &indexLvel, std::vector<unsigned int> &succST);
  void ReachingPath(unsigned int levelIdx, unsigned int parIdx, unsigned int start, unsigned int extent, unsigned int ptId, unsigned int path);
  void SSWrite(unsigned int levelIdx, unsigned int predIdx, unsigned int setPos, unsigned int bufIdx, const class NuxLH &nux) const;
  unsigned int FlushRear();
  void DefForward(unsigned int levelIdx, unsigned int predIdx);
  void Buffers(const SPPair &mrra, unsigned int bufIdx, SPNode *&source, unsigned int *&relIdxSource, SPNode *&targ, unsigned int *&relIdxTarg) const;
  void Restage();
  bool IsFactor(unsigned int predIdx) const;
  void SetLive(unsigned int ndx, unsigned int targIdx, unsigned int stx, unsigned int path, unsigned int ndBase);
  void SetExtinct(unsigned int termIdx, unsigned int stIdx);
  void SubtreeFrontier(class PreTree *preTree) const;
  void Terminal(unsigned int termBase, unsigned int extent, unsigned int ptId);

  /**
     @brief Terminates node-relative path an extinct index.  Also
     terminates subtree-relative path if currently live.

     @param nodeIdx is a node-relative index.

     @return void.
  */
  void SetExtinct(unsigned int nodeIdx, unsigned int termIdx, unsigned int stIdx) {
    levelFront->SetExtinct(nodeIdx);
    SetExtinct(termIdx, stIdx);
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
    return del == 0 ? levelIdx : history[levelIdx + (del-1) * splitCount];
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


  inline unsigned int SplitCount() const {
    return splitCount;
  }


  inline Level *LevelFront() const {
    return levelFront;
  }
};


#endif

