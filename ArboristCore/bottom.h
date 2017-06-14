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
   @brief Defines the parameters needed to place a dense cell with respect
   the position of its defining node.  Parameters are maintained as relative
   values to facilitate recognition of cells no longer requiring dense
   representation.
 */
class DenseCoord {
  unsigned int margin;
  unsigned int implicit; // Nonincreasing.

 public:

  /**
     @brief Applies dense parameters to offsets derived from index node.

     @param startIdx inputs the nodewise starting offset and outputs the
     same value, minus the margin.

     @param extent inputs the nodewise index count and outputs the same
     value, minus the number of implicit indices.

     @return dense count.
   */
  inline unsigned int AdjustDense(unsigned int &startIdx, unsigned int &extent) const {
    startIdx -= margin;
    extent -= implicit;
    return implicit;
  }


  /**
     @brief Sets the dense placement parameters for a cell.

     @return void.
   */
  inline void Init(unsigned int _implicit, unsigned int _margin = 0) {
    implicit = _implicit;
    margin = _margin;
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
  static const unsigned int oneBit = 2;
  static const unsigned int denseBit = 4;

  // Addition bits available for multiple buffers:
  static const unsigned int bufBit = 8;

  unsigned char raw;
 public:

 
  inline void Init() {
    raw = 0;
  }

  
  inline void Init(unsigned int bufIdx, bool singleton) {
    raw = defBit | (singleton ? oneBit : 0) | (bufIdx == 0 ? 0 : bufBit);
  }


  inline bool Singleton() const {
    return (raw & oneBit) != 0;
  }

  inline bool Singleton(unsigned int &bufIdx) const {
    bufIdx = (raw & bufBit) == 0 ? 0 : 1;
    return Singleton();
  }
  

  inline void SetDense() {
    raw |= denseBit;
  }

  
  /**
     @brief Determines whether cell requires dense placement, i.e, is either
     unaligned within a dense region or is itself dense.

     @return true iff dense bit set.
   */
  inline bool Dense() const {
    return (raw & denseBit) != 0;
  }


  inline void SetSingleton() {
    raw |= oneBit;
  }

  
  inline bool Defined() const {
    return (raw & defBit) != 0;
  }
  

  inline bool Undefine() {
    bool wasDefined = Defined();
    raw &= ~defBit;
    return wasDefined;
  }


  /**
     @brief Looks up position parameters and resets definition bit.

     @return void, with output reference parameters.
  */
  inline void Consume(unsigned int &bufIdx, bool &singleton) {
    singleton = Singleton(bufIdx);
    (void) Undefine();
  }
};


/**
   @brief Per-level reaching definitions.
 */
class Level {
  const unsigned int nPred;
  const std::vector<unsigned int> &denseIdx;
  const unsigned int nPredDense;
  const unsigned int splitCount;
  const unsigned int noIndex; // Inattainable node index value.
  const unsigned int idxLive; // Total # sample indices at level.
  const bool nodeRel;  // Subtree- or node-relative indexing.

  unsigned int defCount; // # live definitions.
  unsigned char del; // Position in deque.  Increments.

  // Persistent:
  std::vector<IndexAnc> indexAnc; // Stage coordinates, by node.

  // More elegant and parsimonious to use std::map from pair to node,
  // but hashing much too slow.
  std::vector<MRRA> def; // Indexed by pair-offset.
  std::vector<DenseCoord> denseCoord;

  // Recomputed:
  class IdxPath *relPath;
  std::vector<class NodePath> nodePath; // Indexed by <node, predictor> pair.
  std::vector<unsigned int> liveCount; // Indexed by node.

 public:
  Level(unsigned int _splitCount, unsigned int _nPred, const std::vector<unsigned int> &_denseIdx, unsigned int _nPredDense, unsigned int _noIndex, unsigned int _idxLive, bool _nodeRel);
  ~Level();

  
  void Flush(class Bottom *bottom, bool forward = true);
  void FlushDef(class Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx);
  bool NonreachPurge();
  void Paths();
  void PathInit(const class Bottom *bottom, unsigned int levelIdx, unsigned int path, unsigned int start, unsigned int extent, unsigned int relBase);
  void Bounds(const SPPair &mrra, unsigned int &startIdx, unsigned int &extent);
  void FrontDef(class Bottom *bottom, unsigned int mrraIdx, unsigned int predIdx, unsigned int bufIdx, bool singleton);
  void OffsetClone(const SPPair &mrra, unsigned int reachOffset[], unsigned int reachBase[]);
  unsigned int DiagRestage(const SPPair &mrra, unsigned int reachOffset[]);
  void RunCounts(class Bottom *bottom, const SPPair &mrra, const unsigned int pathCount[], const unsigned int rankCount[]) const;

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
     @brief Dense offsets maintained separately, as a special case.

     @return offset strided by 'nPredDense'.
   */
  inline unsigned int DenseOffset(unsigned int mrraIdx, unsigned int predIdx) const {
    return mrraIdx * nPredDense + denseIdx[predIdx];
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

     @param implicit is only set directly by staging.  Otherwise it has a
     default setting of zero, which is later reset by restaging.
   */
  inline bool Define(unsigned int levelIdx, unsigned predIdx, unsigned int bufIdx, bool singleton, unsigned int implicit = 0) {
    if (levelIdx != noIndex) {
      def[PairOffset(levelIdx, predIdx)].Init(bufIdx, singleton);
      SetDense(levelIdx, predIdx, implicit);
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


  inline void Consume(unsigned int levelIdx, unsigned int predIdx, unsigned int &bufIdx, bool &singleton) {
    def[PairOffset(levelIdx, predIdx)].Consume(bufIdx, singleton);
    defCount--;
  }


  /**
     @brief Determines whether pair consists of a single run.

     @param bufIdx outputs the buffer index.

     @return true iff a singleton.
   */
  inline bool Singleton(unsigned int levelIdx, unsigned int predIdx) {
    return def[PairOffset(levelIdx, predIdx)].Singleton();
  }


  inline bool Singleton(unsigned int levelIdx, unsigned int predIdx, unsigned int &bufIdx) {
    return def[PairOffset(levelIdx, predIdx)].Singleton(bufIdx);
  }


  inline unsigned int AdjustDense(unsigned int levelIdx, unsigned int predIdx, unsigned int &startIdx, unsigned int &extent) const {
    return def[PairOffset(levelIdx, predIdx)].Dense() ?
      denseCoord[DenseOffset(levelIdx, predIdx)].AdjustDense(startIdx, extent) : 0;
  }


  inline void Ref(unsigned int levelIdx, unsigned int predIdx, unsigned int &bufIdx, bool &singleton) {
    singleton = def[PairOffset(levelIdx, predIdx)].Singleton(bufIdx);
  }


  inline bool Defined(unsigned int levelIdx, unsigned int predIdx) const {
    return def[PairOffset(levelIdx, predIdx)].Defined();
  }


  inline bool Dense(unsigned int levelIdx, unsigned int predIdx) const {
    return def[PairOffset(levelIdx, predIdx)].Dense();
  }

  /**
     @brief Sets the density-associated parameters for a reached node.

     @return void.
  */
  inline void SetDense(unsigned int levelIdx, unsigned int predIdx, unsigned int implicit, unsigned int margin = 0) {
    if (implicit > 0 || margin > 0) {
      def[PairOffset(levelIdx, predIdx)].SetDense();
      denseCoord[DenseOffset(levelIdx, predIdx)].Init(implicit, margin);
    }
  }


  /**
     @brief Establishes front-level IndexSet as future ancestor.

     @return void.
  */
  void Ancestor(unsigned int levelIdx, unsigned int start, unsigned int extent) {
    indexAnc[levelIdx].Init(start, extent);
  }


  /**
     @brief Numeric run counts are constrained to be either 1, if singleton,
     or zero otherwise.  Singleton iff dense and all indices implicit or
     not dense and all indices have identical rank.

     @return void.
  */
  inline void SetSingleton(unsigned int levelIdx, unsigned int predIdx) {
    def[PairOffset(levelIdx, predIdx)].SetSingleton();
  }
};


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
  std::vector<unsigned int> termST; // Frontier subtree indices.
  std::vector<class TermKey> termKey; // Frontier map keys:  uninitialized.
  bool nodeRel; // Subtree- or node-relative indexing.  Sticky, once node-.

  static constexpr double efficiency = 0.15; // Work efficiency threshold.

  IdxPath *stPath; // IdxPath accessed by subtree.
  unsigned int splitPrev;
  unsigned int splitCount; // # nodes in the level about to split.
  const class PMTrain *pmTrain;
  class SamplePred *samplePred;
  const class RowRank *rowRank;
  class SplitPred *splitPred;  // constant?
  class SplitSig *splitSig;
  class Run *run;
  class BV *replayExpl; // Whether sample employs explicit replay.
  std::vector<unsigned int> history;
  std::vector<unsigned int> historyPrev;
  std::vector<unsigned char> levelDelta;
  std::vector<unsigned char> deltaPrev;
  Level *levelFront; // Current level.
  std::vector<unsigned int> runCount;
  std::deque<Level *> level;
  
  std::vector<RestageCoord> restageCoord;

  // Restaging methods.
  void Restage(RestageCoord &rsCoord);
  void Restage(const SPPair &mrra, unsigned int bufIdx, unsigned int del, const unsigned int reachBase[], unsigned int reachOffset[]);
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
  void RootDef(unsigned int predIdx, bool singleton, unsigned int implicit);
  void ScheduleRestage(unsigned int del, unsigned int mrraIdx, unsigned int predIdx, unsigned int bufIdx);
  int RestageIdx(unsigned int bottomIdx);
  void RestagePath(unsigned int startIdx, unsigned int extent, unsigned int lhOff, unsigned int rhOff, unsigned int level, unsigned int predIdx);
  bool Preschedule(unsigned int levelIdx, unsigned int predIdx, unsigned int &bufIdx);
  bool ScheduleSplit(unsigned int levelIdx, unsigned int predIdx, unsigned int &rCount) const;

  static Bottom *FactoryReg(const class PMTrain *_pmTrain, const class RowRank *_rowRank, class SamplePred *_samplePred, unsigned int _bagCount);
  static Bottom *FactoryCtg(const class PMTrain *_pmTrain, const class RowRank *_rowRank, class SamplePred *_samplePred, const std::vector<class SampleNode> &_sampleCtg, unsigned int _bagCount);
  
  Bottom(const class PMTrain *_pmTrain, class SamplePred *_samplePred, const class RowRank *_rowRank, class SplitPred *_splitPred, unsigned int _bagCount);
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
  void Restage();
  bool IsFactor(unsigned int predIdx) const;
  unsigned int FacIdx(unsigned int predIdx, bool &isFactor) const;
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


  /**
     @brief Accessor.  SSNode only client.
   */
  inline class Run *Runs() {
    return run;
  }


  inline bool DensePlacement(const SPPair &mrra, unsigned int del = 0) const {
    return level[del]->Dense(mrra.first, mrra.second);
  }


  inline void Bounds(const SPPair &mrra, unsigned int del, unsigned int &startIdx, unsigned int &extent) const {
    level[del]->Bounds(mrra, startIdx, extent);
  }


  void OffsetClone(const SPPair &mrra, unsigned int del, unsigned int reachOffset[], unsigned int reachBase[] = nullptr) {
    level[del]->OffsetClone(mrra, reachOffset, reachBase);
  }

  
  unsigned int SplitCount(unsigned int del) {
    return level[del]->SplitCount();
  }


  /**
     @brief Flips source bit if a definition reaches to current level.

     @return void
   */
  inline void AddDef(unsigned int reachIdx, unsigned int predIdx, unsigned int bufIdx, bool singleton) {
    if (levelFront->Define(reachIdx, predIdx, bufIdx, singleton)) {
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
    return levelFront->AdjustDense(levelIdx, predIdx, startIdx, extent);
  }


  inline IdxPath *FrontPath(unsigned int del) const {
    return level[del]->FrontPath();
  }


  inline unsigned int SplitCount() const {
    return splitCount;
  }


  inline void SetSingleton(unsigned int levelIdx, unsigned int predIdx) const {
    levelFront->SetSingleton(levelIdx, predIdx);
  }
  

  inline void SetRunCount(unsigned int levelIdx, unsigned int predIdx, bool hasImplicit, unsigned int rankCount) {
    bool dummy;
    unsigned int rCount = hasImplicit ? rankCount + 1 : rankCount;
    if (rCount == 1) {
      SetSingleton(levelIdx, predIdx);
    }
    if (IsFactor(predIdx)) {
      runCount[levelIdx * nPredFac + FacIdx(predIdx, dummy)] = rCount;
    }
  }

  
  inline Level *LevelFront() const {
    return levelFront;
  }
};


#endif

