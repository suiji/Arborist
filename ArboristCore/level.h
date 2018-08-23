// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file level.h

   @brief Definitions for the classes managing a single tree level.

   @author Mark Seligman

 */

#ifndef ARBORIST_LEVEL_H
#define ARBORIST_LEVEL_H

#include <vector>
#include "typeparam.h"


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


  inline bool isSingleton() const {
    return (raw & oneBit) != 0;
  }

  inline bool isSingleton(unsigned int &bufIdx) const {
    bufIdx = (raw & bufBit) == 0 ? 0 : 1;
    return isSingleton();
  }
  

  inline void setDense() {
    raw |= denseBit;
  }

  
  /**
     @brief Determines whether cell requires dense placement, i.e, is either
     unaligned within a dense region or is itself dense.

     @return true iff dense bit set.
   */
  inline bool isDense() const {
    return (raw & denseBit) != 0;
  }


  inline void setSingleton() {
    raw |= oneBit;
  }

  
  inline bool isDefined() const {
    return (raw & defBit) != 0;
  }
  

  inline bool undefine() {
    bool wasDefined = isDefined();
    raw &= ~defBit;
    return wasDefined;
  }


  /**
     @brief Looks up position parameters and resets definition bit.

     @return void, with output reference parameters.
  */
  inline void Consume(unsigned int &bufIdx, bool &singleton) {
    singleton = isSingleton(bufIdx);
    (void) undefine();
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
     @brief Compresses index node coordinates for dense access.

     @param[in, out] startIdx inputs the nodewise starting offset which is
     then decremented by the margin.

     @param[in, out] extent inputs the nodewise index count, which is then
     decremented by the implicit count.

     @return count of implicit indices, i.e., size of dense blob..
   */
  inline unsigned int adjustDense(unsigned int &startIdx,
                                  unsigned int &extent) const {
    startIdx -= margin;
    extent -= implicit;
    return implicit;
  }


  /**
     @brief Sets the dense placement parameters for a cell.

     @return void.
   */
  inline void Init(unsigned int implicit,
                   unsigned int margin = 0) {
    this->implicit = implicit;
    this->margin = margin;
  }
};


/**
   @brief Per-level reaching definitions.
 */
class Level {
  const unsigned int nPred;
  const vector<unsigned int> &denseIdx;
  const unsigned int nPredDense;
  const unsigned int nSplit;
  const unsigned int noIndex; // Inattainable node index value.
  const unsigned int idxLive; // Total # sample indices at level.

  unsigned int defCount; // # live definitions.
  unsigned char del; // Position in deque.  Increments.

  // Immutable:
  //
  static unsigned int predFixed;
  static vector<double> predProb;

  // Persistent:
  vector<IndexAnc> indexAnc; // Stage coordinates, by node.

  // More elegant and parsimonious to use map from pair to node,
  // but hashing much too slow.
  vector<MRRA> def; // Indexed by pair-offset.
  vector<DenseCoord> denseCoord;

  // Recomputed:
  class IdxPath *relPath;
  vector<unsigned int> offCand;
  vector<class NodePath> nodePath; // Indexed by <node, predictor> pair.
  vector<unsigned int> liveCount; // Indexed by node.

  unsigned int spanCand; // Total candidate span.
  const bool nodeRel;  // Subtree- or node-relative indexing.
  class Bottom *bottom;

/**
   @brief Schedules a non-singleton splitting candidate.

   @param splitIdx

   @param predIdx

   @return true iff pair scheduled for splitting.
 */
  bool preschedule(class SplitNode *splitNode,
                   unsigned int levelIdx,
                   unsigned int predIdx,
                   unsigned int extent,
                   unsigned int &spanCand);
  
 public:
  Level(unsigned int _nSplit, unsigned int _nPred, const vector<unsigned int> &_denseIdx, unsigned int _nPredDense, unsigned int _noIndex, unsigned int _idxLive, bool _nodeRel, class Bottom *bottom);
  ~Level();

  static void Immutables(unsigned int feFixed, const vector<double> &feProb);
  static void DeImmutables();


  /**
     @brief Signals SplitNode to schedule splitable pairs.

     @param index

     @param splitNode maintains the candidate list.

     @return void.
  */
  void candidates(const class IndexLevel *index,
                  class SplitNode *splitNode);

  /**
   @brief Set splitable flag by Bernoulli sampling.

   @param splitIdx is the level-relative node index.

   @param ruPred is a vector of uniformly-sampled variates.

   @param offCand accumulates offsets for splitable pairs.
 */
  void candidateProb(class SplitNode *splitNode,
                     unsigned int splitIdx,
                     const double ruPred[],
                     unsigned int extent,
                     unsigned int &offCand);

  /**
   @brief Sets splitable flag for a fixed number of predictors.

   @param ruPred is a vector of uniformly-sampled variates.

   @param heap orders probability-weighted variates.

   @param extent is the index count of the splitting node.

   @param offCand accumulates offsets for splitable pairs.
 */
  void candidateFixed(class SplitNode *splitNode,
                      unsigned int splitIdx,
                      const double ruPred[],
                      class BHPair heap[],
                      unsigned int extent,
                      unsigned int &offCand);

  void rankRestage(class SamplePred *samplePred,
                   const SPPair &mrra,
                   Level *levelFront,
                   unsigned int bufIdx);
  void indexRestage(class SamplePred *samplePred, const SPPair &mrra, const Level *levelFront, unsigned int bufIdx);

  void rankRestage(class SamplePred *samplePred,
                   const SPPair &mrra,
                   Level *levelFront,
                   unsigned int bufIdx,
                   unsigned int reachOffset[], 
                   const unsigned int reachBase[] = nullptr);

  void indexRestage(class SamplePred *samplePred, const SPPair &mrra, const Level *levelFront, unsigned int bufIdx, const unsigned int reachBase[], unsigned int reachOffset[], unsigned int splitOffset[]);

  void flush(bool forward = true);
  void flushDef(unsigned int mrraIdx, unsigned int predIdx);
  bool NonreachPurge();
  void Paths();
  void PathInit(const class Bottom *bottom, unsigned int levelIdx, unsigned int path, unsigned int start, unsigned int extent, unsigned int relBase);

  /**
     @brief Determines whether a cell is suitable for splitting.  It may,
     for example, have become unsplitiable as a result of restaging's
     precipitating a singleton instance.

     @param levelIdx is the split index.

     @param predIdx is the predictor index.

     @param runCount outputs the run count iff not singleton.

     @return true iff candidate remains splitable.
  */
  bool scheduleSplit(unsigned int levelIdx, unsigned int predIdx, unsigned int &rCount) const;

  void getBounds(const SPPair &mrra,
              unsigned int &startIdx,
              unsigned int &extent);

  void FrontDef(unsigned int mrraIdx,
                unsigned int predIdx,
                unsigned int bufIdx,
                bool singleton);

  void offsetClone(const SPPair &mrra,
                   unsigned int reachOffset[],
                   unsigned int reachBase[] = nullptr);

  void offsetClone(const SPPair &mrra,
                   unsigned int reachOffset[],
                   unsigned int splitOffset[],
                   unsigned int reachBase[]);

  void setRunCounts(class Bottom *bottom,
                    const SPPair &mrra,
                    const unsigned int pathCount[],
                    const unsigned int rankCount[]) const;

  void packDense(unsigned int idxLeft,
                 const unsigned int pathCount[],
                 Level *levelFront,
                 const SPPair &mrra,
                 unsigned int reachOffset[]) const;

  void setExtinct(unsigned int idx);

  bool backdate(const class IdxPath *one2Front);
  void setSingleton(unsigned int levelIdx, unsigned int predIdx);
  bool Splitable(unsigned int levelIdx, unsigned int predIdx, unsigned int &bufIdx);
  void setLive(unsigned int idx, unsigned int path, unsigned int targIdx, unsigned int ndBase);


  /**
   */
  inline unsigned int Del() const {
    return del;
  }

  /**
     @brief Accessor for indexing mode.  Currently two-valued.
   */
  inline bool isNodeRel() const {
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
  inline unsigned int backScale(unsigned int val) const {
    return val << (unsigned int) del;
  }


  /**
     @brief Produces mask approprate for level:  lowest 'del' bits high.

     @return bit mask value.
   */
  inline unsigned int PathMask() const {
    return backScale(1) - 1;
  }
  

  /**
     @brief Accessor.  What more can be said?

     @return definition count at this level.
  */
  inline unsigned int getDefCount() {
    return defCount;
  }


  inline unsigned int getSplitCount() {
    return nSplit;
  }


  /**
     @brief

     @param implicit is only set directly by staging.  Otherwise it has a
     default setting of zero, which is later reset by restaging.
   */
  inline bool Define(unsigned int levelIdx,
                     unsigned predIdx,
                     unsigned int bufIdx,
                     bool singleton,
                     unsigned int implicit = 0) {
    if (levelIdx != noIndex) {
      def[PairOffset(levelIdx, predIdx)].Init(bufIdx, singleton);
      setDense(levelIdx, predIdx, implicit);
      defCount++;
      return true;
    }
    else {
      return false;
    }
  }


  inline void undefine(unsigned int levelIdx,
                       unsigned int predIdx) {
    bool wasDefined = def[PairOffset(levelIdx, predIdx)].undefine();
    defCount -= wasDefined ? 1 : 0;
  }


  inline void Consume(unsigned int levelIdx,
                      unsigned int predIdx,
                      unsigned int &bufIdx,
                      bool &singleton) {
    def[PairOffset(levelIdx, predIdx)].Consume(bufIdx, singleton);
    defCount--;
  }


  /**
     @brief Determines whether pair consists of a single run.

     @param levelIdx is the level-relative split index.

     @param predIdx is the predictor index.

     @return true iff a singleton.
   */
  inline bool isSingleton(unsigned int levelIdx,
                          unsigned int predIdx) const {
    return def[PairOffset(levelIdx, predIdx)].isSingleton();
  }


  /**
     @brief As above, but with output buffer index parameter.

     @param[out] bufIdx is the buffer index for the cell.
   */
  inline bool isSingleton(unsigned int levelIdx,
                          unsigned int predIdx,
                          unsigned int &bufIdx) const {
    return def[PairOffset(levelIdx, predIdx)].isSingleton(bufIdx);
  }


  inline unsigned int adjustDense(unsigned int levelIdx,
                                  unsigned int predIdx,
                                  unsigned int &startIdx,
                                  unsigned int &extent) const {
    return def[PairOffset(levelIdx, predIdx)].isDense() ?
      denseCoord[DenseOffset(levelIdx, predIdx)].adjustDense(startIdx, extent) : 0;
  }


  /**
     @brief Multiple accessor.
   */
  inline void Ref(unsigned int levelIdx,
                  unsigned int predIdx,
                  unsigned int &bufIdx,
                  bool &singleton) {
    singleton = def[PairOffset(levelIdx, predIdx)].isSingleton(bufIdx);
  }


  inline bool isDefined(unsigned int levelIdx,
                        unsigned int predIdx) const {
    return def[PairOffset(levelIdx, predIdx)].isDefined();
  }


  inline bool isDense(unsigned int levelIdx,
                      unsigned int predIdx) const {
    return def[PairOffset(levelIdx, predIdx)].isDense();
  }

  /**
     @brief Sets the density-associated parameters for a reached node.

     @return void.
  */
  inline void setDense(unsigned int levelIdx,
                       unsigned int predIdx,
                       unsigned int implicit,
                       unsigned int margin = 0) {
    if (implicit > 0 || margin > 0) {
      def[PairOffset(levelIdx, predIdx)].setDense();
      denseCoord[DenseOffset(levelIdx, predIdx)].Init(implicit, margin);
    }
  }


  /**
     @brief Establishes front-level IndexSet as future ancestor.

     @return void.
  */
  void Ancestor(unsigned int levelIdx,
                unsigned int start,
                unsigned int extent) {
    indexAnc[levelIdx].Init(start, extent);
  }


  /**
     @brief Sets the number of span candidates.
   */
  void setSpan(unsigned int spanCand) {
    this->spanCand = spanCand;
  }
};

#endif
