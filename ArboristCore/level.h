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

  inline void init(unsigned int _start, unsigned int _extent) {
    start = _start;
    extent = _extent;
  }

  
  /**
     @brief Dual field accessor, specific to sample indexing.
   */
  inline void Ref(unsigned int &start_, unsigned int &extent_) {
    start_ = start;
    extent_ = extent;
  }
};


/**
   @brief Inherited state for most-recently-restaged ancestor.
 */
class MRRA {
  static constexpr unsigned int defBit = 1;
  static constexpr unsigned int oneBit = 2;
  static constexpr unsigned int denseBit = 4;

  // Addition bits available for multiple buffers:
  static constexpr unsigned int bufBit = 8;

  unsigned char raw;
 public:

 
  inline void init() {
    raw = 0;
  }


  /**
     @brief Initializes as live and sets descriptor values.

     @param bufIdx is the buffer in which the definition resides.

     @param singleton is true iff the value is singleton.
   */
  inline void init(unsigned int bufIdx, bool singleton) {
    raw = defBit | (singleton ? oneBit : 0) | (bufIdx == 0 ? 0 : bufBit);
  }


  /**
     @brief Getter for singleton bit.

     @return true iff value is singleton.
   */
  inline bool isSingleton() const {
    return (raw & oneBit) != 0;
  }


  /**
     @brief Determines both buffer index and singleton state.

     @param[out] bufIdx outputs the resident buffer index.

     @return true iff singleton.
   */
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


  /**
     @brief Sets the singleton bit.
   */
  inline void setSingleton() {
    raw |= oneBit;
  }


  /**
     @brief Indicates whether value is live.
   */
  inline bool isDefined() const {
    return (raw & defBit) != 0;
  }
  

  /**
     @brief Marks value as extinct.
     
     @return true iff the value was live on entry.
   */
  inline bool undefine() {
    bool wasDefined = isDefined();
    raw &= ~defBit;
    return wasDefined;
  }


  /**
     @brief Looks up position parameters and resets definition bit.

     @param[out] bufIdx outputs the buffer index containing the definition.

     @param[out] singleton outputs whether the value is singleton.
  */
  inline void consume(unsigned int &bufIdx, bool &singleton) {
    singleton = isSingleton(bufIdx);
    (void) undefine();
  }
};


/**
   @brief Defines the parameters needed to place a dense cell with respect
   the position of its defining node.

   Parameters are maintained as relative values to facilitate recognition
   of cells no longer requiring dense representation.
 */
class DenseCoord {
  unsigned int margin; // # unused slots in cell.
  unsigned int implicit; // Nonincreasing value.

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
  inline void init(unsigned int implicit,
                   unsigned int margin = 0) {
    this->implicit = implicit;
    this->margin = margin;
  }
};


/**
   @brief Per-level reaching definitions.
 */
class Level {
  const unsigned int nPred; // Predictor count.
  const vector<unsigned int> &denseIdx; // Compressed mapping to dense offsets.
  const unsigned int nPredDense; // # dense predictors.
  const unsigned int nSplit; // # splitable nodes at level.
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

  unsigned int candExtent; // Total candidate index extent.
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
  Level(unsigned int _nSplit,
        unsigned int _nPred,
        const vector<unsigned int> &_denseIdx,
        unsigned int _nPredDense,
        unsigned int _noIndex,
        unsigned int _idxLive,
        bool _nodeRel,
        class Bottom *bottom);
  ~Level();

  static void immutables(unsigned int feFixed, const vector<double> &feProb);
  static void deImmutables();


  /**
     @brief Signals SplitNode to schedule splitable pairs.

     @param index summarizes the index sets at the current level.

     @param splitNode maintains the candidate list.
  */
  void candidates(const class IndexLevel *index,
                  class SplitNode *splitNode);

  /**
   @brief Determines splitable candidates by Bernoulli sampling.

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
   @brief Determines splitable candidates from fixed number of predictors.

   @param ruPred is a vector of uniformly-sampled variates.

   @param heap orders probability-weighted variates.

   @param extent is the index count of the splitting node.

   @param offCand accumulates offsets for splitable pairs.
 */
  void candidateFixed(class SplitNode *splitNode,
                      unsigned int splitIdx,
                      const double ruPred[],
                      struct BHPair heap[],
                      unsigned int extent,
                      unsigned int &offCand);

  void rankRestage(class SamplePred *samplePred,
                   const SPPair &mrra,
                   Level *levelFront,
                   unsigned int bufIdx);
  void indexRestage(class SamplePred *samplePred, const SPPair &mrra, const Level *levelFront, unsigned int bufIdx);

  /**
     @brief Precomputes path vector prior to restaging.

     This is necessary in the case of dense ranks, as cell sizes are not
     derivable directly from index nodes.

     Decomposition into two paths adds ~5% performance penalty, but
     appears necessary for dense packing or for coprocessor loading.
  */
  void rankRestage(class SamplePred *samplePred,
                   const SPPair &mrra,
                   Level *levelFront,
                   unsigned int bufIdx,
                   unsigned int reachOffset[], 
                   const unsigned int reachBase[] = nullptr);

  void indexRestage(class SamplePred *samplePred, const SPPair &mrra, const Level *levelFront, unsigned int bufIdx, const unsigned int reachBase[], unsigned int reachOffset[], unsigned int splitOffset[]);

  /**
     @brief Moves entire level's defnitions to restaging schedule.

     @param bottom is the active bottom state.

     @return void.
  */
  void flush(bool forward = true);

  /**
     @brief Removes definition from a back level and builds definition
     for each descendant reached in current level.

     @param mrra is the coordinate pair of the ancestor to flush.
  */
  void flushDef(unsigned int mrraIdx, unsigned int predIdx);


  /**
     @brief Walks the definitions, purging those which no longer reach.

     @return true iff a definition was purged at this level.
  */
  bool nonreachPurge();

  /**
     @brief Initializes paths reaching from non-front levels.
   */
  void reachingPaths();

  void pathInit(const class Bottom *bottom,
                unsigned int levelIdx,
                unsigned int path,
                unsigned int start,
                unsigned int extent,
                unsigned int relBase);

  /**
     @brief Determines whether a cell is suitable for splitting.

     It may, for example, have become unsplitiable as a result of restaging's
     precipitating a singleton instance.

     @param splitIdx is the split index.

     @param predIdx is the predictor index.

     @param[out] rCount outputs the run count iff not singleton.

     @return true iff candidate remains splitable.
  */
  bool scheduleSplit(unsigned int splitIdx,
                     unsigned int predIdx,
                     unsigned int &rCount) const;

  /**
     @brief Looks up the ancestor cell built for the corresponding index
     node and adjusts start and extent values by corresponding dense parameters.
  */
  void getBounds(const SPPair &mrra,
              unsigned int &startIdx,
              unsigned int &extent);

  void frontDef(unsigned int mrraIdx,
                unsigned int predIdx,
                unsigned int bufIdx,
                bool singleton);

  /**
     @brief Clones offsets along path reaching from ancestor node.

     @param mrra is an MRRA coordinate.

     @param[out] reachOffset outputs node starting offsets.

     @param[out] reachBase outputs node-relative offsets, iff nonnull.

     @return path origin at the index passed.
  */
  void offsetClone(const SPPair &mrra,
                   unsigned int reachOffset[],
                   unsigned int reachBase[] = nullptr);

  void offsetClone(const SPPair &mrra,
                   unsigned int reachOffset[],
                   unsigned int splitOffset[],
                   unsigned int reachBase[]);

/**
   @brief Sets dense count on target MRRA and, if singleton, sets run count to
   unity.
 */
  void setRunCounts(class Bottom *bottom,
                    const SPPair &mrra,
                    const unsigned int pathCount[],
                    const unsigned int rankCount[]) const;

/**
   @brief Sets the packed offsets for each successor.  Relies on Swiss Cheese
   index numbering ut prevent cell boundaries from crossing.

   @param idxLeft is the left-most index of the predecessor.

   @param pathCount inputs the counts along each reaching path.

   @param[out] reachOffset outputs the dense starting offsets.
 */
  void packDense(unsigned int idxLeft,
                 const unsigned int pathCount[],
                 Level *levelFront,
                 const SPPair &mrra,
                 unsigned int reachOffset[]) const;

  void setExtinct(unsigned int idx);

  /**
     @brief Revises node-relative indices, as appropriae.  Irregular,
     but data locality improves with tree depth.

     @param one2Front maps first level to front indices.

     @return true iff level employs node-relative indexing.
  */
  bool backdate(const class IdxPath *one2Front);

  /**
     @brief Sets the definition's heritable singleton bit and clears the
     current level's splitable bit.
  */
  void setSingleton(unsigned int splitIdx,
                    unsigned int predIdx);

  /**
     @brief Sets path, target and node-relative offse.
  */
  void setLive(unsigned int idx,
               unsigned int path,
               unsigned int targIdx,
               unsigned int ndBase);


  /**
     @brief Getter for level delta.
   */
  inline unsigned int getDel() const {
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
  const inline class IdxPath *getFrontPath() const {
    return relPath;
  }

  
  /**
     @brief Getter for count of live sample indices.
  */
  inline unsigned int IdxLive() {
    return idxLive;
  }


  /**
     @brief Obtains absolute offset of split/predictor pair.

     @param mrraIdx is the index of a split w.r.t its level.

     @param predIdx is a predictor index.

     @return offset strided by 'nPred'.
   */
  inline size_t pairOffset(unsigned int mrraIdx, unsigned int predIdx) const {
    return mrraIdx * nPred + predIdx;
  }


  /**
     @brief Dense offsets maintained separately, as a special case.

     @return offset strided by 'nPredDense'.
   */
  inline unsigned int denseOffset(unsigned int mrraIdx, unsigned int predIdx) const {
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
  inline unsigned int pathMask() const {
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
  inline bool define(unsigned int levelIdx,
                     unsigned predIdx,
                     unsigned int bufIdx,
                     bool singleton,
                     unsigned int implicit = 0) {
    if (levelIdx != noIndex) {
      def[pairOffset(levelIdx, predIdx)].init(bufIdx, singleton);
      setDense(levelIdx, predIdx, implicit);
      defCount++;
      return true;
    }
    else {
      return false;
    }
  }


  /**
     @brief Marks definition at given coordinate as extinct.

     @param splitIdx is the split index.

     @param predIdx is the predictor index.
  */
  inline void undefine(unsigned int levelIdx,
                       unsigned int predIdx) {
    defCount -= def[pairOffset(levelIdx, predIdx)].undefine() ? 1 : 0;
  }

  /**
     @brief As above, but assumes live and offers output parameters:
  
     @param[out] bufIdx outputs the buffer index of the definition.

     @param[out] singleton outputs whether the definition is singleton.
   */
  inline void consume(unsigned int levelIdx,
                      unsigned int predIdx,
                      unsigned int &bufIdx,
                      bool &singleton) {
    def[pairOffset(levelIdx, predIdx)].consume(bufIdx, singleton);
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
    return def[pairOffset(levelIdx, predIdx)].isSingleton();
  }


  /**
     @brief As above, but with output buffer index parameter.

     @param[out] bufIdx is the buffer index for the cell.
   */
  inline bool isSingleton(unsigned int levelIdx,
                          unsigned int predIdx,
                          unsigned int &bufIdx) const {
    return def[pairOffset(levelIdx, predIdx)].isSingleton(bufIdx);
  }


  inline unsigned int adjustDense(unsigned int levelIdx,
                                  unsigned int predIdx,
                                  unsigned int &startIdx,
                                  unsigned int &extent) const {
    return def[pairOffset(levelIdx, predIdx)].isDense() ?
      denseCoord[denseOffset(levelIdx, predIdx)].adjustDense(startIdx, extent) : 0;
  }


  /**
     @brief Multiple accessor.
   */
  inline void Ref(unsigned int levelIdx,
                  unsigned int predIdx,
                  unsigned int &bufIdx,
                  bool &singleton) {
    singleton = def[pairOffset(levelIdx, predIdx)].isSingleton(bufIdx);
  }


  inline bool isDefined(unsigned int levelIdx,
                        unsigned int predIdx) const {
    return def[pairOffset(levelIdx, predIdx)].isDefined();
  }


  inline bool isDense(unsigned int levelIdx,
                      unsigned int predIdx) const {
    return def[pairOffset(levelIdx, predIdx)].isDense();
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
      def[pairOffset(levelIdx, predIdx)].setDense();
      denseCoord[denseOffset(levelIdx, predIdx)].init(implicit, margin);
    }
  }


  /**
     @brief Establishes front-level IndexSet as future ancestor.
  */
  void initAncestor(unsigned int splitIdx,
                unsigned int start,
                unsigned int extent) {
    indexAnc[splitIdx].init(start, extent);
  }


  /**
     @brief Sets the number of span candidates.
   */
  void setSpan(unsigned int spanCand) {
    candExtent = spanCand;
  }
};

#endif
