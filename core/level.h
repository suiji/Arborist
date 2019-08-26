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

#ifndef CORE_LEVEL_H
#define CORE_LEVEL_H

#include "splitcoord.h"
#include "typeparam.h"

#include <vector>

/**
   @brief Inherited state for most-recently-restaged ancestor.
 */
class MRRA {
  static constexpr unsigned int defBit = 1;
  static constexpr unsigned int oneBit = 2;
  static constexpr unsigned int denseBit = 4;

  // Additional bits available for multiple buffers:
  static constexpr unsigned int bufBit = 8;

  unsigned char raw;
 public:

  MRRA() : raw(0) {
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

     @param[in, out] idxRange inputs the unadjusted range and outputs the adjusted range.

     @return count of implicit indices, i.e., size of dense blob..
   */
  inline unsigned int adjustRange(IndexRange& idxRange) const {
    idxRange.adjust(margin, implicit);
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
  const vector<unsigned int>& denseIdx; // Compressed mapping to dense offsets.
  const unsigned int nPredDense; // # dense predictors.
  const IndexType nSplit; // # splitable nodes at level.
  const IndexType noIndex; // Inattainable node index value.
  const IndexType idxLive; // Total # sample indices at level.

  unsigned int defCount; // # live definitions.
  unsigned char del; // Position in deque.  Increments.

  // Immutable:
  //
  static unsigned int predFixed;
  static vector<double> predProb;

  // Persistent:
  vector<IndexRange> indexAnc; // Stage coordinates, by node.

  // More elegant and parsimonious to use map from pair to node,
  // but hashing much too slow.
  vector<MRRA> def; // Indexed by pair-offset.
  vector<DenseCoord> denseCoord;

  // Recomputed:
  unique_ptr<class IdxPath> relPath;
  vector<IndexType> offCand;
  vector<class NodePath> nodePath; // Indexed by <node, predictor> pair.
  vector<IndexType> liveCount; // Indexed by node.

  IndexType candExtent; // Total candidate index extent.
  const bool nodeRel;  // Subtree- or node-relative indexing.
  class Bottom *bottom;

/**
   @brief Schedules a non-singleton splitting candidate.

   @param splitIdx

   @param predIdx

   @return true iff pair scheduled for splitting.
 */
  bool preschedule(class SplitFrontier *splitNode,
                   const SplitCoord& splitCoord,
                   unsigned int &spanCand);
  
public:
  Level(unsigned int _nSplit,
        unsigned int _nPred,
        const class RankedFrame* rankedFrame,
        unsigned int _noIndex,
        unsigned int _idxLive,
        bool _nodeRel,
        class Bottom *bottom);
  ~Level();

  static void immutables(unsigned int feFixed, const vector<double> &feProb);
  static void deImmutables();


  /**
     @brief Signals SplitFrontier to schedule splitable pairs.

     @param index summarizes the index sets at the current level.

     @param splitNode maintains the candidate list.
  */
  void candidates(const class Frontier *index,
                  class SplitFrontier *splitNode);

  /**
   @brief Determines splitable candidates by Bernoulli sampling.

   @param splitIdx is the level-relative node index.

   @param ruPred is a vector of uniformly-sampled variates.

   @param offCand accumulates offsets for splitable pairs.
 */
  void candidateProb(class SplitFrontier *splitNode,
                     IndexType splitIdx,
                     const double ruPred[],
                     IndexType &offCand);

  /**
   @brief Determines splitable candidates from fixed number of predictors.

   @param ruPred is a vector of uniformly-sampled variates.

   @param heap orders probability-weighted variates.

   @param extent is the index count of the splitting node.

   @param offCand accumulates offsets for splitable pairs.
 */
  void candidateFixed(class SplitFrontier *splitNode,
                      IndexType splitIdx,
                      const double ruPred[],
                      struct BHPair heap[],
                      IndexType& offCand);


  void rankRestage(class ObsPart *samplePred,
                   const SplitCoord& mrra,
                   Level *levelFront,
                   unsigned int bufIdx);


  void indexRestage(class ObsPart* obsPart,
                    const SplitCoord& mrra,
                    const Level* levelFront,
                    unsigned int bufIdx);

  /**
     @brief Precomputes path vector prior to restaging.

     This is necessary in the case of dense ranks, as cell sizes are not
     derivable directly from index nodes.

     Decomposition into two paths adds ~5% performance penalty, but
     appears necessary for dense packing or for coprocessor loading.
  */
  void rankRestage(class ObsPart *samplePred,
                   const SplitCoord& mrra,
                   Level *levelFront,
                   unsigned int bufIdx,
                   unsigned int reachOffset[], 
                   const unsigned int reachBase[] = nullptr);

  void indexRestage(class ObsPart *samplePred,
                    const SplitCoord& mrra,
                    const Level *levelFront,
                    unsigned int bufIdx,
                    const unsigned int reachBase[],
                    unsigned int reachOffset[],
                    unsigned int splitOffset[]);

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
  void flushDef(const SplitCoord& splitCoord);


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
                const IndexRange& bufRange,
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
  bool scheduleSplit(const SplitCoord& splitCoord,
                     unsigned int &rCount) const;

  /**
     @brief Looks up the ancestor cell built for the corresponding index
     node and adjusts start and extent values by corresponding dense parameters.
  */
  IndexRange getRange(const SplitCoord& mrra);

  void frontDef(const SplitCoord& splitCoord,
                unsigned int bufIdx,
                bool singleton);

  /**
     @brief Clones offsets along path reaching from ancestor node.

     @param mrra is an MRRA coordinate.

     @param[out] reachOffset outputs node starting offsets.

     @param[out] reachBase outputs node-relative offsets, iff nonnull.

     @return path origin at the index passed.
  */
  void offsetClone(const SplitCoord& mrra,
                   unsigned int reachOffset[],
                   unsigned int reachBase[] = nullptr);

  void offsetClone(const SplitCoord& mrra,
                   unsigned int reachOffset[],
                   unsigned int splitOffset[],
                   unsigned int reachBase[]);

/**
   @brief Sets dense count on target MRRA and, if singleton, sets run count to
   unity.
 */
  void setRunCounts(class Bottom *bottom,
                    const SplitCoord& mrra,
                    const unsigned int pathCount[],
                    const unsigned int rankCount[]) const;

/**
   @brief Sets the packed offsets for each successor.  Relies on Swiss Cheese
   index numbering ut prevent cell boundaries from crossing.

   @param idxLeft is the left-most index of the predecessor.

   @param pathCount inputs the counts along each reaching path.

   @param[out] reachOffset outputs the dense starting offsets.
 */
  void packDense(IndexType idxLeft,
                 const unsigned int pathCount[],
                 Level *levelFront,
                 const SplitCoord& mrra,
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
  void setSingleton(const SplitCoord& splitCoord);

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
    return relPath.get();
  }

  
  /**
     @brief Getter for count of live sample indices.
  */
  inline unsigned int IdxLive() {
    return idxLive;
  }


  /**
     @brief Dense offsets maintained separately, as a special case.

     @return offset strided by 'nPredDense'.
   */
  inline unsigned int denseOffset(const SplitCoord& splitCoord) const {
    return splitCoord.nodeIdx * nPredDense + denseIdx[splitCoord.predIdx];
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
  inline bool define(const SplitCoord& splitCoord,
                     unsigned int bufIdx,
                     bool singleton,
                     unsigned int implicit = 0) {
    if (splitCoord.nodeIdx != noIndex) {
      def[splitCoord.strideOffset(nPred)].init(bufIdx, singleton);
      setDense(splitCoord, implicit);
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
  inline void undefine(const SplitCoord& splitCoord) {
    defCount -= def[splitCoord.strideOffset(nPred)].undefine() ? 1 : 0;
  }

  /**
     @brief As above, but assumes live and offers output parameters:
  
     @param[out] bufIdx outputs the buffer index of the definition.

     @param[out] singleton outputs whether the definition is singleton.
   */
  inline void consume(const SplitCoord& splitCoord,
                      unsigned int &bufIdx,
                      bool &singleton) {
    def[splitCoord.strideOffset(nPred)].consume(bufIdx, singleton);
    defCount--;
  }


  /**
     @brief Determines whether pair consists of a single run.

     @param levelIdx is the level-relative split index.

     @param predIdx is the predictor index.

     @return true iff a singleton.
   */
  inline bool isSingleton(const SplitCoord& splitCoord) const {
    return def[splitCoord.strideOffset(nPred)].isSingleton();
  }


  /**
     @brief As above, but with output buffer index parameter.

     @param[out] bufIdx is the buffer index for the cell.
   */
  inline bool isSingleton(const SplitCoord& splitCoord,
                          unsigned int& bufIdx) const {
    return def[splitCoord.strideOffset(nPred)].isSingleton(bufIdx);
  }


  /**
     @brief Adjusts starting index and extent if definition is dense.

     @param[in, out] startIdx is adjust by the dense margin.

     @param[in, out] extent is adjust by the implicit count.
   */
  void adjustRange(const SplitCoord& splitCoord,
                   IndexRange& idxRange) const;


  /**
     @param[out] implicit outputs the number of implicit indices.

     @return adjusted index range.
   */
  IndexRange adjustRange(const SplitCoord& splitCoord,
                         const class Frontier* index,
                         unsigned int& implicit) const;
  

  inline bool isDefined(const SplitCoord& splitCoord) const {
    return def[splitCoord.strideOffset(nPred)].isDefined();
  }


  inline bool isDense(const SplitCoord& splitCoord) const {
    return def[splitCoord.strideOffset(nPred)].isDense();
  }

  /**
     @brief Sets the density-associated parameters for a reached node.

     @return void.
  */
  inline void setDense(const SplitCoord& splitCoord,
                       unsigned int implicit,
                       unsigned int margin = 0) {
    if (implicit > 0 || margin > 0) {
      def[splitCoord.strideOffset(nPred)].setDense();
      denseCoord[denseOffset(splitCoord)].init(implicit, margin);
    }
  }


  /**
     @brief Establishes front-level IndexSet as future ancestor.
  */
  void initAncestor(unsigned int splitIdx,
                    const IndexRange& bufRange) {
    indexAnc[splitIdx].set(bufRange.getStart(), bufRange.getExtent());
  }


  /**
     @brief Sets the number of span candidates.
   */
  void setSpan(unsigned int spanCand) {
    candExtent = spanCand;
  }
};

#endif
