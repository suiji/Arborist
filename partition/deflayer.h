// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file deflayer.h

   @brief Definitions for the classes managing a single definition layer.

   @author Mark Seligman

 */

#ifndef PARTITION_DEFLAYER_H
#define PARTITION_DEFLAYER_H

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
  inline void
  init(unsigned int bufIdx, bool singleton) {
    raw = defBit | (singleton ? oneBit : 0) | (bufIdx == 0 ? 0 : bufBit);
  }


  /**
     @brief Getter for singleton bit.

     @return true iff value is singleton.
   */
  inline bool
  isSingleton() const {
    return (raw & oneBit) != 0;
  }


  /**
     @brief Determines both buffer index and singleton state.

     @param[out] bufIdx outputs the resident buffer index.

     @return true iff singleton.
   */
  inline bool
  isSingleton(unsigned int& bufIdx) const {
    bufIdx = (raw & bufBit) == 0 ? 0 : 1;
    return isSingleton();
  }
  

  inline void
  setDense() {
    raw |= denseBit;
  }

  
  /**
     @brief Determines whether cell requires dense placement, i.e, is either
     unaligned within a dense region or is itself dense.

     @return true iff dense bit set.
   */
  inline bool
  isDense() const {
    return (raw & denseBit) != 0;
  }


  /**
     @brief Sets the singleton bit.
   */
  inline void
  setSingleton() {
    raw |= oneBit;
  }


  /**
     @brief Indicates whether value is live.
   */
  inline bool
  isDefined() const {
    return (raw & defBit) != 0;
  }
  

  /**
     @brief Looks up position parameters and resets definition bit.

     @param[out] singleton outputs whether the value is singleton.
  */
  inline DefCoord
  consume(const SplitCoord& splitCoord,
	  unsigned int del,
	  bool& singleton) {
    unsigned int bufIdx;
    singleton = isSingleton(bufIdx);
    (void) undefine();
    return DefCoord(splitCoord, bufIdx, del);
  }


  /**
     @brief Marks value as extinct.
     
     @return true iff the value was live on entry.
   */
  inline bool
  undefine() {
    bool wasDefined = isDefined();
    raw &= ~defBit;
    return wasDefined;
  }
};


/**
   @brief Defines the parameters needed to place a dense cell with respect
   the position of its defining node.

   Parameters are maintained as relative values to facilitate recognition
   of cells no longer requiring dense representation.
 */
class DenseCoord {
  IndexT margin; // # unused slots in cell.
  IndexT implicit; // Nonincreasing value.

 public:

  inline IndexT
  getImplicit() const {
    return implicit;
  }

  
  /**
     @brief Compresses index node coordinates for dense access.

     @param[in, out] idxRange inputs the unadjusted range and outputs the adjusted range.

     @return count of implicit indices, i.e., size of dense blob..
   */
  inline void
  adjustRange(IndexRange& idxRange) const {
    idxRange.adjust(margin, implicit);
  }


  /**
     @brief Sets the dense placement parameters for a cell.

     @return void.
   */
  inline void
  init(IndexT implicit,
       IndexT margin = 0) {
    this->implicit = implicit;
    this->margin = margin;
  }
};


/**
   @brief Per-level reaching definitions.
 */
class DefLayer {
  const PredictorT nPred; // Predictor count.
  const vector<IndexT>& denseIdx; // Compressed mapping to dense offsets.
  const PredictorT nPredDense; // # dense predictors.
  const IndexT nSplit; // # splitable nodes at level.
  const IndexT noIndex; // Inattainable node index value.
  const IndexT idxLive; // Total # sample indices at level.

  IndexT defCount; // # live definitions.
  unsigned char del; // Position in deque.  Increments.

  // Persistent:
  vector<IndexRange> indexAnc; // Stage coordinates, by node.

  // More elegant and parsimonious to use map from pair to node,
  // but hashing much too slow.
  vector<MRRA> def; // Indexed by pair-offset.
  vector<DenseCoord> denseCoord;

  // Recomputed:
  unique_ptr<class IdxPath> relPath;
  vector<class NodePath> nodePath; // Indexed by <node, predictor> pair.
  vector<IndexT> liveCount; // Indexed by node.

  IndexT candExtent; // Total candidate index extent.
  const bool nodeRel;  // Subtree- or node-relative indexing.
  class DefMap *defMap;

  /**
     @brief Schedules a non-singleton splitting candidate.

     @param splitCoord is the pair.

     @return 1 if pair scheduled else 0.
  */
  unsigned int
  preschedule(class SplitFrontier* splitNode,
	      const SplitCoord& splitCoord,
	      IndexT& spanCand);
  
public:
  DefLayer(IndexT nSplit_,
        PredictorT nPred_,
        const class RankedFrame* rankedFrame,
        IndexT noIndex_,
        IndexT idxLive_,
        bool nodeRel_,
        class DefMap* defMap);
  ~DefLayer();


  void rankRestage(class ObsPart *samplePred,
                   const DefCoord& mrra,
                   DefLayer *levelFront);


  void indexRestage(class ObsPart* obsPart,
                    const DefCoord& mrra,
                    const DefLayer* levelFront,
		    const vector<IndexT>& offCand);

  /**
     @brief Precomputes path vector prior to restaging.

     This is necessary in the case of dense ranks, as cell sizes are not
     derivable directly from index nodes.

     Decomposition into two paths adds ~5% performance penalty, but
     appears necessary for dense packing or for coprocessor loading.
  */
  void rankRestage(class ObsPart *samplePred,
                   const DefCoord& mrra,
                   DefLayer *levelFront,
                   unsigned int reachOffset[], 
                   const unsigned int reachBase[] = nullptr);

  void indexRestage(class ObsPart *samplePred,
                    const DefCoord& mrra,
                    const DefLayer *levelFront,
                    const unsigned int reachBase[],
                    unsigned int reachOffset[],
                    unsigned int splitOffset[]);

  /**
     @brief Moves entire level's defnitions to restaging schedule.

     @param defMap is the active defMap state.
  */
  void flush(class SplitFrontier* splitFrontier = nullptr);

  /**
     @brief Removes definition from a back level and builds definition
     for each descendant reached in current level.

     @param mrra is the coordinate pair of the ancestor to flush.
  */
  void
  flushDef(class SplitFrontier* splitFrontier,
	   const SplitCoord& splitCoord);


  /**
     @brief Walks the definitions, purging those which no longer reach.

     @return true iff a definition was purged at this level.
  */
  bool
  nonreachPurge();

  /**
     @brief Initializes paths reaching from non-front levels.
   */
  void
  reachingPaths();

  void
  pathInit(IndexT levelIdx,
	   unsigned int path,
	   const IndexRange& bufRange,
	   IndexT relBase);

  
  /**
     @brief Looks up the ancestor cell built for the corresponding index
     node and adjusts start and extent values by corresponding dense parameters.
  */
  IndexRange
  getRange(const DefCoord& mrra) const;

  void
  frontDef(const DefCoord& defCoord,
	   bool singleton);

  /**
     @brief Clones offsets along path reaching from ancestor node.

     @param mrra is an MRRA coordinate.

     @param[out] reachOffset outputs node starting offsets.

     @param[out] reachBase outputs node-relative offsets, iff nonnull.
  */
  void
  offsetClone(const SplitCoord& mrra,
	      IndexT reachOffset[],
	      IndexT reachBase[] = nullptr);

  void
  offsetClone(const SplitCoord& mrra,
	      const vector<IndexT>& offCand,
	      IndexT reachOffset[],
	      IndexT splitOffset[],
	      IndexT reachBase[] = nullptr);

/**
   @brief Sets dense count on target MRRA and, if singleton, sets run count to
   unity.
 */
  void
  setRunCounts(const SplitCoord& mrra,
	       const unsigned int pathCount[],
	       const unsigned int rankCount[]) const;

/**
   @brief Sets the packed offsets for each successor.  Relies on Swiss Cheese
   index numbering ut prevent cell boundaries from crossing.

   @param idxLeft is the left-most index of the predecessor.

   @param pathCount inputs the counts along each reaching path.

   @param[out] reachOffset outputs the dense starting offsets.
 */
  void
  packDense(IndexT idxLeft,
	    const unsigned int pathCount[],
	    DefLayer *levelFront,
	    const DefCoord& mrra,
	    unsigned int reachOffset[]) const;

  void
  setExtinct(IndexT idx);

  /**
     @brief Revises node-relative indices, as appropriae.  Irregular,
     but data locality improves with tree depth.

     @param one2Front maps first level to front indices.

     @return true iff level employs node-relative indexing.
  */
  bool
  backdate(const class IdxPath *one2Front);

  /**
     @brief Sets the definition's heritable singleton bit and clears the
     current level's splitable bit.
  */
  void
  setSingleton(const SplitCoord& splitCoord);

  /**
     @brief Sets path, target and node-relative offse.
  */
  void
  setLive(IndexT idx,
	  unsigned int path,
	  unsigned int targIdx,
	  unsigned int ndBase);


  IndexT
  denseOffset(const DefCoord& cand) const;


  /**
     @param[in, out] threshold below which not to flush:  decremented.

     @retun true iff flush occurs.
   */
  bool
  flush(class SplitFrontier* splitFrontier,
	IndexT& thresh) {
    if (defCount <= thresh) {
      flush(splitFrontier);
      thresh -= defCount;
      return true;
    }
    else {
      return false;
    }
  }

  
  /**
     @brief Getter for level delta.
   */
  inline unsigned int
  getDel() const {
    return del;
  }

  /**
     @brief Accessor for indexing mode.  Currently two-valued.
   */
  inline bool
  isNodeRel() const {
    return nodeRel;
  }

  
  /**
     @brief Front path accessor.

     @return reference to front path.
   */
  const inline class IdxPath*
  getFrontPath() const {
    return relPath.get();
  }

  
  /**
     @brief Getter for count of live sample indices.
  */
  inline IndexT
  IdxLive() {
    return idxLive;
  }


  /**
     @brief Dense offsets maintained separately, as a special case.

     @return offset strided by 'nPredDense'.
   */
  inline IndexT
  denseOffset(const SplitCoord& splitCoord) const {
    return splitCoord.nodeIdx * nPredDense + denseIdx[splitCoord.predIdx];
  }


  /**
     @brief Shifts a value by the number of back-levels to compensate for
     effects of binary branching.

     @param val is the value to shift.

     @return shifted value.
   */  
  inline unsigned int
  backScale(unsigned int val) const {
    return val << (unsigned int) del;
  }


  /**
     @brief Produces mask approprate for level:  lowest 'del' bits high.

     @return bit mask value.
   */
  inline unsigned int
  pathMask() const {
    return backScale(1) - 1;
  }
  

  /**
     @brief Accessor.  What more can be said?

     @return definition count at this level.
  */
  inline IndexT
  getDefCount() {
    return defCount;
  }


  inline IndexT getSplitCount() {
    return nSplit;
  }


  /**
     @brief

     @param implicit is only set directly by staging.  Otherwise it has a
     default setting of zero, which is later reset by restaging.
   */
  inline bool
  define(const DefCoord& defCoord,
	 bool singleton,
	 IndexT implicit = 0) {
    if (defCoord.splitCoord.nodeIdx != noIndex) {
      def[defCoord.splitCoord.strideOffset(nPred)].init(defCoord.bufIdx, singleton);
      setDense(defCoord.splitCoord, implicit);
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
  inline void
  undefine(const SplitCoord& splitCoord) {
    defCount -= def[splitCoord.strideOffset(nPred)].undefine() ? 1 : 0;
  }

  /**
     @brief As above, but assumes live and offers output parameters:
  
     @param[out] bufIdx outputs the buffer index of the definition.

     @param[out] singleton outputs whether the definition is singleton.
   */
  inline DefCoord
  consume(const SplitCoord& splitCoord,
	  bool& singleton) {
    defCount--;
    return def[splitCoord.strideOffset(nPred)].consume(splitCoord, del, singleton);
  }


  /**
     @brief Determines whether pair consists of a single run.

     @param levelIdx is the level-relative split index.

     @param predIdx is the predictor index.

     @return true iff a singleton.
   */
  inline bool
  isSingleton(const SplitCoord& splitCoord) const {
    return def[splitCoord.strideOffset(nPred)].isSingleton();
  }


  /**
     @brief As above, but with output buffer index parameter.

     @param[out] bufIdx is the buffer index for the cell.
   */
  inline bool
  isSingleton(const SplitCoord& splitCoord,
	      DefCoord& defCoord) const {
    unsigned int bufIdx;
    if (def[splitCoord.strideOffset(nPred)].isSingleton(bufIdx)) {
      return true;
    }
    else {
      defCoord = DefCoord(splitCoord, bufIdx);
      return false;
    }
  }


  /**
     @brief Adjusts starting index and extent if definition is dense.

     @param[in, out] startIdx is adjust by the dense margin.

     @param[in, out] extent is adjust by the implicit count.
   */
  void
  adjustRange(const SplitCoord& splitCoord,
	      IndexRange& idxRange) const;


  /**
     @param[in, out] cand may have modified run position and index range.

     @param[out] implicit outputs the number of implicit indices.

     @return adjusted index range.
   */
  IndexRange
  adjustRange(const DefCoord& cand,
	      const class SplitFrontier* splitFrontier) const;


  IndexT
  getImplicit(const DefCoord& cand) const;
  

  inline bool
  isDefined(const SplitCoord& splitCoord) const {
    return def[splitCoord.strideOffset(nPred)].isDefined();
  }


  inline bool
  isDense(const SplitCoord& splitCoord) const {
    return def[splitCoord.strideOffset(nPred)].isDense();
  }

  bool
  isDense(const DefCoord& cand) const {
    return isDense(cand.splitCoord);
  }



  /**
     @brief Sets the density-associated parameters for a reached node.

     @return void.
  */
  inline void
  setDense(const SplitCoord& splitCoord,
	   IndexT implicit,
	   IndexT margin = 0) {
    if (implicit > 0 || margin > 0) {
      def[splitCoord.strideOffset(nPred)].setDense();
      denseCoord[denseOffset(splitCoord)].init(implicit, margin);
    }
  }


  /**
     @brief Establishes front-level IndexSet as future ancestor.
  */
  void
  initAncestor(IndexT splitIdx,
	       const IndexRange& bufRange) {
    indexAnc[splitIdx] = IndexRange(bufRange.getStart(), bufRange.getExtent());
  }


  /**
     @brief Sets the number of span candidates.
   */
  void
  setSpan(IndexT spanCand) {
    candExtent = spanCand;
  }
};

#endif
