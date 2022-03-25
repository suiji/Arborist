// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file deffrontier.h

   @brief Tracks repartition definitions associated with a single frontier instance.

   Definitions cache the repartition state of a given splitting cell.
   Some algorithms, such as Random Forests, employ variable selection
   and do not require repartitioning of all cells at each frontier
   instance.  This allows repartitioning to be performed lazily and
   sparingly.

   @author Mark Seligman

 */

#ifndef FRONTIER_DEFRONTIER_H
#define FRONTIER_DEFRONTIER_H

#include "mrra.h"
#include "typeparam.h"

#include <vector>

/**
   @brief Minimal liveness information for most-recently-restaged ancestor.
 */
class LiveBits {
  static constexpr unsigned int defBit = 1;
  static constexpr unsigned int singletonBit = 2;
  static constexpr unsigned int denseBit = 4;

  // Additional bits available for multiple buffers:
  static constexpr unsigned int bufBit = 8;

  unsigned char raw; // Encodes liveness, denseness and whether singleton.
 public:

  LiveBits() : raw(0) {
  }

  
  /**
     @brief Initializes as live and sets descriptor values.

     @param bufIdx is the buffer in which the definition resides.

     @param singleton is true iff the value is singleton.
   */
  inline void init(unsigned int bufIdx, bool singleton) {
    raw = defBit | (singleton ? singletonBit : 0) | (bufIdx == 0 ? 0 : bufBit);
  }


  /**
     @brief Singleton indicator.

     @return true iff value is singleton.
   */
  inline bool isSingleton() const {
    return (raw & singletonBit) != 0;
  }


  /**
     @brief Singleton and buffer indicator.

     @param[out] bufIdx outputs the resident buffer index.

     @return true iff singleton.
   */
  inline bool isSingleton(unsigned int& bufIdx) const {
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
  inline void setSingleton(bool isSingleton) {
    raw |= isSingleton ? singletonBit : 0;
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

     @param[out] singleton outputs whether the value is singleton.
  */
  inline MRRA consume(const SplitCoord& splitCoord,
			 unsigned int del,
			 bool& singleton) {
    unsigned int bufIdx;
    singleton = isSingleton(bufIdx);
    (void) undefine();
    return MRRA(splitCoord, bufIdx, del);
  }


  void setSingleton(const class StageCount& stageCount);
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

  inline IndexT getImplicit() const {
    return implicit;
  }

  
  /**
     @brief Compresses index node coordinates for dense access.

     @param[in, out] idxRange inputs the unadjusted range and outputs the adjusted range.
   */
  inline void adjustRange(IndexRange& idxRange) const {
    idxRange.adjust(margin, implicit);
  }


  /**
     @brief Sets the dense placement parameters for a cell.
   */
  inline void init(IndexT implicit,
		   IndexT margin = 0) {
    this->implicit = implicit;
    this->margin = margin;
  }
};


/**
   @brief Caches previous frontier definitiions by layer.
 */
class DefFrontier {
  class DefMap* defMap;
  const PredictorT nPred; // Predictor count.
  const IndexT nSplit; // # splitable nodes at level.
  const IndexT noIndex; // Inattainable node index value.

  IndexT defCount; // # live definitions.
  unsigned char del; // Position in deque.  Increments.

  // Persistent:
  vector<IndexRange> rangeAnc; // Stage coordinates, by node.

  // More elegant and parsimonious to use map from pair to node,
  // but hashing much too slow.
  vector<LiveBits> mrra; // Indexed by pair-offset.
  vector<DenseCoord> denseCoord;

  // Recomputed:
  unique_ptr<class IdxPath> relPath;
  vector<class NodePath> nodePath; // Indexed by <node, predictor> pair.
  vector<IndexT> liveCount; // Indexed by node.

  IndexT candExtent; // Total candidate index extent.
  const bool nodeRel;  // Subtree- or node-relative indexing.

public:
  DefFrontier(IndexT nSplit_,
        PredictorT nPred_,
        IndexT noIndex_,
        IndexT idxLive_,
        bool nodeRel_,
	   class DefMap* defMap);

  
  void rankRestage(class ObsPart *samplePred,
                   const MRRA& mrra,
                   DefFrontier *levelFront);

  /**
     @brief Precomputes path vector prior to restaging.

     This is necessary in the case of dense ranks, as cell sizes are not
     derivable directly from index nodes.

     Decomposition into two paths adds ~5% performance penalty, but
     appears necessary for dense packing or for coprocessor loading.
  */
  void rankRestage(class ObsPart *samplePred,
                   const MRRA& mrra,
                   DefFrontier *levelFront,
                   IndexT reachOffset[], 
                   const IndexT reachBase[] = nullptr);

  /**
     @brief Moves entire level's defnitions to restaging schedule.

     @param defMap is the active defMap state.
  */
  void flush(class DefMap* defMap = nullptr);


  /**
     @brief Walks the definitions, purging those which no longer reach.

     @return true iff a definition was purged at this level.
  */
  bool nonreachPurge();

  /**
     @brief Initializes paths reaching from non-front levels.
   */
  void reachingPaths();


  /**
     @param idxStart is node starting position in upcoming level.
   */
  void pathInit(IndexT levelIdx,
		PathT path,
		const IndexRange& bufRange,
		IndexT idxStart);

  /**
     @param[in, out] cand may have modified run position and index range.

     @param[out] implicit outputs the number of implicit indices.
  */
  void adjustRange(const MRRA& cand,
		   IndexRange& idxRange) const;


  /**
     @brief Looks up the ancestor cell built for the corresponding index
     node and adjusts start and extent values by corresponding dense parameters.
  */
  IndexRange getRange(const MRRA& mrra) const {
    IndexRange idxRange = rangeAnc[mrra.splitCoord.nodeIdx];
    adjustRange(mrra, idxRange);

    return idxRange;
  }


  /**
     @brief Precipitates a top-level precandidate from a definition.

     @param splitCoord is a split coordinate.

     @param[in, out] restageCand collects precandidates for restaging.
   */
  void flushDef(const SplitCoord& splitCoord,
		class DefMap* defMap);


  /**
     @brief Clones offsets along path reaching from ancestor node.

     @param mrra is an MRRA coordinate.

     @param[out] reachOffset outputs node starting offsets.

     @param[out] reachBase outputs node-relative offsets, iff nonnull.
  */
  void offsetClone(const SplitCoord& mrra,
	      IndexT reachOffset[],
	      IndexT reachBase[] = nullptr);
/**
   @brief Sets stage counts on successor cells.
 */
  void setStageCounts(const class MRRA& preCand,
		      const IndexT pathCount[],
		      const IndexT rankCount[]) const;

/**
   @brief Sets the packed offsets for each successor.  Relies on Swiss Cheese
   index numbering ut prevent cell boundaries from crossing.

   @param pathCount inputs the counts along each reaching path.

   @param[out] reachOffset outputs the dense starting offsets.
 */
  void packDense(const IndexT pathCount[],
		 DefFrontier *levelFront,
		 const MRRA& mrra,
		 IndexT reachOffset[]) const;

  /**
     @brief Marks the node-relative index as extinct.
   */
  void relExtinct(IndexT idx);

  /**
     @brief Revises node-relative indices, as appropriae.  Irregular,
     but data locality improves with tree depth.

     @param one2Front maps first level to front indices.

     @return true iff level employs node-relative indexing.
  */
  bool backdate(const class IdxPath *one2Front);

  
  /**
     @brief Sets the definition's heritable singleton bit according to StageCount.
  */
  void setStageCount(const SplitCoord& splitCoord,
		     const class StageCount& stageCount);

  /**
     @brief Sets path, target and node-relative offse.
  */
  void relLive(IndexT idx,
	       PathT path,
	       IndexT targIdx,
	       IndexT nodeBase);


  /**
     @param[in, out] threshold below which not to flush:  decremented.

     @retun true iff flush occurs.
   */
  bool flush(class DefMap* defMap,
	     IndexT& thresh) {
    if (defCount <= thresh) {
      flush(defMap);
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
  inline auto getDel() const {
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
  const inline class IdxPath* getFrontPath() const {
    return relPath.get();
  }

  
  /**
     @brief Shifts a value by the number of back-levels to compensate for
     effects of binary branching.

     @param val is the value to shift.

     @return shifted value.
   */  
  inline IndexT backScale(IndexT idx) const {
    return idx << (unsigned int) del;
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
  inline IndexT getDefCount() {
    return defCount;
  }


  inline IndexT getSplitCount() {
    return nSplit;
  }


  /**
     @brief Creates the root definition for a predictor following staging.

     @param predIdx is the predictor index.

     @param stageCount enumerates the staging sample and rank counts.
   */
  void rootDefine(PredictorT predIdx,
		  const struct StageCount& stageCount);


  /**
     @brief As above, but for not-root case:  general split coordinate.

     The implicit count is only set directly in the root case.  Otherwise it has an
     initial setting of zero, which is later updated by restaging.
   */
  inline bool define(const MRRA& defCoord,
		     bool singleton) {
    if (defCoord.splitCoord.nodeIdx != noIndex) {
      mrra[defCoord.splitCoord.strideOffset(nPred)].init(defCoord.bufIdx, singleton);
      setDense(defCoord.splitCoord, 0); // Initial implicit count of zero, later updated.
      defCount++;
      return true;
    }
    else { // Dummy case.
      return false;
    }
  }


  /**
     @brief Marks definition at given coordinate as extinct.

     @param splitIdx is the split index.

     @param predIdx is the predictor index.
  */
  inline void undefine(const SplitCoord& splitCoord) {
    defCount -= mrra[splitCoord.strideOffset(nPred)].undefine() ? 1 : 0;
  }

  /**
     @brief As above, but assumes live and offers output parameters:
  
     @param[out] bufIdx outputs the buffer index of the definition.

     @param[out] singleton outputs whether the definition is singleton.
   */
  inline MRRA consume(const SplitCoord& splitCoord,
			  bool& singleton) {
    defCount--;
    return mrra[splitCoord.strideOffset(nPred)].consume(splitCoord, del, singleton);
  }


  /**
     @brief Determines whether pair consists of a single run.

     @param levelIdx is the level-relative split index.

     @param predIdx is the predictor index.

     @return true iff a singleton.
   */
  inline bool isSingleton(const SplitCoord& splitCoord) const {
    return mrra[splitCoord.strideOffset(nPred)].isSingleton();
  }


  /**
     @brief As above, but with output buffer index parameter.

     @return true iff non-singleton precandidate appendable.
   */
  inline bool isSingleton(const SplitCoord& splitCoord,
			  unsigned int& bufIdx) const {
    return mrra[splitCoord.strideOffset(nPred)].isSingleton(bufIdx);
  }


  IndexT getImplicit(const MRRA& cand) const;

  inline bool isDefined(const SplitCoord& splitCoord) const {
    return mrra[splitCoord.strideOffset(nPred)].isDefined();
  }


  inline bool isDense(const SplitCoord& splitCoord) const {
    return mrra[splitCoord.strideOffset(nPred)].isDense();
  }

  
  bool isDense(const MRRA& cand) const {
    return isDense(cand.splitCoord);
  }


  void setDense(const SplitCoord& splitCoord,
		IndexT implicit,
		IndexT margin = 0);


  /**
     @brief Establishes front-level IndexSet as future ancestor.
  */
  void initAncestor(IndexT splitIdx,
	       const IndexRange& bufRange) {
    rangeAnc[splitIdx] = IndexRange(bufRange.getStart(), bufRange.getExtent());
  }


  /**
     @brief Sets the number of span candidates.
   */
  void setSpan(IndexT spanCand) {
    candExtent = spanCand;
  }
};

#endif
