// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file obspart.h

   @brief Observation frame, partitioned by tree node.

   @author Mark Seligman
 */

#ifndef PARTITION_OBSPART_H
#define PARTITION_OBSPART_H


#include "splitcoord.h"
#include "typeparam.h"

#include <vector>

#include "samplenux.h" // Temporary


/**
 @brief Contains the sample data used by predictor-specific sample-walking pass.
*/
class ObsPart {
  // ObsPart appear in predictor order, grouped by node.  They store the
  // y-value, run class and sample index for the predictor position to which they
  // correspond.


  // Predictor-based sample orderings, double-buffered by level value.
  //
  const IndexT bagCount;
  const IndexT bufferSize; // <= nRow * nPred.

  vector<PathT> pathIdx;
  SampleRank* nodeVec;

  // 'indexBase' could be boxed with SampleRank.  While it is used in both
  // replaying and restaging, though, it plays no role in splitting.  Maintaining
  // a separate vector permits a 16-byte stride to be used for splitting.  More
  // significantly, it reduces memory traffic incurred by transposition on the
  // coprocessor.
  //
  IndexT* indexBase; // RV index for this row.  Used by CTG as well as on replay.

  /**
     @brief Sets branch sense over specified range.

     @param[in, out] enc accumulates sampled response statistics.
   */  
  void branchSet(const class SplitNux* nux,
		 const IndexRange& range,
		 class BranchSense* branchSense,
		 CritEncoding& enc) const;

  /**
     @brief Unsets branch sense over specified range.

     Parameters as above.
   */
  void branchUnset(const class SplitNux* nux,
		   const IndexRange& range,
		   class BranchSense* branchSense,
		   CritEncoding& enc) const;

  
 protected:
  unsigned int *destRestage;
  //unsigned int *destSplit; // Coprocessor restaging.

  
 public:
  vector<IndexRange> stageRange;

  ObsPart(const class Layout* frame, IndexT bagCount_);
  virtual ~ObsPart();

  
  /**
     @brief Localizes copies of the paths to each index position.

     Also localizes index positions themselves, if in a node-relative regime.

     @param reachBase is non-null iff index offsets enter as node relative.

     @param idxUpdate is true iff the index is to be updated.

     @param startIdx is the beginning index of the cell.

     @param extent is the count of indices in the cell.

     @param pathMask mask the relevant bits of the path value.

     @param idxVec inputs the index offsets, relative either to the
     current subtree or the containing node and may output an updated
     value.

     @param[out] prePath outputs the (masked) path reaching the current index.

     @param pathCount enumerates the number of times a path is hit.  Only
     client is currently dense packing.
  */
  void prepath(const class IdxPath *idxPath,
               const unsigned int reachBase[],
               bool idxUpdate,
               const IndexRange& idxRange,
               unsigned int pathMask,
               unsigned int idxVec[],
               PathT prepath[],
               unsigned int pathCount[]) const;

  /**
     @brief Pass-through to Path method.

     Looks up reaching cell in appropriate buffer.
     Parameters as above.
  */
  void prepath(const class IdxPath *idxPath,
               const unsigned int reachBase[],
	       const PreCand& mrra,
               const IndexRange& idxRange,
               unsigned int pathMask,
               bool idxUpdate,
               unsigned int pathCount[]);


  void branchUpdate(const class SplitNux* nux,
		    const vector<IndexRange>& range,
		    class BranchSense* branchSense,
		    CritEncoding& enc) const;


  void branchUpdate(const class SplitNux* nux,
		    const IndexRange& range,
		    class BranchSense* branchSense,
		    CritEncoding& enc) const;


  /**
     @brief Restages and tabulates rank counts.
  */
  void rankRestage(const PreCand& defCoord,
                   const IndexRange& idxRange,
                   unsigned int reachOffset[],
                   unsigned int rankPrev[],
                   unsigned int rankCount[]);

  void indexRestage(const class IdxPath *idxPath,
                    const unsigned int reachBase[],
                    const PreCand& mrra,
                    const IndexRange& idxRange,
                    unsigned int pathMask,
                    bool idxUpdate,
                    unsigned int reachOffset[],
                    unsigned int splitOffset[]);

  
  /**
     @brief Passes through to bufferOff() using definition coordinate.
   */
  IndexT* getBufferIndex(const class SplitNux* nux) const;


  SampleRank* getBuffers(const class SplitNux* nux, IndexT*& sIdx) const;


  SampleRank* getPredBase(const class SplitNux* nux) const;
  
  
  inline IndexT getBagCount() const {
    return bagCount;
  }


  /**
     @brief Returns the staging position for a dense predictor.
   */
  inline auto getStageOffset(PredictorT predIdx) const {
    return stageRange[predIdx].idxStart;
  }


  // The category could, alternatively, be recorded in an object subclassed
  // under class ObsPart.  This would require that the value be restaged,
  // which happens for all predictors at all splits.  It would also require
  // that distinct ObsPart classes be maintained for SampleReg and
  // SampleCtg.  Recomputing the category value on demand, then, seems an
  // easier way to go.
  //

  /**
     @brief Toggles between positions in workspace double buffer, by level.

     @return workspace starting position for this level.
   */
  inline IndexT buffOffset(unsigned int bufferBit) const {
    return (bufferBit & 1) == 0 ? 0 : bufferSize;
  }

  /**

     @param predIdx is the predictor coordinate.

     @param level is the current level.

     @return starting position within workspace.
   */
  inline IndexT bufferOff(PredictorT predIdx, unsigned int bufBit) const {
    return stageRange[predIdx].idxStart + buffOffset(bufBit);
  }


  inline IndexT bufferOff(const PreCand& defCoord,
			  bool comp = false) const {
    return bufferOff(defCoord.splitCoord.predIdx, comp ? defCoord.compBuffer() : defCoord.bufIdx);
  }


  /**
     @return base of the index buffer.
   */
  inline IndexT* bufferIndex(const PreCand& mrra) const {
    return indexBase + bufferOff(mrra);
  }


  /**
     @return base of node buffer.
   */
  inline SampleRank *bufferNode(PredictorT predIdx, unsigned int bufBit) const {
    return nodeVec + bufferOff(predIdx, bufBit);
  }
  
  
  /**
   */
  inline SampleRank* buffers(PredictorT predIdx,
			     unsigned int bufBit,
			     IndexT*& sIdx) const {
    IndexT offset = bufferOff(predIdx, bufBit);
    sIdx = indexBase + offset;
    return nodeVec + offset;
  }


  

  inline IndexT* indexBuffer(const PreCand& defCoord) const {
    return indexBase + bufferOff(defCoord.splitCoord.predIdx, defCoord.bufIdx);
  }


  /**
     @brief Passes through to above after looking up splitting parameters.
   */
  SampleRank* buffers(const PreCand& defCoord,
                      IndexT*& sIdx) const {
    return buffers(defCoord.splitCoord.predIdx, defCoord.bufIdx, sIdx);
  }


  /**
     @brief Allows lightweight lookup of predictor's SampleRank vector.

     @param bufBit is the containing buffer, currently 0/1.
 
     @param predIdx is the predictor index.

     @return node vector section for this predictor.
   */
  SampleRank* getPredBase(const PreCand& defCoord) const {
    return nodeVec + bufferOff(defCoord);
  }
  
  /**
     @brief Returns buffer containing splitting information.
   */
  inline SampleRank* Splitbuffer(PredictorT predIdx, unsigned int bufBit) {
    return nodeVec + bufferOff(predIdx, bufBit);
  }


  inline void buffers(const PreCand& mrra,
		      SampleRank*& source,
		      IndexT*& sIdxSource,
		      SampleRank*& targ,
		      IndexT*& sIdxTarg) {
    source = buffers(mrra.splitCoord.predIdx, mrra.bufIdx, sIdxSource);
    targ = buffers(mrra.splitCoord.predIdx, mrra.compBuffer(), sIdxTarg);
  }

  
  // To coprocessor subclass:
  inline void indexBuffers(const PreCand& mrra,
                           IndexT*& sIdxSource,
                           IndexT*& sIdxTarg) {
    sIdxSource = indexBase + bufferOff(mrra);
    sIdxTarg = indexBase + bufferOff(mrra, true);
  }
  

  // TODO:  Move somewhere appropriate.
  /**
     @brief Finds the smallest power-of-two multiple >= 'count'.

     @param count is the size to align.

     @return alignment size.
   */
  static constexpr unsigned int alignPow(unsigned int count, unsigned int pow) {
    return ((count + (1 << pow) - 1) >> pow) << pow;
  }


  /**
     @param Determines whether the predictors within a nonempty cell
     all have the same rank.

     @param extent is the number of indices subsumed by the cell.

     @return true iff cell consists of a single rank.
   */
  inline bool singleRank(PredictorT predIdx,
                         unsigned int bufIdx,
                         unsigned int idxStart,
                         unsigned int extent) const {
    SampleRank *spNode = bufferNode(predIdx, bufIdx);
    return extent > 0 ? (spNode[idxStart].getRank() == spNode[extent - 1].getRank()) : false;
  }


  /**
     @brief Singleton iff either:
     i) Dense and all indices implicit or ii) Not dense and all ranks equal.

     @param stageCount is the number of staged indices.

     @param predIdx is the predictor index at which to initialize.

     @return true iff entire staged set has single rank.  This might be
     a property of the training data or may arise from bagging. 
  */
  bool singleton(IndexT stageCount,
                 PredictorT predIdx) const {
    return bagCount == stageCount ? singleRank(predIdx, 0, 0, bagCount) : (stageCount == 0 ? true : false);
  }
};

#endif
