// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplepred.h

   @brief Class definitions supporting maintenance of per-predictor sample orderings.

   @author Mark Seligman

 */


#ifndef ARBORIST_SAMPLEPRED_H
#define ARBORIST_SAMPLEPRED_H

#include "typeparam.h"

#include <vector>

#include "samplenux.h" // Temporary


/**
   @brief Summarizes staging operation.
 */
class StageCount {
 public:
  unsigned int expl;
  bool singleton;
};


/**
 @brief Contains the sample data used by predictor-specific sample-walking pass.
*/
class SamplePred {
  const unsigned int nPred;
  // SamplePred appear in predictor order, grouped by node.  They store the
  // y-value, run class and sample index for the predictor position to which they
  // correspond.


  // Predictor-based sample orderings, double-buffered by level value.
  //
  const unsigned int bagCount;
  const unsigned int bufferSize; // <= nRow * nPred.

  vector<PathT> pathIdx;
  vector<unsigned int> stageOffset;
  vector<unsigned int> stageExtent; // Client:  debugging only.
  SampleRank* nodeVec;

  // 'indexBase' could be boxed with SampleRank.  While it is used in both
  // replaying and restaging, though, it plays no role in splitting.  Maintaining
  // a separate vector permits a 16-byte stride to be used for splitting.  More
  // significantly, it reduces memory traffic incurred by transposition on the
  // coprocessor.
  //
  unsigned int *indexBase; // RV index for this row.  Used by CTG as well as on replay.

 protected:
  unsigned int *destRestage; // Coprocessor restaging.
  unsigned int *destSplit; // Coprocessor restaging.

  
 public:
  SamplePred(unsigned int _nPred, unsigned int _bagCount, unsigned int _bufferSize);
  virtual ~SamplePred();


  void setStageBounds(const class RowRank* rowRank,
                   unsigned int predIdx);

  vector<StageCount> stage(const class RowRank* rowRank,
                           const vector<SampleNux> &sampleNode,
                           const class Sample* sample);

  void stage(const class RowRank* rowRank,
             const vector<SampleNux> &sampleNode,
             const class Sample* sample,
             unsigned int predIdx,
             StageCount& stageCount);

  void stage(const class RowRank* rowRank,
             unsigned int rrTot,
             const vector<class SampleNux> &sampleNode,
             const class Sample* sample,
             vector<class StageCount> &stageCount);

  void stage(const vector<class SampleNux> &sampleNode,
             const class RRNode &rrNode,
             const class Sample *sample,
             unsigned int &expl,
             SampleRank spn[],
             unsigned int smpIdx[]) const;

  /**
     @brief Replays explicitly-reference samples associated with candidate.

     @param cand is a splitting node.

     @param[out] replayExpl sets bits associated with explicit side.

     @param[out] ctgExpl stores explicit response sum and sample count by category.
   */
  double blockReplay(const class SplitCand& cand,
                     class BV* replayExpl,
                     vector<class SumCount>& ctgExpl);


  /**
   @brief Looks up SampleRank block and dispatches appropriate replay method.

   @param blockStart is the starting SampleRank index for the split.

   @param blockExtent is the number of explicit such indices subsumed.

   @param replayExpl sets bits corresponding to explicit indices defined
   by the split.  Indices are either node- or subtree-relative, depending
   on Bottom's current indexing mode.

   @param ctgExpl summarizes explicit sum and sample count by category.

   @return sum of explicit responses within the block.
*/
  double blockReplay(const class SplitCand& cand,
                     unsigned int blockStart,
                     unsigned int blockExtent,
                     class BV *replayExpl,
                     vector<class SumCount> &ctgExpl);


  /**
     @brief Replays a block of categorical sample ranks.

     @param start is the beginning SampleRank index in the block.

     @param extent is the number of indices in the block.

     @param idx[] maps SampleRank indices to sample indices.

     @param[out] replayExpl bits set high at each sample index in block.

     @return sum of explict responses.
   */
  double replayCtg(const SampleRank spn[],
                   unsigned int start,
                   unsigned int extent,
                   const unsigned int idx[],
                   class BV* replayExpl,
                   vector<class SumCount>& ctgExpl);

  /**
     @brief Replays a block of numerical sample ranks.

     Parameters and return as above.
   */
  double replayNum(const SampleRank spn[],
                   unsigned int start,
                   unsigned int extent,
                   const unsigned int idx[],
                   class BV* replayExpl);

  /**
     @brief Drives restaging from an ancestor node and level to current level.

     @param levelBack is the ancestor's level.

     @param levelFront is the current level.

     @param mrra is the ancestor.

     @param bufIdx is the buffer indes of the ancestor.
   */
  void restage(class Level *levelBack,
               class Level *levelFront,
               const SPPair &mrra,
               unsigned int bufIdx);
  
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
               unsigned int startIdx,
               unsigned int extent,
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
               unsigned int predIdx,
               unsigned int bufIdx,
               unsigned int startIdx,
               unsigned int extent,
               unsigned int pathMask,
               bool idxUpdate,
               unsigned int pathCount[]);

  void rankRestage(unsigned int predIdx,
                   unsigned int bufIdx,
                   unsigned int start,
                   unsigned int extent,
                   unsigned int reachOffset[],
                   unsigned int rankPrev[],
                   unsigned int rankCount[]);

  void indexRestage(const class IdxPath *idxPath,
                    const unsigned int reachBase[],
                    unsigned int predIdx,
                    unsigned int bufIdx,
                    unsigned int startIdx,
                    unsigned int extent,
                    unsigned int pathMask,
                    bool idxUpdate,
                    unsigned int reachOffset[],
                    unsigned int splitOffset[]);


  inline unsigned int getBagCount() const {
    return bagCount;
  }


  /**
     @brief Returns the staging position for a dense predictor.
   */
  inline unsigned int getStageOffset(unsigned int predIdx) {
    return stageOffset[predIdx];
  }


  // The category could, alternatively, be recorded in an object subclassed
  // under class SamplePred.  This would require that the value be restaged,
  // which happens for all predictors at all splits.  It would also require
  // that distinct SamplePred classes be maintained for SampleReg and
  // SampleCtg.  Recomputing the category value on demand, then, seems an
  // easier way to go.
  //

  /**
     @brief Toggles between positions in workspace double buffer, by level.

     @return workspace starting position for this level.
   */
  inline unsigned int buffOffset(unsigned int bufferBit) const {
    return (bufferBit & 1) == 0 ? 0 : bufferSize;
  }

  /**

     @param predIdx is the predictor coordinate.

     @param level is the current level.

     @return starting position within workspace.
   */
  inline unsigned int bufferOff(unsigned int predIdx, unsigned int bufBit) const {
    return stageOffset[predIdx] + buffOffset(bufBit);
  }


  /**
     @return base of the index buffer.
   */
  inline unsigned int *bufferIndex(unsigned int predIdx, unsigned int bufBit) const {
    return indexBase + bufferOff(predIdx, bufBit);
  }


  /**
     @return base of node buffer.
   */
  inline SampleRank *bufferNode(unsigned int predIdx, unsigned int bufBit) const {
    return nodeVec + bufferOff(predIdx, bufBit);
  }
  
  
  /**
   */
  inline SampleRank* buffers(unsigned int predIdx, unsigned int bufBit, unsigned int*& sIdx) const {
    unsigned int offset = bufferOff(predIdx, bufBit);
    sIdx = indexBase + offset;
    return nodeVec + offset;
  }


  /**
     @brief Allows lightweight lookup of predictor's SampleRank vector.

     @param bufBit is the containing buffer, currently 0/1.
 
     @param predIdx is the predictor index.

     @return node vector section for this predictor.
   */
  SampleRank* PredBase(unsigned int predIdx, unsigned int bufBit) const {
    return nodeVec + bufferOff(predIdx, bufBit);
  }
  

  /**
     @brief Returns buffer containing splitting information.
   */
  inline SampleRank* Splitbuffer(unsigned int predIdx, unsigned int bufBit) {
    return nodeVec + bufferOff(predIdx, bufBit);
  }


  /**
   @brief Looks up source and target vectors.

   @param predIdx is the predictor column.

   @param level is the upcoming level.

   @return void, with output parameter vectors.
 */
  inline void buffers(int predIdx,
                      unsigned int bufBit,
                      SampleRank *&source,
                      unsigned int *&sIdxSource,
                      SampleRank *&targ,
                      unsigned int *&sIdxTarg) {
    source = buffers(predIdx, bufBit, sIdxSource);
    targ = buffers(predIdx, 1 - bufBit, sIdxTarg);
  }

  // To coprocessor subclass:
  inline void indexBuffers(unsigned int predIdx,
                           unsigned int bufBit,
                           unsigned int *&sIdxSource,
                           unsigned int *&sIdxTarg) {
    sIdxSource = indexBase + bufferOff(predIdx, bufBit);
    sIdxTarg = indexBase + bufferOff(predIdx, 1 - bufBit);
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
     @brief Accessor for staging extent field.
   */
  inline unsigned int StageExtent(unsigned int predIdx) {
    return stageExtent[predIdx];
  }

  
  /**
     @param Determines whether the predictors within a nonempty cell
     all have the same rank.

     @param extent is the number of indices subsumed by the cell.

     @return true iff cell consists of a single rank.
   */
  inline bool singleRank(unsigned int predIdx,
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
  bool singleton(unsigned int stageCount,
                 unsigned int predIdx) const {
    return bagCount == stageCount ? singleRank(predIdx, 0, 0, bagCount) : (stageCount == 0 ? true : false);
  }
};

#endif
