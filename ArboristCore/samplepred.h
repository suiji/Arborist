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
 @brief Contains the sample data used by predictor-specific sample-walking pass.
*/
class SamplePred {
  // SamplePred appear in predictor order, grouped by node.  They store the
  // y-value, run class and sample index for the predictor position to which they
  // correspond.


  // Predictor-based sample orderings, double-buffered by level value.
  //
  const unsigned int bufferSize; // <= nRow * nPred.
  const unsigned int pitchSP; // Pitch of SampleRank vector, in bytes.
  const unsigned int pitchSIdx; // Pitch of SIdx vector, in bytes.

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
  const unsigned int nPred;
  const unsigned int bagCount;
  unsigned int *destRestage; // Coprocessor restaging.
  unsigned int *destSplit; // Coprocessor restaging.

  
 public:
  SamplePred(unsigned int _nPred, unsigned int _bagCount, unsigned int _bufferSize);
  virtual ~SamplePred();


  SampleRank *StageBounds(unsigned int predIdx, unsigned int safeOffset, unsigned int extent, unsigned int *&smpIdx);

  virtual void Stage(const class RRNode *rrNode, unsigned int rrTot, const vector<class SampleNux> &sampleNode, const vector<unsigned int> &row2Sample, vector<class StageCount> &stageCount);
  unsigned int Stage(const vector<class SampleNux> &sampleNode, const class RRNode *rrPred, const vector<unsigned int> &row2Sample, unsigned int explMax, unsigned int predIdx, unsigned int safeOffset, unsigned int extent, bool &singleton);
  void Stage(const vector<class SampleNux> &sampleNode, const class RRNode &rrNode, const vector<unsigned int> &row2Sample, SampleRank *spn, unsigned int *smpIdx, unsigned int &expl);

  double BlockReplay(unsigned int predIdx, unsigned int sourceBit, unsigned int start, unsigned int extent, class BV *replayExpl, vector<class SumCount> &ctgExpl);

  virtual void Restage(class Level *levelBack, class Level *levelFront, const SPPair &mrra, unsigned int bufIdx);
  
  void Prepath(const class IdxPath *idxPath, const unsigned int reachBase[], unsigned int predIdx, unsigned int bufIdx, unsigned int startIdx, unsigned int extent, unsigned int pathMask, bool idxUpdate, unsigned int pathCount[]);
  void Prepath(const class IdxPath *idxPath, const unsigned int reachBase[], bool idxUpdate, unsigned int startIdx, unsigned int extent, unsigned int pathMask, unsigned int idxVec[], PathT prepath[], unsigned int pathCount[]) const;
  void RankRestage(unsigned int predIdx, unsigned int bufIdx, unsigned int start, unsigned int extent, unsigned int reachOffset[], unsigned int rankPrev[], unsigned int rankCount[]);
  void IndexRestage(const class IdxPath *idxPath, const unsigned int reachBase[], unsigned int predIdx, unsigned int bufIdx, unsigned int startIdx, unsigned int extent, unsigned int pathMask, bool idxUpdate, unsigned int reachOffset[], unsigned int splitOffset[]);

  
  inline unsigned int BagCount() const {
    return bagCount;
  }

  
  inline unsigned int PitchSP() {
    return pitchSP;
  }

  inline unsigned int PitchSIdx() {
    return pitchSIdx;
  }


  /**
     @brief Returns the staging position for a dense predictor.
   */
  inline unsigned int StageOffset(unsigned int predIdx) {
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
  inline unsigned int BuffOffset(unsigned int bufferBit) const {
    return (bufferBit & 1) == 0 ? 0 : bufferSize;
  }

  /**

     @param predIdx is the predictor coordinate.

     @param level is the current level.

     @return starting position within workspace.
   */
  inline unsigned int BufferOff(unsigned int predIdx, unsigned int bufBit) const {
    return stageOffset[predIdx] + BuffOffset(bufBit);
  }


  /**
     @return base of the index buffer.
   */
  inline unsigned int *BufferIndex(unsigned int predIdx, unsigned int bufBit) {
    return indexBase + BufferOff(predIdx, bufBit);
  }


  /**
     @return base of node buffer.
   */
  inline SampleRank *BufferNode(unsigned int predIdx, unsigned int bufBit) {
    return nodeVec + BufferOff(predIdx, bufBit);
  }
  
  
  /**
   */
  inline SampleRank* Buffers(unsigned int predIdx, unsigned int bufBit, unsigned int*& sIdx) {
    unsigned int offset = BufferOff(predIdx, bufBit);
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
    return nodeVec + BufferOff(predIdx, bufBit);
  }
  

  /**
     @brief Returns buffer containing splitting information.
   */
  inline SampleRank* SplitBuffer(unsigned int predIdx, unsigned int bufBit) {
    return nodeVec + BufferOff(predIdx, bufBit);
  }


  /**
   @brief Looks up source and target vectors.

   @param predIdx is the predictor column.

   @param level is the upcoming level.

   @return void, with output parameter vectors.
 */
  inline void Buffers(int predIdx, unsigned int bufBit, SampleRank *&source, unsigned int *&sIdxSource, SampleRank *&targ, unsigned int *&sIdxTarg) {
    source = Buffers(predIdx, bufBit, sIdxSource);
    targ = Buffers(predIdx, 1 - bufBit, sIdxTarg);
  }

  // To coprocessor subclass:
  inline void IndexBuffers(unsigned int predIdx, unsigned int bufBit, unsigned int *&sIdxSource, unsigned int *&sIdxTarg) {
    sIdxSource = indexBase + BufferOff(predIdx, bufBit);
    sIdxTarg = indexBase + BufferOff(predIdx, 1 - bufBit);
  }
  

  // TODO:  Move somewhere appropriate.
  /**
     @brief Finds the smallest power-of-two multiple >= 'count'.

     @param count is the size to align.

     @return alignment size.
   */
  static inline unsigned int AlignPow(unsigned int count, unsigned int pow) {
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
  inline bool SingleRank(unsigned int predIdx, unsigned int bufIdx, unsigned int idxStart, unsigned int extent) {
    SampleRank *spNode = BufferNode(predIdx, bufIdx);
    return extent > 0 ? (spNode[idxStart].Rank() == spNode[extent - 1].Rank()) : false;
  }


  /**
     @brief Singleton iff either:
     i) Dense and all indices implicit or ii) Not dense and all ranks equal.

     @param stageCount is the number of staged indices.

     @param predIdx is the predictor index at which to initialize.

     @return true iff entire staged set has single rank.  This might be
     a property of the training data or may arise from bagging. 
  */
  bool Singleton(unsigned int stageCount, unsigned int predIdx) {
    return bagCount == stageCount ? SingleRank(predIdx, 0, 0, bagCount) : (stageCount == 0 ? true : false);
  }
};

#endif
