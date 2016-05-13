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

#include "param.h"

#include <vector>

/**
   @brief Container for staging initialization, viz. minimizing communication
   from host or head node.
 */
class StagePack {
  unsigned int sIdx;
  unsigned int rank;
  unsigned int sCount;
  unsigned int ctg;
  FltVal ySum;
 public:
  inline void Ref(unsigned int &_sIdx, unsigned int &_rank, unsigned int &_sCount, unsigned int &_ctg, FltVal &_ySum) const {
    _sIdx = sIdx;
    _rank = rank;
    _sCount = sCount;
    _ctg = ctg;
    _ySum = ySum;
  }

  inline void Set(unsigned int _sIdx, unsigned int _rank, unsigned int _sCount, unsigned int _ctg, FltVal _ySum) {
    sIdx = _sIdx;
    rank = _rank;
    sCount = _sCount;
    ctg = _ctg;
    ySum = _ySum;
  }
};


/**
 */
class SPNode {
  static unsigned int runShift; // Pack:  nonzero iff categorical response.
 protected:
  FltVal ySum; // sum of response values associated with sample.
  unsigned int rank; // True rank, with ties identically receiving lowest applicable value.
  unsigned int sCount; // # occurrences of row sampled.  << # rows.
 public:
  static void Immutables(unsigned int ctgWidth);
  static void DeImmutables();
  unsigned int Init(const StagePack &stagePack);

  // These methods should only be called when the response is known
  // to be regression, as it relies on a packed representation specific
  // to that case.
  //

  /**
     @brief Reports SamplePred contents for regression response.  Cannot
     be used with categorical response, as 'sCount' value reported here
     is unshifted.

     @param _ySum outputs the response value.

     @param _rank outputs the predictor rank.

     @param _sCount outputs the multiplicity of the row in this sample.

     @return void.
   */
  inline void RegFields(FltVal &_ySum, unsigned int &_rank, unsigned int &_sCount) const {
    _ySum = ySum;
    _rank = rank;
    _sCount = sCount;
  }

  // These methods should only be called when the response is known
  // to be categorical, as it relies on a packed representation specific
  // to that case.
  //
  
  /**
     @brief Reports SamplePred contents for categorical response.  Can
     be called with regression response if '_yCtg' value ignored.

     @param _ySum outputs the proxy response value.

     @param _rank outputs the predictor rank.

     @param _yCtg outputs the response value.

     @return sample count, with output reference parameters.
   */
  inline unsigned int CtgFields(FltVal &_ySum, unsigned int &_rank, unsigned int &_yCtg) const {
    _ySum = ySum;
    _rank = rank;
    _yCtg = sCount & ((1 << runShift) - 1);

    return sCount >> runShift;
  }


    /**
     @brief Variant of above, for which rank determined separately.

     @param _ySum outputs the proxy response value.

     @param _sCount outputs the multiplicity of the row in this sample.

     @param _yCtg outputs the response value.

     @return sample count of node, with output reference parameters.
   */
  inline unsigned int CtgFields(FltVal &_ySum, unsigned int &_yCtg) const {
    _ySum = ySum;
    _yCtg = sCount & ((1 << runShift) - 1);

    return sCount >> runShift;
  }

  
  /**
   @brief Determines whether the consecutive index positions are a run of predictor values.

   @param start is starting index position of potential run.

   @param end is the ending index position of potential run.

   @return whether a run is encountered.
  */
  inline bool IsRun(int start, int end) const {
    return (this + start)->rank == (this + end)->rank;
  }


  /**
     @brief Accessor for 'rank' field

     @return rank value.
   */
  inline unsigned int Rank() const {
    return rank;
  }


  /**
     @brief Accessor for 'ySum' field

     @return sum of y-values for sample.
   */
  inline FltVal YSum() {
    return ySum;
  }

};


/**
 @brief Contains the sample data used by predictor-specific sample-walking pass.
*/
class SamplePred {
  // SamplePred appear in predictor order, grouped by node.  They store the
  // y-value, run class and sample index for the predictor position to which they
  // correspond.

  const unsigned int bagCount;
  const unsigned int nPred;

  // Predictor-based sample orderings, double-buffered by level value.
  //
  const int bufferSize; // bagCount * nPred.
  const unsigned int pitchSP; // Pitch of SPNode vector, in bytes.
  const unsigned int pitchSIdx; // Pitch of SIdx vector, in bytes.

  SPNode* nodeVec;

  // 'sampleIdx' could be boxed with SPNode.  While it is used in both
  // replaying and restaging, though, it plays no role in splitting.  Maintaining
  // a separate vector permits a 16-byte stride to be used for splitting.  More
  // significantly, it reduces memory traffic incurred by transposition on the
  // coprocessor.
  //
  unsigned int *sampleIdx; // RV index for this row.  Used by CTG as well as on replay.
 public:
  SamplePred(unsigned int _nPred, unsigned int _bagCount);
  ~SamplePred();
  static SamplePred *Factory(unsigned int _nPred, unsigned int _bagCount);

  void Stage(const std::vector<StagePack> &stagePack, unsigned int predIdx);
 
  inline unsigned int PitchSP() {
    return pitchSP;
  }

  inline unsigned int PitchSIdx() {
    return pitchSIdx;
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
  inline int BuffOffset(unsigned int bufferBit) const {
    return (bufferBit & 1) == 0 ? 0 : bufferSize;
  }

  /**

     @param predIdx is the predictor coordinate.

     @param level is the current level.

     @return starting position within workspace.
   */
  inline unsigned int BufferOff(unsigned int predIdx, unsigned int bufBit) const {
    return bagCount * predIdx + BuffOffset(bufBit);
  }

  
  /**
   */
  inline SPNode* Buffers(unsigned int predIdx, unsigned int bufBit, unsigned int*& sIdx) {
    unsigned int offset = BufferOff(predIdx, bufBit);
    sIdx = sampleIdx + offset;
    return nodeVec + offset;
  }


  /**
     @brief Allows lightweight lookup of predictor's SPNode vector.

     @param bufBit is the containing buffer, currently 0/1.
 
     @param predIdx is the predictor index.

     @return node vector section for this predictor.
   */
  SPNode* PredBase(unsigned int predIdx, unsigned int bufBit) const {
    return nodeVec + BufferOff(predIdx, bufBit);
  }
  

  /**
     @brief Returns buffer containing splitting information.
   */
  inline SPNode* SplitBuffer(unsigned int predIdx, unsigned int bufBit) {
    return nodeVec + BufferOff(predIdx, bufBit);
  }


  /**
   @brief Looks up source and target vectors.

   @param predIdx is the predictor column.

   @param level is the upcoming level.

   @return void, with output parameter vectors.
 */
  inline void Buffers(int predIdx, unsigned int bufBit, SPNode *&source, unsigned int *&sIdxSource, SPNode *&targ, unsigned int *&sIdxTarg) {
    source = Buffers(predIdx, bufBit, sIdxSource);
    targ = Buffers(predIdx, 1 - bufBit, sIdxTarg);
  }

  void SplitRanks(unsigned int predIdx, unsigned int targBit, int spIdx, unsigned int &rkLow, unsigned int &rkHigh);
  double Replay(unsigned int sample2PT[], unsigned int predIdx, unsigned int targBit, int start, int end, unsigned int ptId);

  // TODO:  Move somewhere appropriate.
  /**
     @brief Finds the smallest power-of-two multiple >= 'count'.

     @param count is the size to align.

     @return alignment size.
   */
  static inline unsigned int AlignPow(unsigned int count, unsigned int pow) {
    return ((count + (1 << pow) - 1) >> pow) << pow;
  }

};

#endif
