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

/**
 */
class SPNode {
  static unsigned int runShift; // Pack:  nonzero iff categorical response.
 protected:
  FltVal yVal; // sum of response values associated with sample.
  unsigned int rank; // True rank, with ties identically receiving lowest applicable value.
  unsigned int sCount; // # occurrences of row sampled.  << # rows.
 public:
  static void Immutables(unsigned int ctgWidth);
  static void DeImmutables();

  inline unsigned int Rank() {
    return rank;
  }
  // These methods should only be called when the response is known
  // to be regression, as it relies on a packed representation specific
  // to that case.
  //
  /**
     @brief Sets the SamplePred contents for regression response.

     @param samplePred is the array base.

     @param idx is the index at which to write.

     @param sIdx is the associated sample index.

     @param _yVal is the response value.

     @param _rank is the rank of the associated predictor value.

     @param _sCount is the number of instances of the current row in this sample.

     @return void, with output reference parameters.
   */
  void SetReg(FltVal _yVal, unsigned int _rank, unsigned int _sCount) {
    yVal = _yVal;
    rank = _rank;
    sCount = _sCount;
  }


  /**
     @brief Reports SamplePred contents for regression response.  Cannot
     be used with categorical response, as 'sCount' value reported here
     is unshifted.

     @param _yVal outputs the response value.

     @param _rank outputs the predictor rank.

     @param _sCount outputs the multiplicity of the row in this sample.

     @return void.
   */
  inline void RegFields(FltVal &_yVal, unsigned int &_rank, unsigned int &_sCount) const {
    _yVal = yVal;
    _rank = rank;
    _sCount = sCount;
  }

  // These methods should only be called when the response is known
  // to be categorical, as it relies on a packed representation specific
  // to that case.
  //

  /**
     @brief Sets the SamplePred contents for categorical response.

     @param samplePred is the base array.

     @param idx is the index at which to write.

     @param sIdx is the associated sample index.

     @param _yVal is the proxy response.

     @param _rank is the rank of the associated predictor value.

     @param _sCount is the number of instances of the current row in this sample.

     @param _ctg is the categorical response value.

     @return void.
   */
  void SetCtg(FltVal _yVal, unsigned int _rank, unsigned int _sCount, unsigned int _ctg) {
    yVal = _yVal;
    rank = _rank;
    sCount = _ctg + (_sCount << runShift);
  }

  
  /**
     @brief Reports SamplePred contents for categorical response.  Can
     be called with regression response if '_yCtg' value ignored.

     @param _yVal outputs the proxy response value.

     @param _rank outputs the predictor rank.

     @param _yCtg outputs the response value.

     @return sample count, with output reference parameters.
   */
  inline unsigned int CtgFields(FltVal &_yVal, unsigned int &_rank, unsigned int &_yCtg) const {
    _yVal = yVal;
    _rank = rank;
    _yCtg = sCount & ((1 << runShift) - 1);

    return sCount >> runShift;
  }


    /**
     @brief Variant of above, for which rank determined separately.

     @param _yVal outputs the proxy response value.

     @param _sCount outputs the multiplicity of the row in this sample.

     @param _yCtg outputs the response value.

     @return sample count of node, with output reference parameters.
   */
  inline unsigned int CtgFields(FltVal &_yVal, unsigned int &_yCtg) const {
    _yVal = yVal;
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
     @brief Accessor for 'yVal' field

     @return sum of y-values for sample.
   */
  inline FltVal YVal() {
    return yVal;
  }

};


/**
 @brief Contains the sample data used by predictor-specific sample-walking pass.
*/
class SamplePred {
  // SamplePred appear in predictor order, grouped by node.  They store the
  // y-value, run class and sample index for the predictor position to which they
  // correspond.

  // Predictor-based sample orderings, double-buffered by level value.
  //
  static int bufferSize; // Number of cells in single buffer:  nSamp * nPred.
  static unsigned int pitchSP; // Pitch of SPNode vector, in bytes.
  static unsigned int pitchSIdx; // Pitch of SIdx vector, in bytes.
  static int predNumFirst;
  static int predNumSup;
  static int predFacFirst;
  static int predFacSup;

 protected:
  static unsigned int nRow;
  static int nPred;
  static int nSamp;
  SPNode* nodeVec;

  // 'sampleIdx' could be boxed with SPNode.  While it is used in both
  // replaying and restaging, though, it plays no role in splitting.  Maintaining
  // a separate vector permits a 16-byte stride to be used for splitting.  More
  // significantly, it reduces memory traffic incurred by transposition on the
  // coprocessor.
  //
  unsigned int *sampleIdx; // RV index for this row.  Used by CTG as well as on replay.
 public:
  static void Immutables(int _nPred, int _nSamp, unsigned int _nRow, unsigned int _ctgWidth);
  static void DeImmutables();
  SamplePred();
  ~SamplePred();

  void StageReg(const class PredOrd *predOrd, const class SampleNode sampleReg[], const int sCountRow[], const int sIdxRow[]);
  void StageReg(const class PredOrd *dCol, const class SampleNode sampleReg[], const int sCountRow[], const int sIdxRow[], int predIdx);
  
  void StageCtg(const class PredOrd *predOrd, const class SampleNodeCtg sampleCtg[], const int sCountRow[], const int sIdxRow[]);
  void StageCtg(const class PredOrd *dCol, const class SampleNodeCtg sampleCtg[], const int sCountRow[], const int sIdxRow[], int predIdx);
  
  static inline unsigned int PitchSP() {
    return pitchSP;
  }

  static inline unsigned int PitchSIdx() {
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
  static inline int LevelOff(int level) {
    return (level & 1) == 0 ? 0 : bufferSize;
  }

  /**
     @brief Allows caller to cache the node buffer for the current level.

     @param level is the current level.

     @return buffer starting position.
   */
  inline SPNode* NodeBase(int level) const {
    return nodeVec + LevelOff(level);
  }

  /**
     @brief Allows lightweight lookup of predictor's SPNode vector using
     cached node base.
 
     @param nodeBase is the node base, cached by the caller.

     @param preIdx is the predictor index.

     @return node vector section for this predictor.
   */
  inline static SPNode* PredBase(class SPNode *nodeBase, int predIdx) {
    return nodeBase + nSamp * predIdx; 
  }
  
  /**

     @param predIdx is the predictor coordinate.

     @param level is the current level.

     @return starting position within workspace.
   */
  static inline int BufferOff(int predIdx, int level) {
    return nSamp * predIdx + LevelOff(level);
  }

  inline SPNode* Buffers(int predIdx, int level, unsigned int*& sIdx) {
    int offset = BufferOff(predIdx, level);
    sIdx = sampleIdx + offset;
    return nodeVec + offset;
  }

  /**
     @brief Returns buffer containing splitting information.
   */
  inline SPNode* SplitBuffer(int predIdx, int level) {
    return nodeVec + BufferOff(predIdx, level);
  }

/**
   @brief Looks up source and target vectors.

   @param predIdx is the predictor column.

   @param level is the upcoming level.

   @return void, with output parameter vectors.
 */
  inline void Buffers(int predIdx, int level, SPNode *&source, unsigned int *&sIdxSource, SPNode *&targ, unsigned int *&sIdxTarg) {
    source = Buffers(predIdx, level-1, sIdxSource);
    targ = Buffers(predIdx, level, sIdxTarg);
  }

  void SplitRanks(int predIdx, int level, int spIdx, unsigned int &rkLow, unsigned int &rkHigh);
  double Replay(int sample2PT[], int predIdx, int level, int start, int end, int ptId);

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
