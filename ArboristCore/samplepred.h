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

/**
 @brief Contains the sample data used by predictor-specific sample-walking pass.
*/
class SamplePred {
  // SamplePred appear in predictor order, grouped by node.  They store the
  // y-value, run class and sample index for the predictor position to which they
  // correspond.

  // Predictor-based sample orderings, double-buffered by level value.
  //
  static SamplePred *samplePredWS;
  static int nSamp;
 public:
  double yVal; // sum of response values associated with sample.
  unsigned int rowRun; // # occurrences of row sampled.  << # rows.
  int rank; // True rank, with ties identically receiving lowest applicable value.
  int sampleIdx; // RV index for this row.  Used by CTG as well as on replay.

  static int ctgShift; // Packing parameter:  only client is categorical response.
  static void RestageOne(const SamplePred source[], SamplePred targ[], int startIdx, int endIdx, int pt, int idx);
  static void RestageTwo(const SamplePred source[], SamplePred targ[], int startIdx, int endIdx, int ptL, int ptR, int lhIdx, int rhIdx);

  // The category could, alternatively, be recorded in an object subclassed
  // under class SamplePred.  This would require that the value be restaged,
  // which happens for all predictors at all splits.  It would also require
  // that distinct SamplePred classes be maintained for SampleReg and
  // SampleCtg.  Recomputing the category value on demand, then, seems an
  // easier way to go.
  //

  // TODO:  check that field is wide enough to hold values of longest run and
  // response cardinality.
  /**
     @brief  Packs the category associated with a sample (i.e., _yCtg) into
     the same field as the run value.

     @param maxRun is the largest run among the bagged samples.

     @return void.
   */
  static void SetCtgShift(int maxRun) {
    ctgShift = 1;
    int ctgMask = 1;
    while (ctgMask < maxRun) {
      ctgShift++;
      ctgMask <<= 1;
    }
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

     @param _rowRun is the number of instances of the current row in this sample.

     @param _ctg is the categorical response value.

     @return void.
   */
  static void SetCtg(SamplePred samplePred[], int idx, int sIdx, double _yVal, int _rank, int _rowRun, int _ctg) {
    SamplePred *tOrd = &samplePred[idx];
    tOrd->sampleIdx = sIdx;
    tOrd->yVal = _yVal;
    tOrd->rank = _rank;
    tOrd->rowRun = _rowRun + (_ctg << ctgShift);
  }

  /**
     @brief Reports SamplePred contents for categorical response.

     @param samplePred is the array base.

     @param i is the index at which to write.

     @param _yVal outputs the proxy response value.

     @param _rank outputs the predictor rank.

     @param _rowRun outputs the multiplicity of the row in this sample.

     @param _yCtg outputs the response value.

     @return void, with output reference parameters.
   */
  static void CtgFields(const SamplePred samplePred[], int i, double &_yVal, int &_rank, int &_rowRun, int &_yCtg) {
    SamplePred tOrd = samplePred[i];
    _yVal = tOrd.yVal;
    _rank = tOrd.rank;
    _rowRun = tOrd.rowRun & ((1 << ctgShift) - 1);
    _yCtg = tOrd.rowRun >> ctgShift;
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

     @param _rowRun is the number of instances of the current row in this sample.

     @return void, with output reference parameters.
   */
  static void SetReg(SamplePred samplePred[], int idx, int sIdx, double _yVal, int _rank, int _rowRun) {
    SamplePred *tOrd = &samplePred[idx];
    tOrd->sampleIdx = sIdx;
    tOrd->yVal = _yVal;
    tOrd->rank = _rank;
    tOrd->rowRun = _rowRun;
  }

  /**
     @brief Reports SamplePred contents for categorical response.

     @param samplePred is the array base.

     @param i is the index at which to write.

     @param _yVal outputs the response value.

     @param _rank outputs the predictor rank.

     @param _rowRun outputs the multiplicity of the row in this sample.

     @return void.
   */
  static void RegFields(const SamplePred samplePred[], int i, double &_yVal, int &_rank, int &_rowRun) {
    SamplePred tOrd = samplePred[i];
    _yVal = tOrd.yVal;
    _rank = tOrd.rank;
    _rowRun = tOrd.rowRun;
  }

  /**
     @brief Toggles between positions in workspace double buffer, by level.

     @param predIdx is the predictor coordinate.

     @param level is the current level.

     @return starting position within workspace.
   */
  static inline SamplePred* BufferOff(int predIdx, int level) {
    return samplePredWS + nSamp * (2*predIdx + ((level & 1) > 0 ? 1 : 0));
  }

  /**
   @brief Determines whether the consecutive index positions are a run of predictor values.

   @param start is beginning index position of potential run.

   @param end is the ending index position of a potential run.

   @return whether a run is encountered.
  */
  static inline bool IsRun(SamplePred *samplePred, int start, int end) {
    return samplePred[start].rank == samplePred[end].rank;
  }

  static void TreeInit(int nPred, int nSamp);
  static void TreeClear();
  static void SplitRanks(int predIdx, int level, int spIdx, int &rkLow, int &rkHigh);
  static double Replay(int predIdx, int level, int start, int end, int ptId);
  static void Restage(int predIdx, int splitCount, int rhStart, int level);
};

class RestageMap {
  static RestageMap *restageMap;
  static int splitCount; // Number of splits at this level.
  static int totLhIdx; // For stable partitions, first RH index follows all LH indices.
  int lNext;
  int rNext;
  int lhIdxCount;
  int rhIdxCount;
  int ptL;
  int ptR;
  int startIdx;
  int endIdx;
  void Restage(const SamplePred source[], SamplePred targ[], int lhIdx, int rhIdx);
public:
  static void Factory(int levelMax);
  static void ReFactory(int levelMax);
  static void DeFactory();

  /**
     @brief Sets the state fields for the next level.

     @param _splitCount is the number of live index nodes in the upcoming level.

     @param _totLhIdx is the total count of indices subsumed by these nodes.

     @return void.
   */
  static void Commence(int _splitCount, int _totLhIdx) {
    splitCount = _splitCount;
    totLhIdx = _totLhIdx;
  }
  static void ConsumeSplit(int _splitIdx, int _lNext, int _rNext, int _lhIdxCount, int _rhIdxCount);
  static void Restage(int predIdx, int level);
  void NoteRuns(int predIdx, int level, const SamplePred targ[], int lhIdx, int rhIdx);
  void TransmitRun(int splitIdx, int predIdx, int level);
};


#endif
