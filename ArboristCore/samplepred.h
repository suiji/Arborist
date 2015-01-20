// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_SAMPLEPRED_H
#define ARBORIST_SAMPLEPRED_H

// Contains the sample data used by predictor-specific sample-walking pass.
// SamplePred appear in predictor order, grouped by node.  They store the
// y-value, run class and sample index for the predictor position to which they
// correspond.
//
class SamplePred {
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

  // Returns the category associated with a sample (i.e., _yCtg) by
  // de-jittering the mean response with a round-nearest.
  //
  // The category could, alternatively, be recorded in an object subclassed
  // under class SamplePred.  This would require that the value be restaged,
  // which happens for all predictors at all splits.  It would also require
  // that distinct SamplePred classes be maintained for SampleReg and
  // SampleCtg.  Recomputing the category value on demand, then, seems an
  // easier way to go.
  //

  // TODO:  check bounds.
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
  static void SetCtg(SamplePred samplePred[], int idx, int sIdx, double _yVal, int _rank, int _rowRun, int _ctg) {
    SamplePred *tOrd = &samplePred[idx];
    tOrd->sampleIdx = sIdx;
    tOrd->yVal = _yVal;
    tOrd->rank = _rank;
    tOrd->rowRun = _rowRun + (_ctg << ctgShift);
  }

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
  static void SetReg(SamplePred samplePred[], int idx, int sIdx, double _yVal, int _rank, int _rowRun) {
    SamplePred *tOrd = &samplePred[idx];
    tOrd->sampleIdx = sIdx;
    tOrd->yVal = _yVal;
    tOrd->rank = _rank;
    tOrd->rowRun = _rowRun;
  }

  static void RegFields(const SamplePred samplePred[], int i, double &_yVal, int &_rank, int &_rowRun) {
    SamplePred tOrd = samplePred[i];
    _yVal = tOrd.yVal;
    _rank = tOrd.rank;
    _rowRun = tOrd.rowRun;
  }

  static inline SamplePred* BufferOff(int predIdx, int level) {
    return samplePredWS + nSamp * (2*predIdx + ((level & 1) > 0 ? 1 : 0));
  }

  // Determines whether the samples from 'start' to 'end' contain a run.
  //
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
