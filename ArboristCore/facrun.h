/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

// Implements binary heap tailored to this application.
//
#ifndef ARBORIST_FACRUN_H
#define ARBORIST_FACRUN_H

#include "predictor.h"

class FacRun {
 public:
  static int accumCount;
  int start; // Buffer position of start of factor run.
  int end;
  int sCount; // Sample count of factor run:  not always same as length.
  double sum; // Sum of responses associated with run.
  static int WSOffset(int facIdx, int accumIdx);
  static void Reset(const int facIdx, const int accumIdx);
  static FacRun *levelFR; // Workspace for FacRuns used along level.
  static int *levelFROrd; // Workspace for sorted FacRun offsets.
  static void Terminus(int facIdx, int accumIdx, int ord, int lhIdx, bool rh = false);
  static void Insert(const int facIdx, const int accumIdx, const int ord, int _sCount, double sumR);
  static void Transition(const int, const int, const int, const int, const double);
  static int DePop(const int facIdx, const int accumIdx, int * &frOrd);
  static FacRun *Ref(const int facIdx, const int accumIdx, const int ord);
  static void Factory(int _accumCount);
  static void ReFactory(int _accumCount);
  static void DeFactory();
};

class FacRunCtg : public FacRun {
  static double *facCtgSum;
  static int ctgWidth;
 public:
  static void Factory(const int _accumCount, const int _ctgWidth);
  static void ReFactory(const int _accumCount);
  static void DeFactory();
  static void Reset(const int facIdx, const int accumIdx);
  static double *Sum(const int facIdx, const int accumIdx, const int ord);
  static void Terminus(const int facIdx, const int accumIdx, const int ord, const int lhIdx, const int yCtg, const double yVal, const bool rh = false);
};

// Implemented as a set of arrays parallel to the FacAccum set.
// Specialized to work with the FacRun structure.
//
// The 'key' and 'fac' arrays should be long enough to allow 
// indexing of the full factor set from within a vector local to
// a given predictor.  The 'vacant' array, on the other hand, need
// only have length
//
typedef struct { double key; int fac; } BHPair;

class BHeap {
 public:
  static int *vacant; // One per heap (i.e., per factor/accumulator pair).
  static BHPair *bhPair; // #fac per heap.
  static inline int Parent(const int idx) { return (idx-1) >> 1; };
  static int *Vacant(const int facIdx, const int accumIdx); 
  static int Depopulate(const int, const int, int*);
  static void Insert(const int, const int, const int, const double);
  static void Reset(const int, const int);
};

// Offset of factor-wide workspace for this accumulator/factor pair.
//
inline int FacRun::WSOffset(int facIdx, int accumIdx) {
  return Predictor::facTot * accumIdx + Predictor::FacOffset(facIdx);
}

// Zeroes the 'sCount' fields of all FacRuns associated with a given factor-
// valued predictor.  This provides a means to distinguish active runs from
// inactive.
//
inline void FacRun::Reset(const int facIdx, const int accumIdx) {
  FacRun *frBase = levelFR + WSOffset(facIdx, accumIdx);
  for (int i = 0; i < Predictor::FacWidth(facIdx); i++) {
    FacRun *fr = frBase + i;
    fr->start = fr->end = fr->sCount = 0;
    fr->sum = 0.0;
  }
}

inline void FacRunCtg::Reset(const int facIdx, const int accumIdx) {
  FacRun::Reset(facIdx, accumIdx);
  for (int ord = 0; ord < Predictor::FacWidth(facIdx); ord++) {
    double *ctgSum = Sum(facIdx, accumIdx, ord); // Can also just increment by 'facWidth'.
    for (int ctg = 0; ctg < ctgWidth; ctg++)
      ctgSum[ctg] = 0.0;
  }
}

// Stamps the left terminus of the current run, which is assumed to be visited
// from right to left.  Stamps right terminus to same value, if requested for
// initializations.
//
inline void FacRun::Terminus(int facIdx, int accumIdx, int ord, int lhIdx, bool rh) {
  FacRun *facRun = levelFR + WSOffset(facIdx, accumIdx) + ord;
  facRun->start = lhIdx;
  if (rh)
    facRun->end = lhIdx;
}

inline double *FacRunCtg::Sum(const int facIdx, const int accumIdx, const int ord) {
  return facCtgSum + (Predictor::facTot * accumIdx + Predictor::FacOffset(facIdx) + ord) * ctgWidth;
}

inline void FacRunCtg::Terminus(const int facIdx, const int accumIdx, const int ord, const int lhIdx, const int yCtg, const double yVal, const bool rh) {
  FacRun::Terminus(facIdx, accumIdx, ord, lhIdx, rh);
  double *ctgSum = Sum(facIdx, accumIdx, ord);
  if (rh) {
    ctgSum[yCtg] = yVal;
  }
  else { // No transition:  counters accumulate.
    ctgSum[yCtg] += yVal;
  }
}

inline void FacRun::Transition(const int facIdx, const int accumIdx, int const ord, const int _sCount, const double _sumR) {
  FacRun *facRun = levelFR + WSOffset(facIdx, accumIdx) + ord;
  facRun->sum = _sumR;
  facRun->sCount = _sCount;
}

inline void FacRun::Insert(const int facIdx, const int accumIdx, int const ord, const int _sCount, const double _sumR) {
  BHeap::Insert(facIdx, accumIdx, ord, _sumR / _sCount);
}

inline int FacRun::DePop(int facIdx, int accumIdx, int * &facOrd) {
  facOrd = levelFROrd + WSOffset(facIdx, accumIdx);
  int depth = BHeap::Depopulate(facIdx, accumIdx, facOrd);

  return depth;
}

inline FacRun *FacRun::Ref(const int facIdx, const int accumIdx, const int ord) {
  return levelFR + WSOffset(facIdx, accumIdx) + ord;
}


// Resets the top of the heap to zero.  This should already happen if
// the heap is depopulated some time after it is filled.
//
inline void BHeap::Reset(const int facIdx, const int accumIdx) {
  int *vac = Vacant(facIdx, accumIdx);
  *vac = 0;
}

inline int *BHeap::Vacant(const int facIdx, const int accumIdx) {
  return vacant + facIdx * FacRun::accumCount + accumIdx;
}

inline void BHeap::Insert(const int facIdx, const int accumIdx, const int _fac, const double _key) {
  BHPair *pairVec = bhPair + FacRun::WSOffset(facIdx, accumIdx);
  int *vac = Vacant(facIdx, accumIdx);
  int idx = *vac;
  *vac = idx + 1;
  //cout << "Insert " << _fac << " / " << _key <<  " entry: " << idx << endl;
  pairVec[idx].key = _key;
  pairVec[idx].fac = _fac;

  int parIdx = Parent(idx);
  while (parIdx >= 0 && pairVec[parIdx].key > _key) {
    pairVec[idx].key = pairVec[parIdx].key;
    pairVec[idx].fac = pairVec[parIdx].fac;
    pairVec[parIdx].fac = _fac;
    pairVec[parIdx].key = _key;
    idx = parIdx;
    parIdx = Parent(idx);
  }
}

// Empties the contents of the heap into 'container' in sorted order.
// Returns number of elements copied.
//
inline int BHeap::Depopulate(const int facIdx, const int accumIdx, int *container) {
  BHPair *pairVec = bhPair + FacRun::WSOffset(facIdx, accumIdx);
  int vac = *Vacant(facIdx, accumIdx);

  for (int bot = vac-1; bot >= 0; bot--) {
    container[vac - 1 - bot] = pairVec[0].fac;

    // Places bottom element at head and refiles.
    int idx = 0;
    int facRefile = pairVec[idx].fac = pairVec[bot].fac;
    double keyRefile = pairVec[idx].key = pairVec[bot].key;
    int chL = 1;
    int chR = 2;

    // 'chR' remains the lower of the two child indices.  Some short-circuiting below.
    //
    while((chR <= bot && keyRefile > pairVec[chR].key) || (chL <= bot && keyRefile > pairVec[chL].key)) {
      int chIdx =  (chR <= bot && pairVec[chR].key < pairVec[chL].key) ?  chR : chL;
      pairVec[idx].key = pairVec[chIdx].key;
      pairVec[idx].fac = pairVec[chIdx].fac;
      pairVec[chIdx].key = keyRefile;
      pairVec[chIdx].fac = facRefile;
      idx = chIdx;
      chL = 1 + (idx << 1);
      chR = (1 + idx) << 1;
    }
  }
  *Vacant(facIdx, accumIdx) = 0;

  return vac;
}

#endif
