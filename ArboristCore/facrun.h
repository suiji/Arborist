// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

// FacRuns hold field values accumulated from runs of factors having the
// same value.  That is, they group factor-valued predictors into block
// representations. These values live for a single level, so must be consumed
// before a new level is started.
//
// The 'levelFac' array holds the factor values encountered as the FacRuns are
// built.  Bit encodings for split represenations are built using the contents
// of this array.
//
#ifndef ARBORIST_FACRUN_H
#define ARBORIST_FACRUN_H

#include "predictor.h"

class FacRun {
 protected:
  static int nCardTot;
  static int nPredFac;
  static int FacVal(int pairOffset, int pos);
  static void SetFacVal(int pairOffset, int pos, int fac);
 public:
  static FacRun *levelFR; // Workspace for FacRuns used along level.
  static int *levelFac; // Workspace for FacRun values.
  static int levelMax;
  int start; // Buffer position of start of factor run.
  int end;
  int sCount; // Sample count of factor run:  not always same as length.
  double sum; // Sum of responses associated with run.
  static int PairOffset(int splitIdx, int predIdx);
  static FacRun *RunLookup(int splitIdx, int predIdx, int rk = 0);
  static void LeftTerminus(int pairOffset, int rk, int lhIdx, bool rh = false);
  static void Transition(int pairOffset, int rk, int _sCount, double _sumR);
  static int FacVal(int splitIdx, int predIdx, int pos);
  static void Pack(int pairOffset, int posTo, int posFrom);
  static double Accum(int pairOffset, int pos, int &count, int &length);
  static void LevelReset(int splitCount);
  static double Replay(int splitIdx, int predIdx, int level, int bitStart, int ptLH, int ptRH);
  static void Factory(int _levelMax, int _nPredFac, int _nCardTot);
  static void ReFactory(int _levelMax);
  static void DeFactory();
};

class FacRunReg : public FacRun {
 public:
  static void Transition(int splitIdx, int predIdx, int val, int _sCuont, double _sumR);
  static int DePop(int splitIdx, int predIdx);
};

class FacRunCtg : public FacRun {
 private:
  static const int maxWidthDirect = 10; // Wide threshold:  when to sample i/o use all subsets.
  static int totalWide; // Sum of wide offsets;
  static int *wideOffset;
  static double *rvWide; // Random variates for wide factors.
  static double *facCtgSum;
  static int ctgWidth;
  static int SetWide(); // Sets offsets once per simulation.
  static int SumOffset(int splitIdx, int predIdx, int val, int yCtg);
  static void Init(int splitIdx, int predIdx, int val, int yCtg, double yVal);
  static void Incr(int splitIdx, int predIdx, int val, int yCtg, double yVal);
  static int Shrink(int depth, int container[]);
 public:
  static void LeftTerminus(int splitIdx, int predIdx, int rk, int lhIdx, int yCtg, double yVal, bool rh = false);
  static void Transition(int pairOffset, int top, int rk, int _sCount, double _sumR);
  static double SlotSum(int splitIdx, int predIdx, int slot, int yCtg);
  static int Shrink(int pairOffset, int depth);
  static void TreeInit();
  static void ClearTree();
  static void Factory(int _levelMax, int _nPredFac, int _nCardTot, int _ctgWidth);
  static void ReFactory(int _levelMax);
  static void DeFactory();

  //
  static inline void LevelReset(int splitCount) {
    FacRun::LevelReset(splitCount);
    for (int i = 0; i < splitCount * nCardTot * ctgWidth; i++)
      facCtgSum[i] = 0.0;
  }

};

// Implemented as a set of arrays parallel to the FacAccum set.
// Specialized to work with the FacRun structure.
//
// The 'key' and 'fac' arrays should be long enough to allow 
// indexing of the full factor set from within a vector local to
// a given predictor.  The 'vacant' array, on the other hand, need
// only have length
//
// Implements binary heap tailored to this application.
//
typedef struct { double key; int fac; } BHPair;

class BHeap {
 public:
  static int *vacant; // One per heap (i.e., per factor/split-index pair).
  static BHPair *bhPair; // #fac per heap.
  static inline int Parent(int idx) { return (idx-1) >> 1; };
  static int *Vacant(int splitIdx, int predIdx); 
  static int Depopulate(int splitIdx, int predIdx, int container[]);
  static void Insert(int pairOffset , int splitIdx , int predIdx, int _sCount, double _sumR);
  static void Reset(int, int);
};

// Offset of factor-wide workspace for this factor/split-index pair.
//
inline int FacRun::PairOffset(int splitIdx, int predIdx) {
  return  Predictor::FacCard(predIdx) * splitIdx + Predictor::FacOffset(predIdx) * levelMax;
}

//
inline int FacRun::FacVal(int pairOffset, int pos) {
  return levelFac[pairOffset + pos];
}
// Returns the factor value at the specified pair and slot coordinates.
inline int FacRun::FacVal(int splitIdx, int predIdx, int pos) {
  return FacVal(PairOffset(splitIdx, predIdx), pos);
}

// Assuming that 'posTo' <= 'posFrom', effects a packing of compressed
// rank vector.
//
inline void FacRun::Pack(int pairOffset, int posTo, int posFrom) {
  int rk = levelFac[pairOffset + posFrom];
  levelFac[pairOffset + posTo] = rk;
}

inline void FacRun::SetFacVal(int pairOffset, int pos, int rk) {
  levelFac[pairOffset + pos] = rk;
}

inline void FacRun::Transition(int pairOffset, int rk, int _sCount, double _sumR) {
  FacRun *facRun = levelFR + pairOffset + rk;
  facRun->sum = _sumR;
  facRun->sCount = _sCount;
}

// Invokes its FacRun antecedent, then inserts the rank into the binary heap.
// The compressed rank vector is not written until Depop(), as a result of
// which the heap sorts the ranks by weight.
//
inline void FacRunReg::Transition(int splitIdx, int predIdx, int rk, int _sCount, double _sumR) {
  int pairOffset = PairOffset(splitIdx, predIdx);
  FacRun::Transition(pairOffset, rk, _sCount, _sumR);

  BHeap::Insert(pairOffset, splitIdx, predIdx, rk, _sumR / _sCount);
}

// Invokes its FacRun antecedent, then records the rank in the compressed vector
// at the current top.
//
inline void FacRunCtg::Transition(int pairOffset, int top, int rk, int _sCount, double _sumR) {
  FacRun::Transition(pairOffset, rk, _sCount, _sumR);
  SetFacVal(pairOffset, top, rk);
}


// Pulls sorted indices from the heap and places into 'facOrd[]'.
// Returns count of items pulled.
//
inline int FacRunReg::DePop(int splitIdx, int predIdx) {
  return BHeap::Depopulate(splitIdx, predIdx, levelFac + PairOffset(splitIdx, predIdx));
}

inline double FacRun::Accum(int pairOffset, int pos, int &count, int &length) {
  int rk = FacVal(pairOffset, pos);
  FacRun *fRun = levelFR + pairOffset + rk;
  count += fRun->sCount;
  length += 1 + fRun->end - fRun->start;

  return fRun->sum;
}

// Computes the split/pred/rank/ctg coordinate for use by checkerboard scorer.
//
inline int FacRunCtg::SumOffset(int splitIdx, int predIdx, int rk, int yCtg) {
  //  int pairOffset = PairOffset(splitIdx, predIdx);
  //  return yCtg + ctgWidth * (pairOffset + rk);
  return yCtg + ctgWidth * (nCardTot * splitIdx + Predictor::FacOffset(predIdx) + rk);
}

// Initializes the checkerboard value at the split/pred/rank/ctg coordinate.
//
inline void FacRunCtg::Init(int splitIdx, int predIdx, int rk, int yCtg, double yVal) {
  facCtgSum[SumOffset(splitIdx, predIdx, rk, yCtg)] = yVal;
}

// Increments the checkerboard value at the split/pred/rank/ctg coordinate.
//
inline void FacRunCtg::Incr(int splitIdx, int predIdx, int rk, int yCtg, double yVal) {
  facCtgSum[SumOffset(splitIdx, predIdx, rk, yCtg)] += yVal;
}

// Looks up the rank associated with the split/pred/pos coordinate.
// Ranks are known when the checkboard values are set, but must be retrieved
// from the rank vector when needed later.
//
// Returns the checkerboard value accumalated at that coordinate.
//
// N.B.:  The actual rank associated with 'pos' can remain hidden.
//
inline double FacRunCtg::SlotSum(int splitIdx, int predIdx, int pos, int yCtg) {
  int pairOffset = PairOffset(splitIdx, predIdx);
  int rk = FacVal(pairOffset, pos);
  return facCtgSum[SumOffset(splitIdx, predIdx, rk, yCtg)];
}

// Deletes randomly-selected elements of facOrd[] to obtain a sample
// set that can be visited in acceptable time.
//
inline int FacRunCtg::Shrink(int pairOffset, int depth) {
  if (depth > maxWidthDirect)
    depth = Shrink(depth, levelFac + pairOffset);

  return depth;
}

// Stamps the left terminus of the current run, which is assumed to be visited
// from right to left.  If 'rEdge' is true, then a new run has commenced and
// right terminus is also set to the current index.
//
inline void FacRun::LeftTerminus(int pairOffset, int rk, int lhIdx, bool rEdge) {
  FacRun *facRun = levelFR + pairOffset + rk;
  facRun->start = lhIdx;
  if (rEdge)
    facRun->end = lhIdx;
}


inline void FacRunCtg::LeftTerminus(int splitIdx, int predIdx, int rk, int lhIdx, int yCtg, double yVal, bool rEdge) {
  int pairOffset = PairOffset(splitIdx, predIdx);
  FacRun::LeftTerminus(pairOffset, rk, lhIdx, rEdge);

  if (rEdge) {
    Init(splitIdx, predIdx, rk, yCtg, yVal);
  }
  else
    Incr(splitIdx, predIdx, rk, yCtg, yVal);
}

// Resets the top of the heap to zero.  This should already happen if
// the heap is depopulated some time after it is filled.
//
inline void BHeap::Reset(int splitIdx, int predIdx) {
  int *vac = Vacant(splitIdx, predIdx);
  *vac = 0;
}

inline int *BHeap::Vacant(int splitIdx, int predIdx) {
  int facIdx = Predictor::FacIdx(predIdx);
  return vacant + facIdx * FacRun::levelMax + splitIdx;
}

inline void BHeap::Insert(int pairOffset, int splitIdx, int predIdx, int _fac, double _key) {
  BHPair *pairVec = bhPair + pairOffset;
  int *vac = Vacant(splitIdx, predIdx);
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

// Empties the rank values contained in the heap into 'container' in
// weight-sorted sorted order.
//
// Returns number of rank values transferred.
//
inline int BHeap::Depopulate(int splitIdx, int predIdx, int container[]) {
  int pairOffset = FacRun::PairOffset(splitIdx, predIdx);
  BHPair *pairVec = bhPair + pairOffset;
  int vac = *Vacant(splitIdx, predIdx);

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
  *Vacant(splitIdx, predIdx) = 0;

  return vac;
}

#endif
