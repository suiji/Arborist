// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file facrun.h

   @brief Definitions for the FacRun classes, which maintain runs of
   factor-valued predictors.  Many methods are inlined and rely on caller-maintained
   state.

   @author Mark Seligman

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

/**
   @brief FacRun objects are allocated per-level, per-predictor.  Reallocation is
   necessary if the static 'levelMax' value increases.
 */
class FacRun {
 protected:
  static int nCardTot;
  static int nPredFac;
  static int predFacFirst; // Useful for iterators.
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
  static void Factory(int _levelMax, int _nPredFac, int _nCardTot, int _predFacFirst);
  static void ReFactory(int _levelMax);
  static void DeFactory();
};

/**
   @brief Factor run methods specific to regression trees.
 */
class FacRunReg : public FacRun {
 public:
  static void Transition(int splitIdx, int predIdx, int val, int _sCuont, double _sumR);
  static int DePop(int splitIdx, int predIdx);
};

/**
   @brief Factor run methods and members specific to classification trees.
 */
class FacRunCtg : public FacRun {
 private:
  static const int maxWidthDirect = 10; // Wide threshold:  sampling vs. all subsets.
  static int *wideOffset;  // Offset position lookup vector for wide factors.
  static int totalWide; // Sum of wide offsets, according to threshold.
  static double *rvWide; // Random variates for selecting wide factor values.
  static double *facCtgSum;
  static int ctgWidth;
  static int SetWideOffset(); // Sets offsets once per simulation.
  static int SumOffset(int splitIdx, int predIdx, int val, int yCtg);
  static void Init(int splitIdx, int predIdx, int val, int yCtg, double yVal);
  static void Incr(int splitIdx, int predIdx, int val, int yCtg, double yVal);
  static int WideOffset(int splitIdx, int predIdx, int splitCount);
  static int Shrink(int splitIdx, int predIdx, int splitCount, int depth, int container[]);
 public:
  static void LeftTerminus(int splitIdx, int predIdx, int rk, int lhIdx, int yCtg, double yVal, bool rh = false);
  static void Transition(int pairOffset, int top, int rk, int _sCount, double _sumR);
  static double SlotSum(int splitIdx, int predIdx, int slot, int yCtg);
  static int Shrink(int splitIdx, int predIdx, int splitCount, int depth);
  static void Factory(int _levelMax, int _nPred, int _nPredFac, int _nCardTot, int _predFacFirst, int _ctgWidth);
  static void ReFactory(int _levelMax);
  static void DeFactory();

  //
  static void LevelReset(int splitCount);
};

/**
   @brief Implementation of binary heap tailored to FacRunReg.

   Implemented as a set of arrays parallel to the FacAccum set.
   The 'key' and 'fac' arrays should be long enough to allow 
   indexing of the full factor set from within a vector local to
   a given predictor.
*/

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

/**
  @brief Offset of factor-wide workspace for this factor/split-index pair.

  @param splitIdx is split index of the pair.

  @param predIdx is the predictor index of the pair.

  @return The level-base offset corresponding to this pair.
*/
inline int FacRun::PairOffset(int splitIdx, int predIdx) {
  return  Predictor::FacCard(predIdx) * splitIdx + Predictor::FacOffset(predIdx) * levelMax;
}

/**
   @brief Looks up a rank value for a pair, position coordinate.

   @param pairOffset is a cached offset for the pair.

   @param pos is the position offset.

   @return The factor value at this coordinate.
*/
inline int FacRun::FacVal(int pairOffset, int pos) {
  return levelFac[pairOffset + pos];
}

/**
   @brief As above, but without a cached pair offset.

   @param splitIdx is the split index.

   @param predIndex is the predictor index.

   @param pos is the position offset.

   @return Factor value at coordinate passed.   
 */
inline int FacRun::FacVal(int splitIdx, int predIdx, int pos) {
  return FacVal(PairOffset(splitIdx, predIdx), pos);
}

/**
   @brief Assuming that 'posTo' <= 'posFrom', effects a packing of compressed rank vector.

   @param pairOffset is a cached offset for the pair.

   @param posTo is the target position in the rank vector.

   @param posFrom is the source position in the rank vector.

   @return void.
*/
inline void FacRun::Pack(int pairOffset, int posTo, int posFrom) {
  int rk = levelFac[pairOffset + posFrom];
  levelFac[pairOffset + posTo] = rk;
}

/**
   @brief Sets a value in the rank vector.

   @param pairOffset is a cached pair offset.

   @param pos is the position in the rank vector at which to write.

   @param rk is the rank to set at the specified position.

   @return void.
 */
inline void FacRun::SetFacVal(int pairOffset, int pos, int rk) {
  levelFac[pairOffset + pos] = rk;
}

/**
   @brief Concludes accumulating information for a run.

   @param pairOffset is a cached pair offset for the run.

   @param rk is the rank associated with the run.

   @param _sCount is the accumulated sample count.

   @param _sumR is the accumulated sum of response values.

   @return void.
 */
inline void FacRun::Transition(int pairOffset, int rk, int _sCount, double _sumR) {
  FacRun *facRun = levelFR + pairOffset + rk;
  facRun->sum = _sumR;
  facRun->sCount = _sCount;
}

/**
   @brief Invokes its FacRun antecedent, then inserts the rank into the binary heap.

   The compressed rank vector is not written until Depop(), as a result of
   which the heap sorts the ranks by weight.

   @param splitIdx is the split index.

   @param predIdx is the predictor index.

   @param rk is the associated rank.

   @param _sCount is the accumulated sample count.

   @param _sumR is the accumulated sum of response values.

   @return void.
*/
inline void FacRunReg::Transition(int splitIdx, int predIdx, int rk, int _sCount, double _sumR) {
  int pairOffset = PairOffset(splitIdx, predIdx);
  FacRun::Transition(pairOffset, rk, _sCount, _sumR);

  BHeap::Insert(pairOffset, splitIdx, predIdx, rk, _sumR / _sCount);
}

/**
   @brief Invokes its FacRun antecedent, then records the rank in the compressed vector
   at the current top.

   @param pairOffset is a cached coordinate for the pair.

   @param top is the current top of the rank vector.

   @param rk is the current rank.

   @param _sCount is the accumulated sum of sample counts.

   @param _sumR is the accumulated sum of response values.

   @return void.
*/
inline void FacRunCtg::Transition(int pairOffset, int top, int rk, int _sCount, double _sumR) {
  FacRun::Transition(pairOffset, rk, _sCount, _sumR);
  SetFacVal(pairOffset, top, rk);
}


/**
   @brief Depopulates the heap associated with a pair and places sorted ranks into rank vector.

   @param splitIdx is the split index for the pair.

   @param predIdx is the predictor index.

   @return count of items pulled.
*/
inline int FacRunReg::DePop(int splitIdx, int predIdx) {
  return BHeap::Depopulate(splitIdx, predIdx, levelFac + PairOffset(splitIdx, predIdx));
}

/**
   @brief Accumulates sample and index counts in an order specified by caller.

   @param pairOffset is a cached offset for the pair.

   @param pos is the position to dereference in the rank vector.

   @param count accumulates sample counts.

   @param length accumulates index counts.

   @return response sum for the run, as well as reference parameters.
 */
inline double FacRun::Accum(int pairOffset, int pos, int &count, int &length) {
  int rk = FacVal(pairOffset, pos);
  FacRun *fRun = levelFR + pairOffset + rk;
  count += fRun->sCount;
  length += 1 + fRun->end - fRun->start;

  return fRun->sum;
}

/**
 @brief Computes the split/pred/rank/ctg coordinate for use by checkerboard scorer.

 @param splitIdx is the pair split index.

 @param predIdx is the pair predictor index.

 @param rk is the rank.

 @param yCtg is the categorical response.

 @return Checkerboard offset.
*/
inline int FacRunCtg::SumOffset(int splitIdx, int predIdx, int rk, int yCtg) {
  return yCtg + ctgWidth * (nCardTot * splitIdx + Predictor::FacOffset(predIdx) + rk);
}

/**
 @brief Initializes the split/pred/rank/ctg coordinate for use by checkerboard scorer.

 @param splitIdx is the pair split index.

 @param predIdx is the pair predictor index.

 @param rk is the rank.

 @param yCtg is the categorical response.

 @return void.
*/
inline void FacRunCtg::Init(int splitIdx, int predIdx, int rk, int yCtg, double yVal) {
  facCtgSum[SumOffset(splitIdx, predIdx, rk, yCtg)] = yVal;
}

/**
 @brief Increments the split/pred/rank/ctg coordinate for use by checkerboard scorer.

 @param splitIdx is the pair split index.

 @param predIdx is the pair predictor index.

 @param rk is the rank.

 @param yCtg is the categorical response.

 @param yVal is the (proxy) response score by which to increment.

 @return void.
*/
inline void FacRunCtg::Incr(int splitIdx, int predIdx, int rk, int yCtg, double yVal) {
  facCtgSum[SumOffset(splitIdx, predIdx, rk, yCtg)] += yVal;
}

/**
   @brief Looks up the rank associated with the split/pred/pos/ctg coordinate.

   Ranks are known when the checkboard values are set, but must be retrieved
   from the rank vector when needed later.  The actual rank associated with 'pos'
   can thus remain hidden.

   @param splitIdx is the split index for the pair.

   @param predIdx is the predictor index.

   @param pos is the position in the rank vector.

   @param yCtg is the categorical response.

   @return the checkerboard value accumalated at that coordinate.
*/
inline double FacRunCtg::SlotSum(int splitIdx, int predIdx, int pos, int yCtg) {
  int pairOffset = PairOffset(splitIdx, predIdx);
  int rk = FacVal(pairOffset, pos);
  return facCtgSum[SumOffset(splitIdx, predIdx, rk, yCtg)];
}

/**
 @brief Deletes randomly-selected elements of the rank vector to obtain a sample set
 that can be visited in acceptable time.

 @param splitIdx is the split index of the pair.

 @param predIdx is the predictor index of the pair.

 @param splitCount is the count of splits at the current level.

 @param top is the current top position of the rank vector.

 @return Size of shrunken rank vector.
*/
inline int FacRunCtg::Shrink(int splitIdx, int predIdx, int splitCount, int top) {
  return top > maxWidthDirect ? Shrink(splitIdx, predIdx, splitCount, top, levelFac + PairOffset(splitIdx, predIdx)) : top;
}

/**
   @brief  Stamps the left terminus of the current run, which is assumed to be visited
   from right to left.  If 'rEdge' is true, then a new run has commenced and right terminus
   is also set to the current index.

   @param pairOffset is the cached pair position.

   @param rk is the rank.

   @param lhIdx is the current index position.

   @param rEdge indicates whether to also stamp the right terminus.
   @return void.
*/
inline void FacRun::LeftTerminus(int pairOffset, int rk, int lhIdx, bool rEdge) {
  FacRun *facRun = levelFR + pairOffset + rk;
  facRun->start = lhIdx;
  if (rEdge)
    facRun->end = lhIdx;
}

/**
   @brief As above, but also invokes the checkerboard initializers.

   @param splitIdx is the split index for the pair.

   @param predIdx is the predictor index.

   @param rk is the rank.

   @param lhIdx is the index position.

   @param yCtg is the categorical response.

   @param yVal is the proxy response.

   @return void.
   
 */
inline void FacRunCtg::LeftTerminus(int splitIdx, int predIdx, int rk, int lhIdx, int yCtg, double yVal, bool rEdge) {
  int pairOffset = PairOffset(splitIdx, predIdx);
  FacRun::LeftTerminus(pairOffset, rk, lhIdx, rEdge);

  if (rEdge) {
    Init(splitIdx, predIdx, rk, yCtg, yVal);
  }
  else
    Incr(splitIdx, predIdx, rk, yCtg, yVal);
}

/**
   @brief Computes the workspace offset for predictors known to rank-vector shrinkage.

   @param splitIdx is the split index.

   @param predIdx is the predictor index.

   @param splitCount is the split count for the current level.

   @return workspace offset.
 */
inline int FacRunCtg::WideOffset(int splitIdx, int predIdx, int splitCount) {
  return splitCount * wideOffset[predIdx] + splitIdx * (1 + Predictor::FacCard(predIdx));
}

/**
   @brief Resets the top of the heap to zero.  This should already happen if the heap
   is depopulated some time after it is filled.

   @param splitIdx is the split index.

   @param predIdx is the predictor index.

   @return void.
*/
inline void BHeap::Reset(int splitIdx, int predIdx) {
  int *vac = Vacant(splitIdx, predIdx);
  *vac = 0;
}

/**
   @brief Compute an index in the heap's vacancy vector.

   @param splitIdx is the split index.

   @param predIdx is the predictor index.

   @return vacancy vector index.
 */
inline int *BHeap::Vacant(int splitIdx, int predIdx) {
  int facIdx = Predictor::FacIdx(predIdx);
  return vacant + facIdx * FacRun::levelMax + splitIdx;
}

/**
   @brief Inserts a key, value pair into the heap.

   @param pairOffset is the cached pair coordinate.

   @param splitIdx is the split index.

   @param predIdx is the predictor index.

   @param _fac is the factor value.

   @param _key is the associated key.

   @return void.
 */
inline void BHeap::Insert(int pairOffset, int splitIdx, int predIdx, int _fac, double _key) {
  BHPair *pairVec = bhPair + pairOffset;
  int *vac = Vacant(splitIdx, predIdx);
  int idx = *vac;
  *vac = idx + 1;
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

/**
   @brief Empties the rank values contained in the heap into 'container' in weight-sorted order.

   @param splitIdx is the split index of the pair.

   @param predIdx is the predictor index.

   @param container is an output vector holding the sorted rank values.

   @return number of rank values transferred, with output parameter vector.
*/
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
