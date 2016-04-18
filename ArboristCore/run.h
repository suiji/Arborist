// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file run.h

   @brief Definitions for the Run classes, which maintain predictor
   runs, especially factor-valued predictors.

   @author Mark Seligman

 */

// FRNodes hold field values accumulated from runs of factors having the
// same value.  That is, they group factor-valued predictors into block
// representations. These values live for a single level, so must be consumed
// before a new level is started.
//
#ifndef ARBORIST_RUN_H
#define ARBORIST_RUN_H

/**
 */
class FRNode {
 public:
  unsigned int rank;
  int start; // Buffer position of start of factor run.
  int end;
  unsigned int sCount; // Sample count of factor run:  not always same as length.
  double sum; // Sum of responses associated with run.

  FRNode() : start(-1), end(-1), sCount(0), sum(0.0) {}
  /**
     @brief Bounds accessor.
   */
  inline void ReplayFields(int &_start, int &_end) {
    _start = start;
    _end = end;
  }

  inline void Set(unsigned int _rank, unsigned int _sCount, double _sum, unsigned int _start, unsigned int _end) {
    rank = _rank;
    sCount = _sCount;
    sum = _sum;
    start = _start;
    end = _end;
  }
};

class BHPair {
 public:
  double key; unsigned int slot;
};

/**
  @brief  Runs only:  caches pre-computed workspace starting indices to
  economize on address recomputation during splitting.
*/
class RunSet {
  int runOff; // Temporary offset storage.
  int heapOff; //
  int outOff; //
  FRNode *runZero; // Base for this run set.
  BHPair *heapZero; // Heap workspace.
  int *outZero; // Final LH and/or output for heap-ordered slots.
  double *ctgZero; // Categorical:  run x ctg checkerboard.
  double *rvZero; // Non-binary wide runs:  random variates for sampling.
  unsigned int runCount;  // Current high watermark:  not subject to shrinking.
  int runsLH; // Count of LH runs.
 public:
  const static unsigned int maxWidth = 10;
  static unsigned int ctgWidth;
  unsigned int safeRunCount;
  unsigned int DeWide();
  void DePop(int pop = 0);
  void Reset(FRNode*, BHPair*, int*, double*, double*);
  void OffsetCache(int runIdx, int bhpIdx, int outIdx);
  void HeapRandom();
  void HeapMean();
  void HeapBinary();


  /**
     @brief Accessor for runCount field.

     @return reference to run count.
   */
  inline unsigned int &RunCount() {
    return runCount;
  }

  
  inline unsigned int CountSafe() {
    return safeRunCount;
  }
  
  
  /**
     @brief "Effective" run count, for the sake of splitting, is the lesser
     of the true run count and 'maxWidth'.

     @return effective run count
   */
  inline unsigned int EffCount() {
    return runCount > maxWidth ? maxWidth : runCount;
  }


  /**
     @brief Looks up sum and sample count associated with a given output slot.

     @param outPos is a position in the output vector.

     @param sCount outputs the sample count at the dereferenced output slot.

     @return sum at dereferenced position, with output reference parameter.
   */
  inline double SumHeap(int outPos, unsigned int &sCount) {
    int slot = outZero[outPos];
    sCount = runZero[slot].sCount;
    
    return runZero[slot].sum;
  }

  
  /**
     @brief Sets run parameters and increments run count.

     @return void.
   */
  void Write(unsigned int rank, unsigned int sCount, double sum, int start, int end) {
    runZero[runCount++].Set(rank, sCount, sum, start, end);
  }


  /**
     @brief Accumulates checkerboard values prior to writing topmost
     run.

     @return void.
   */
  inline double &SumCtg(unsigned int yCtg) {
    return ctgZero[runCount * ctgWidth + yCtg];
  }


  /**
     @return checkerboard value at slot for category.
   */
  inline double SumCtg(int slot, unsigned int yCtg) {
    return ctgZero[slot * ctgWidth + yCtg];
  }


  /**
     @brief Looks up the two binary response sums associated with a given
     output slot.

     @param outPos is a position in the output vector.

     @param cell0 outputs the response at rank 0.

     @param cell1 outputs the response at rank 1.

     @return true iff slot counts differ by at least unity.
   */
  inline bool SumBinary(int outPos, double &cell0, double &cell1) {
    int slot = outZero[outPos];
    cell0 = SumCtg(slot, 0);
    cell1 = SumCtg(slot, 1);

    unsigned int sCount = runZero[slot].sCount;
    int slotNext = outZero[outPos+1];
    // Cannot test for floating point equality.  If sCount values are unequal,
    // then assumes the two slots are significantly different.  If identical,
    // then checks whether the response values are likely different, given
    // some jittering.
    // TODO:  replace constant with value obtained from class weighting.
    return sCount != runZero[slotNext].sCount ? true : SumCtg(slotNext, 1) - cell1 > 0.9;
  }


  /**
    @brief Outputs sample and index counts at a given slot.

    @param liveIdx is a cached offset for the pair.

    @param pos is the position to dereference in the rank vector.

    @param count outputs the sample count.

    @return total index count subsumed, with reference accumulator.
  */
  inline unsigned int LHCounts(int slot, unsigned int &sCount) {
    FRNode *fRun = &runZero[slot];
    sCount = fRun->sCount;
    return  1 + fRun->end - fRun->start;
  }


  inline int RunsLH() {
    return runsLH;
  }

  unsigned int LHBits(unsigned int lhBits, unsigned int &lhSampCt);
  unsigned int LHSlots(int outPos, unsigned int &lhSampCt);
  unsigned int Bounds(int outSlot, int &start, int &end);
};


class Run {
  static unsigned int nPred;
  static unsigned int ctgWidth;
  int splitNext; // Cached for next level.
  int splitCount; // Cached for current level.
  int runSetCount;
  unsigned int *runLength;
  unsigned int *lengthNext; // Upcoming level's run lengths.
  RunSet *runSet;
  FRNode *facRun; // Workspace for FRNodes used along level.
  BHPair *bHeap;
  int *lhOut; // Vector of lh-bound slot indices.
  double *rvWide;
  double *ctgSum;

  void ResetRuns();

  int inline PairOffset(int splitCount, int splitIdx, int predIdx) {
    return splitCount * predIdx + splitIdx;
  }

 public:
  Run();
  ~Run();
  static void Immutables(unsigned int _nPred, unsigned int _ctgWidth);
  static void DeImmutables();

  /**
     @brief Reads specified run bit in specified bit vector.

     @param splitIdx is the split index.

     @param predIdx is the predictor index.

     @return whether specified run bit is set.
   */
  bool inline Singleton(int splitCount, int splitIdx, int predIdx) {
    return runLength[PairOffset(splitCount, splitIdx, predIdx)] == 1;
  }

  void LevelInit(int _splitCount);
  void LevelClear();
  void OffsetsReg();
  void OffsetsCtg();


  inline RunSet *RSet(int rsIdx) {
    return &runSet[rsIdx];
  }

  
  inline unsigned int RunBounds(int idx, int outSlot, int &start, int &end) {
    return runSet[idx].Bounds(outSlot, start, end);
  }


  inline int RunsLH(int rsIdx) {
    return runSet[rsIdx].RunsLH();
  }

  
  void RunSets(int _runSetCount);

  /**
     @brief References run length in current level.
   */
  inline unsigned int &RunLength(int splitIdx, int predIdx) {
    return runLength[PairOffset(splitCount, splitIdx, predIdx)];
  }


  void LengthVec(int _splitNext);
  void LengthTransmit(int splitIdx, int lNext, int rNext);

  /**
     @brief References run length in next level.
   */
  inline unsigned int &LengthNext(int splitIdx, int predIdx) {
    return lengthNext[PairOffset(splitNext, splitIdx, predIdx)];
  }

  /**
     @brief Presets runCount field to a conservative value for
     the purpose of allocating storage.
   */
  unsigned int &CountSafe(int idx) {
    return runSet[idx].safeRunCount;
  }

};


/**
   @brief Implementation of binary heap tailored to RunSets.  Not
   so much a class as a collection of static methods.

   TODO:  Templatize and move elsewhere.
*/

class BHeap {
 public:
  static inline int Parent(int idx) { 
    return (idx-1) >> 1;
  };
  static void Depopulate(BHPair pairVec[], int lhOut[], unsigned int pop);
  static void Insert(BHPair pairVec[], unsigned int _slot, double _key);
  static unsigned int SlotPop(BHPair pairVec[], int bot);
};

#endif
