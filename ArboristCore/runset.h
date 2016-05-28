// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file runset.h

   @brief Definitions for the Run classes, which maintain predictor
   runs, especially factor-valued predictors.

   @author Mark Seligman

 */

// FRNodes hold field values accumulated from runs of factors having the
// same value.  That is, they group factor-valued predictors into block
// representations. These values live for a single level, so must be consumed
// before a new level is started.
//
#ifndef ARBORIST_RUNSET_H
#define ARBORIST_RUNSET_H

#include <vector>


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
  unsigned int *outZero; // Final LH and/or output for heap-ordered slots.
  double *ctgZero; // Categorical:  run x ctg checkerboard.
  double *rvZero; // Non-binary wide runs:  random variates for sampling.
  unsigned int runCount;  // Current high watermark:  not subject to shrinking.
  int runsLH; // Count of LH runs.
 public:
  const static unsigned int maxWidth = 10;
  static unsigned int ctgWidth;
  unsigned int safeRunCount;
  unsigned int DeWide();
  void DePop(unsigned int pop = 0);
  void Reset(FRNode*, BHPair*, unsigned int*, double*, double*);
  void OffsetCache(unsigned int _runOff, unsigned int _heapOff, unsigned int _outOff);
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
  inline double SumHeap(unsigned int outPos, unsigned int &sCount) {
    unsigned int slot = outZero[outPos];
    sCount = runZero[slot].sCount;
    
    return runZero[slot].sum;
  }

  
  /**
     @brief Sets run parameters and increments run count.

     @return void.
   */
  void Write(unsigned int rank, unsigned int sCount, double sum, unsigned int start, unsigned int end) {
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
  inline double SumCtg(unsigned int slot, unsigned int yCtg) {
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
  inline bool SumBinary(unsigned int outPos, double &cell0, double &cell1) {
    unsigned int slot = outZero[outPos];
    cell0 = SumCtg(slot, 0);
    cell1 = SumCtg(slot, 1);

    unsigned int sCount = runZero[slot].sCount;
    unsigned int slotNext = outZero[outPos+1];
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
  inline unsigned int LHCounts(unsigned int slot, unsigned int &sCount) {
    FRNode *fRun = &runZero[slot];
    sCount = fRun->sCount;
    return  1 + fRun->end - fRun->start;
  }


  inline int RunsLH() {
    return runsLH;
  }

  unsigned int LHBits(unsigned int lhBits, unsigned int &lhSampCt);
  unsigned int LHSlots(int outPos, unsigned int &lhSampCt);
  unsigned int Bounds(unsigned int outSlot, unsigned int &start, unsigned int &end);
};


class Run {
  unsigned int setCount;
  RunSet *runSet;
  FRNode *facRun; // Workspace for FRNodes used along level.
  BHPair *bHeap;
  unsigned int *lhOut; // Vector of lh-bound slot indices.
  double *rvWide;
  double *ctgSum;

  void ResetRuns();

 public:
  const unsigned int ctgWidth;
  Run(unsigned int _ctgWidth);

  void LevelClear();
  void OffsetsReg();
  void OffsetsCtg();

  inline RunSet *RSet(unsigned int rsIdx) {
    return &runSet[rsIdx];
  }
  
  inline unsigned int RunBounds(unsigned int idx, unsigned int outSlot, unsigned int &start, unsigned int &end) {
    return runSet[idx].Bounds(outSlot, start, end);
  }


  inline unsigned int RunsLH(unsigned int rsIdx) {
    return runSet[rsIdx].RunsLH();
  }

  
  void RunSets(const std::vector<unsigned int> &safeCount);

  /**
     @brief Presets runCount field to a conservative value for
     the purpose of allocating storage.
   */
  unsigned int &CountSafe(unsigned int idx) {
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
  static void Depopulate(BHPair pairVec[], unsigned int lhOut[], unsigned int pop);
  static void Insert(BHPair pairVec[], unsigned int _slot, double _key);
  static unsigned int SlotPop(BHPair pairVec[], int bot);
};

#endif
