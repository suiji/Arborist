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
  int sCount; // Sample count of factor run:  not always same as length.
  double sum; // Sum of responses associated with run.

  FRNode() : start(-1), end(-1), sCount(-1), sum(0.0) {}
  /**
     @brief Bounds accessor.
   */
  inline void ReplayFields(int &_start, int &_end) {
    _start = start;
    _end = end;
  }

  inline void Set(unsigned int _rank, int _sCount, double _sum, int _start, int _end) {
    rank = _rank;
    sCount = _sCount;
    sum = _sum;
    start = _start;
    end = _end;
  }
};

typedef struct { double key; int slot; } BHPair;

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
  int runCount;  // Current high watermark:  not subject to shrinking.
  int runsLH; // Count of LH runs.
 public:
  const static int maxWidth = 10;
  static unsigned int ctgWidth;
  void Shrink();
  void DePop(int pop = 0);
  void Reset(FRNode*, BHPair*, int*, double*, double*);
  void OffsetCache(int runIdx, int bhpIdx, int outIdx);
  void WriteHeap();

  /**
     @brief Characterizes wide run counts.

     @return whether run set is wide.
   */
  inline bool IsWide() {
    return runCount > maxWidth;
  }


  /**
     @brief Accessor for runCount field.

     @return reference to run count.
   */
  inline int &RunCount() {
    return runCount;
  }
  
  
  /**
     @brief "Effective" run count, for the sake of splitting, is the lesser
     of the true run count and 'maxWidth'.

     @return effective run count
   */
  inline int EffCount() {
    return IsWide() ? maxWidth : runCount;
  }


  /**
     @brief Looks up sum and sample count associated with a given output slot.

     @param outPos is a position in the output vector.

     @param sCount outputs the sample count at the dereferenced output slot.

     @return sum at dereferenced position, with output reference parameter.
   */
  inline double SumHeap(int outPos, int &sCount) {
    int slot = outZero[outPos];
    sCount = runZero[slot].sCount;
    
    return runZero[slot].sum;
  }

  /**
     @brief Sets run parameters and increments run count.

     @return void.
   */
  void Write(unsigned int rank, int sCount, double sum, int start, int end) {
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
  inline double SumCtg(unsigned int yCtg, int slot) {
    return ctgZero[slot * ctgWidth + yCtg];
  }


  /**
     @brief Looks up the two binary response sums associated with a given
     output slot.

     @param outPos is a position in the output vector.

     @param cell0 outputs the response at rank 0.

     @param cell1 outputs the response at rank 1.

     @return void, with output reference parameters.
   */
  inline void SumBinary(int outPos, double &cell0, double &cell1) {
    int slot = outZero[outPos];
    cell0 = SumCtg(0, slot);
    cell1 = SumCtg(1, slot);
  }


  /**
    @brief Accumulates sample and index counts in an order specified by caller.

    @param liveIdx is a cached offset for the pair.

    @param pos is the position to dereference in the rank vector.

    @param count accumulates sample counts.

    @param length accumulates index counts.

    @return void, with output reference parameters.
  */
  inline void LHAccum(int slot, int &count, int &length) {
    FRNode *fRun = &runZero[slot];
    count += fRun->sCount;
    length += 1 + fRun->end - fRun->start;
  }


  /**
     @brief Dereferences the slot at a specified output position.

     @param outPos is a position in the output vector.

     @return Reference to output position.
   */
  inline int &OrdSlot(int outpos) {
    return outZero[outpos];
  }

  

  /**
     @brief Looks up run parameters by indirection through output vector.

     @param start outputs starting index of run.

     @param end outputs ending index of run.

     @return rank of referenced run, plus output reference parameters.
  */
  inline unsigned int Bounds(int outSlot, int &start, int &end) {
    int slot = outZero[outSlot];
    FRNode fRun = runZero[slot];
    start = fRun.start;
    end = fRun.end;
    return fRun.rank;
  }

  inline int RunsLH() {
    return runsLH;
  }

  int LHBits(unsigned int lhBits, int &lhSampCt);
  int LHSlots(int outPos, int &lhSampCt);
};


class Run {
  static unsigned int nPred;
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
  
  /**
     @brief Reads specified run bit in specified bit vector.

     @param splitIdx is the split index.

     @param predIdx is the predictor index.

     @return whether specified run bit is set.
   */
  bool inline Singleton(int splitCount, int splitIdx, int predIdx) {
    return runLength[PairOffset(splitCount, splitIdx, predIdx)] == 1;
  }

  static void Immutables(int _nPred, int _ctgWidth);
  static void Immutables(int _nPred);
  static void DeImmutables();

  void LevelInit(int _splitCount);
  void LevelClear();
  void SafeRunCount(int idx, int count);
  void OffsetsReg();
  void OffsetsCtg();


  inline RunSet *RSet(int rsIdx) {
    return &runSet[rsIdx];
  }

  
  inline int RunBounds(int idx, int outSlot, int &start, int &end) {
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
  static void Insert(BHPair pairVec[], int _slot, double _key);
};

#endif
