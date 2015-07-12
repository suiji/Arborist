// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file run.cc

   @brief Methods for maintaining runs of factor-valued predictors during splitting.

   @author Mark Seligman
 */

#include "run.h"
#include "callback.h"
#include "predictor.h"

// Testing only:
//#include <iostream>
using namespace std;

/**
   Run objects are allocated per-tree, and live throughout training.

   RunSets live only during a single level, from argmax pass one (splitting)
   through argmax pass two.  They accumulate summary information for split/
   predictor pairs anticipated to have two or more distinct runs.  RunSets
   are not built for numerical predictors, which are assumed generally to
   have dispersive values.

   The runLength[] vector tracks conservatively-estimated run lengths for
   every split/predictor pair, regardless whether the pair is chosen for
   splitting in a given level (cf., 'mtry' and 'predProb').  The vector
   must be reallocated at each level, to accommodate changes in node numbering
   introduced through splitting.  "Fat ranks", however, which track the
   dense components of sparse predictors, employ a different type of
   mechanism to track runs.

   Run lengths for a given predictor decrease, although not necessarily
   monotonically, with splitting.  Hence once a pair becomes a singleton, the
   fact is worth preserving for the duration of training.  Numerical predictors
   are assigned a nonsensical run length of zero, which is changed to a sticky
   value of unity, should a singleton be identified.  Run lengths are
   transmitted between levels duing restaging, which is the only phase to
   maintain a map between split nodes and their descendants.  Similarly, new
   singletons are very easy to identify during restaging.

   Other than the "bottom" value of unity, run lengths can generally only be
   known precisely by first walking the predictor ranks.  Hence a conservative
   value is used for storage allocation, namely, that obtained during a previous
   level.  Note that this value may be quite conservative, as the pair may not
   have undergone a rank-walk in the previous level.  The one exception to this
   is the case of an argmax split, for which both left and right run counts are
   known from splitting.
*/

unsigned int Run::nPred = 0;
unsigned int RunSet::ctgWidth = 0;

/**
   @brief Constructor initializes predictor run length either to cardinality, 
   for factors, or to a nonsensical zero, for numerical.
 */
Run::Run() {
  runLength = 0;
  lengthNext = new unsigned int[nPred];
  for (unsigned int i = 0; i < nPred; i++) {
    lengthNext[i] = Predictor::FacCard(i);
  }
  runSet = 0;
  facRun = 0;
  bHeap = 0;
  lhOut = 0;
  rvWide = 0;
  ctgSum = 0;
}

/**
   @brief Moves pre-computed split count and run-length information to
   current level.

   @return void.
 */
void Run::LevelInit(int _splitCount) {
  splitCount = _splitCount;
  runLength = lengthNext;
  lengthNext = 0;
}


void Run::RunSets(int _runSetCount) {
  runSetCount = _runSetCount;
  if (runSetCount > 0)
    runSet = new RunSet[runSetCount];
}


/**
   @brief Presets runCount field to a conservative value for
   the purpose of allocating storage.
 */
void Run::SafeRunCount(int idx, int count) {
  runSet[idx].RunCount() = count;
}



/**
   @brief Regression:  all runs employ a heap.

   @return void.
 */
void Run::OffsetsReg() {
  if (runSet == 0)
    return;

  int runCount = 0;
  for (int i = 0; i < runSetCount; i++) {
    runSet[i].OffsetCache(runCount, runCount, runCount);
    runCount += runSet[i].RunCount();
  }

  facRun = new FRNode[runCount];
  bHeap = new BHPair[runCount];
  lhOut = new int[runCount];

  ResetRuns();
}


/**
   @brief Classification:  only wide run sets use the heap.

   @return void.

*/
void Run::OffsetsCtg() {
  if (runSet == 0)
    return;

  // Running counts:
  int runCount = 0; // Factor runs.
  int wideRuns = 0; // Runs in wide factors.
  int outRuns = 0; // Heap output slots.
  for (int i = 0; i < runSetCount; i++) {
    int rCount = runSet[i].RunCount();
    if (rCount > RunSet::maxWidth) { // Only wide runs use heap.
      runSet[i].OffsetCache(runCount, wideRuns, outRuns);
      wideRuns += rCount;
      outRuns += RunSet::ctgWidth == 2 ? rCount : RunSet::maxWidth;
    }
    else {
      runSet[i].OffsetCache(runCount, 0, outRuns);
      outRuns += rCount;
    }
    runCount += rCount;
  }

  int boardWidth = runCount * RunSet::ctgWidth; // Checkerboard.
  ctgSum = new double[boardWidth];
  for (int i = 0; i < boardWidth; i++)
    ctgSum[i] = 0.0;

  if (RunSet::ctgWidth > 2) { // Non-binary samples wide w.o. replacement.
    if (wideRuns > 0) {
      rvWide = new double[wideRuns];
      CallBack::RUnif(wideRuns, rvWide);
    }
  }

  facRun = new FRNode[runCount];
  bHeap = new BHPair[wideRuns];
  lhOut = new int[outRuns];

  ResetRuns();
}


/**
   @brief Adjusts offset and run-count fields of each RunSet.

   @return void.
 */
void Run::ResetRuns() {
  for (int i = 0; i < runSetCount; i++) {
    runSet[i].Reset(facRun, bHeap, lhOut, ctgSum, rvWide);
  }
}


void Run::LevelClear() {
  delete [] runLength;
  runLength = 0;

  if (runSetCount > 0) {
    delete [] runSet;
    delete [] facRun;
    delete [] lhOut;
    if (rvWide != 0)
      delete [] rvWide;
    if (ctgSum != 0)
      delete [] ctgSum;
    if (bHeap != 0)
      delete [] bHeap;
    runSet = 0;
    facRun = 0;
    lhOut = 0;
    rvWide = 0;
    ctgSum = 0;
    bHeap = 0;
    runSetCount = 0;
  }
}


Run::~Run() {
  delete [] lengthNext; // One will always be dangling.
}



/**
 */
void Run::LengthVec(int _splitNext) {
  splitNext = _splitNext;
  lengthNext = new unsigned int[splitNext * nPred];
}


/**
   @brief Transmits next level's lh/rh indices, as needed.  Singletons must
   be transmitted, to avoid referencing dirty fields during splitting.  Non-
   singleton runs are usefully transmitted, in order to set conservative
   bounds on memory allocation.

   @param splitIdx is the current level's split index.

   @param lNext is the split index of the left successor in the next level.

   @param rNext is the split index of the right successor in the next level

   @return void.
*/
void Run::LengthTransmit(int splitIdx, int lNext, int rNext) {
  if (lNext >= 0) {
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      unsigned int rCount = RunLength(splitIdx, predIdx);
      LengthNext(lNext, predIdx) = rCount;
    }
  }
  if (rNext >= 0) {
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      unsigned int rCount = RunLength(splitIdx, predIdx);
      LengthNext(rNext, predIdx) = rCount;
    }
  }
}
  

/**
   @brief Invokes base class factory and lights off class specific initializations.

   @param _nPred is the number of predictors.

   @param _ctgWidth is the response cardinality.

   @return void.
 */
void Run::Immutables(int _nPred, int _ctgWidth) {
  nPred = _nPred;
  RunSet::ctgWidth = _ctgWidth;
}


/**
  @brief Regression variant does not set 'ctgWidth'.

  @return void.
 */
void Run::Immutables(int _nPred) {
  nPred = _nPred;
}


/**
   @brief Restoration of class immutables to static default values.

   @return void.
 */
void Run::DeImmutables() {
  nPred = RunSet::ctgWidth = 0;
}


/**
   @brief Records only the (casted) relative vector offsets, as absolute
   base addresses not yet known.
 */
void RunSet::OffsetCache(int _runOff, int _heapOff, int _outOff) {
  runOff = _runOff;
  heapOff = _heapOff;
  outOff = _outOff;
}


/**
   @brief Updates relative vector addresses with their respective base
   addresses, now known.
 */
void RunSet::Reset(FRNode *runBase, BHPair *heapBase, int *outBase, double *ctgBase, double *rvBase) {
  runZero = runBase + runOff;
  heapZero = heapBase + heapOff;
  outZero = outBase + outOff;
  rvZero = rvBase + heapOff;
  ctgZero = ctgBase + (runOff * ctgWidth);
  runCount = 0;
}


/**
   @brief Writes to heap using appropriate weighting for response.

   @return void.
 */
void RunSet::WriteHeap() {
  // Non-binary classification weights arbitrarily:  sampling w/o replacement.
  if (ctgWidth > 2) {
    for (int i = 0; i < runCount; i++) {
      BHeap::Insert(heapZero, i, rvZero[i]);
    }
  }
  else { // Regression, binary weight by mean (== proportion at unity, for binary).
    for (int i = 0; i < runCount; i++) {
      BHeap::Insert(heapZero, i, runZero[i].sum / runZero[i].sCount);
    }
  }
}


/**
   @brief Depopulates the heap associated with a pair and places sorted ranks into rank vector.

   @param pop is the number of elements to pop from the heap.

   @return void
*/
void RunSet::DePop(int pop) {
  return BHeap::Depopulate(heapZero, outZero, pop == 0 ? runCount : pop);
}


/**
   @brief Hammers the pair's run contents with runs selected for
   sampling.  Since the runs are to be read numerous times, performance
   may be benefit from this elimination of a level of indirection.
 */
void RunSet::Shrink() {
  FRNode tempRun[maxWidth];

  // Copies runs referenced by the slot list to a temporary area.
  DePop(maxWidth);
  for (int i = 0; i < maxWidth; i++) {
    tempRun[i] = runZero[outZero[i]];
  }

  // Overwrites existing runs with the shrunken list
  for (int i = 0; i < maxWidth; i++) {
    runZero[i] = tempRun[i];
  }
}


/**
   @brief Decodes bit vector of slot indices and stores LH indices.

   @param lhBits encodes LH/RH slot indices as on/off bits, respectively.

   @param lhSampCt outputs the LHS sample count.

   @return LHS index count.
*/
int RunSet::LHBits(unsigned int lhBits, int &lhSampCt) {
  int lhIdxCount = 0;
  unsigned int slotSup = EffCount() - 1;
  runsLH = 0;
  lhSampCt = 0;
  if (lhBits != 0) {
    for (unsigned int slot = 0; slot < slotSup; slot++) {
      // If bit # 'slot' set in 'argMax', then the run at index
      // 'slot' belongs to the left-hand side of the split.  Its
      // sample and index counts are accumulated and its index
      // is recorded in the out-set.
      //
      if ((lhBits & (1 << slot)) != 0) {
        LHAccum(slot, lhSampCt, lhIdxCount);
	outZero[runsLH++] = slot;
      }
    }
  }

  return lhIdxCount;
}


/**
   @brief Dereferences out slots and accumulates splitting parameters.

   @param lhTop is the final out slot of the LHS:  < 0 iff no split.

   @param lhSampCt outputs the LHS sample count.

   @return LHS index count.
*/
int RunSet::LHSlots(int lhTop, int &lhSampCt) {
  int lhIdxCount = 0;
  lhSampCt = 0;

  runsLH = lhTop + 1;
  for (int outSlot = 0; outSlot < runsLH; outSlot++) {
    LHAccum(outZero[outSlot], lhSampCt, lhIdxCount);
  }

  return lhIdxCount;  
}


// TODO:  Replace with templated versions and place in separate module.
//

/**
   @brief Inserts a key, value pair into the heap at next vacant slot.

   @param bhOffset is the cached pair coordinate.

   @param _slot is the slot position.

   @param _key is the associated key.

   @return void.
 */
void BHeap::Insert(BHPair pairVec[], int _slot, double _key) {
  unsigned int idx = _slot;
  pairVec[idx].key = _key;
  pairVec[idx].slot = _slot;

  int parIdx = Parent(idx);
  while (parIdx >= 0 && pairVec[parIdx].key > _key) {
    pairVec[idx].key = pairVec[parIdx].key;
    pairVec[idx].slot = pairVec[parIdx].slot;
    pairVec[parIdx].slot = _slot;
    pairVec[parIdx].key = _key;
    idx = parIdx;
    parIdx = Parent(idx);
  }
}


/**
   @brief Empties the slot indices keyed in BHPairs, in weight-sorted order.

   @param pop is the number of elements to pop.

   @param lhOut outputs the popped items, in order.

   @return void.
*/
void BHeap::Depopulate(BHPair pairVec[], int lhOut[], unsigned int pop) {
  for (int bot = pop - 1; bot >= 0; bot--) {
    lhOut[pop - 1 - bot] = pairVec[0].slot;

    // Places bottom element at head and refiles.
    unsigned int idx = 0;
    int slotRefile = pairVec[idx].slot = pairVec[bot].slot;
    double keyRefile = pairVec[idx].key = pairVec[bot].key;
    int descL = 1;
    int descR = 2;

    // 'descR' remains the lower of the two descendant indices.  Some short-circuiting below.
    //
    while((descR <= bot && keyRefile > pairVec[descR].key) || (descL <= bot && keyRefile > pairVec[descL].key)) {
      int chIdx =  (descR <= bot && pairVec[descR].key < pairVec[descL].key) ?  descR : descL;
      pairVec[idx].key = pairVec[chIdx].key;
      pairVec[idx].slot = pairVec[chIdx].slot;
      pairVec[chIdx].key = keyRefile;
      pairVec[chIdx].slot = slotRefile;
      idx = chIdx;
      descL = 1 + (idx << 1);
      descR = (1 + idx) << 1;
    }
  }
}
