// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file runset.cc

   @brief Methods for maintaining runs of factor-valued predictors during splitting.

   @author Mark Seligman
 */

#include "runset.h"
#include "callback.h"

// Testing only:
//#include <iostream>
//using namespace std;

unsigned int RunSet::ctgWidth = 0;
unsigned int RunSet::noStart = 0;


/**
   Run objects are allocated per-tree, and live throughout training.

   RunSets live only during a single level, from argmax pass one (splitting)
   through argmax pass two.  They accumulate summary information for split/
   predictor pairs anticipated to have two or more distinct runs.  RunSets
   are not yet built for numerical predictors, which have so far been
   generally assumed to have dispersive values.

   The runCounts[] vector tracks conservatively-estimated run lengths for
   every split/predictor pair, regardless whether the pair is chosen for
   splitting in a given level (cf., 'mtry' and 'predProb').  The vector
   must be reallocated at each level, to accommodate changes in node numbering
   introduced through splitting.

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

/**
   @brief Constructor initializes predictor run length either to cardinality, 
   for factors, or to a nonsensical zero, for numerical.
 */
Run::Run(unsigned int _ctgWidth, unsigned int nRow, unsigned int bagCount) : noRun(nRow * bagCount), ctgWidth(_ctgWidth) {
  RunSet::ctgWidth = ctgWidth;
  RunSet::noStart = nRow; // Inattainable start value, irrespective of tree.
  runSet = 0;
  facRun = 0;
  bHeap = 0;
  lhOut = 0;
  rvWide = 0;
  ctgSum = 0;
}


/**
   @brief Initializes the run counts to conservative values.

   @param safeCount is a vector of run counts.

   @return void.
 */
void Run::RunSets(const std::vector<unsigned int> &safeCount) {
  setCount = safeCount.size();
  if (setCount > 0) {
    runSet = new RunSet[setCount];
    for (unsigned int setIdx = 0; setIdx < setCount; setIdx++) {
      CountSafe(setIdx, safeCount[setIdx]);
    }
  }
}


/**
   @brief Regression:  all runs employ a heap.

   @return void.
 */
void Run::OffsetsReg() {
  if (setCount == 0)
    return;

  unsigned int runCount = 0;
  for (unsigned int i = 0; i < setCount; i++) {
    runSet[i].OffsetCache(runCount, runCount, runCount);
    runCount += runSet[i].CountSafe();
  }

  facRun = new FRNode[runCount];
  bHeap = new BHPair[runCount];
  lhOut = new unsigned int[runCount];

  ResetRuns();
}


/**
   @brief Classification:  only wide run sets use the heap.

   @return void.

*/
void Run::OffsetsCtg() {
  if (setCount == 0)
    return;

  // Running counts:
  unsigned int runCount = 0; // Factor runs.
  unsigned int heapRuns = 0; // Runs subject to sorting.
  unsigned int outRuns = 0; // Sorted runs of interest.
  for (unsigned int i = 0; i < setCount; i++) {
    unsigned int rCount = runSet[i].CountSafe();
    if (ctgWidth == 2) { // Binary uses heap for all runs.
      runSet[i].OffsetCache(runCount, heapRuns, outRuns);
      heapRuns += rCount;
      outRuns += rCount;
    }
    else if (rCount > RunSet::maxWidth) {
      runSet[i].OffsetCache(runCount, heapRuns, outRuns);
      heapRuns += rCount;
      outRuns += RunSet::maxWidth;
    }
    else {
      runSet[i].OffsetCache(runCount, 0, outRuns);
      outRuns += rCount;
    }
    runCount += rCount;
  }

  unsigned int boardWidth = runCount * ctgWidth; // Checkerboard.
  ctgSum = new double[boardWidth];
  for (unsigned int i = 0; i < boardWidth; i++)
    ctgSum[i] = 0.0;

  if (ctgWidth > 2 && heapRuns > 0) { // Wide non-binary:  w.o. replacement.
    rvWide = new double[heapRuns];
    CallBack::RUnif(heapRuns, rvWide);
  }

  facRun = new FRNode[runCount];
  bHeap = new BHPair[heapRuns];
  lhOut = new unsigned int[outRuns];

  ResetRuns();
}


/**
   @brief Adjusts offset and run-count fields of each RunSet.

   @return void.
 */
void Run::ResetRuns() {
  for (unsigned int i = 0; i < setCount; i++) {
    runSet[i].Reset(facRun, bHeap, lhOut, ctgSum, rvWide);
  }
}


void Run::LevelClear() {
  if (setCount > 0) {
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
    setCount = 0;
  }
}


/**
   @brief Records only the (casted) relative vector offsets, as absolute
   base addresses not yet known.
 */
void RunSet::OffsetCache(unsigned int _runOff, unsigned int _heapOff, unsigned int _outOff) {
  runOff = _runOff;
  heapOff = _heapOff;
  outOff = _outOff;
}


/**
   @brief Updates relative vector addresses with their respective base
   addresses, now known.
 */
void RunSet::Reset(FRNode *runBase, BHPair *heapBase, unsigned int *outBase, double *ctgBase, double *rvBase) {
  runZero = runBase + runOff;
  heapZero = heapBase + heapOff;
  outZero = outBase + outOff;
  rvZero = rvBase + heapOff;
  ctgZero = ctgBase + (runOff * ctgWidth);
  runCount = 0;
}


/**
   @brief Writes to heap arbitrarily:  sampling w/o replacement.

   @return void.
 */
void RunSet::HeapRandom() {
  for (unsigned int slot = 0; slot < runCount; slot++) {
    BHeap::Insert(heapZero, slot, rvZero[slot]);
  }
}


/**
   @brief Writes to heap, weighting by slot mean response.

   @return void.
 */
void RunSet::HeapMean() {
  for (unsigned int slot = 0; slot < runCount; slot++) {
    BHeap::Insert(heapZero, slot, runZero[slot].sum / runZero[slot].sCount);
  }
}


/**
   @brief Writes to heap, weighting by category-1 probability.

   @return void.
 */
void RunSet::HeapBinary() {
  // Ordering by category probability is equivalent to ordering by
  // concentration, as weighting by priors does not affect oder.
  //
  // In the absence of class weighting, numerator can be (integer) slot
  // sample count, instead of slot sum.
  for (unsigned int slot = 0; slot < runCount; slot++) {
    BHeap::Insert(heapZero, slot, SumCtg(slot, 1) / runZero[slot].sum);
  }
}


/**
   @brief Builds a run for the dense rank using residual values.

   @param denseRank is the rank corresponding to the dense factor.

   @param sCountTot is the total sample count over the node.

   @param sumTot is the total sum of responses over the node.

   @return void.
 */
void RunSet::WriteImplicit(unsigned int denseRank, unsigned int sCountTot, double sumTot, unsigned int denseCount, const double nodeSum[]) {
  if (nodeSum != 0) {
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      SumCtgSet(ctg, nodeSum[ctg]);
    }
  }

  for (unsigned int runIdx = 0; runIdx < runCount; runIdx++) {
    sCountTot -= runZero[runIdx].sCount;
    sumTot -= runZero[runIdx].sum;
    if (nodeSum != 0) {
      for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
	AccumCtg(ctg, -SumCtg(runIdx, ctg));
      }
    }
  }

  Write(denseRank, sCountTot, sumTot, denseCount);
}


/**
   @brief Implicit runs are characterized by a start value of 'noStart'.

   @return Whether this run is dense.
 */
bool FRNode::IsImplicit() {
  return start == RunSet::noStart;
}


/**
   @brief Determines whether it is necessary to expose the right-hand
   runs.

   Right-hand runs can often be omitted from consideration by
   presetting a split's next-level contents all to the right-hand
   index, then overwriting those known to lie in the left split.  The
   left indices are always exposed, making this a convenient strategy.

   This cannot be done if the left contains an implicit run, as implicit
   run indices are not directly recorded.  In such cases a complementary
   strategy is employed, in which all indices are preset to the left
   index, with known right-hand indices overwritten.  Hence the
   right-hand runs must be enumerated in such instances.

   @return true iff right-hand runs must be exposed.
 */
bool RunSet::ImplicitLeft() {
  if (!hasImplicit)
    return false;

  for (unsigned int runIdx = 0; runIdx < runsLH; runIdx++) {
    unsigned int outSlot = outZero[runIdx];
    if (runZero[outSlot].IsImplicit()) {
      return true;
    }
  }

  return false;
}


/**
   @brief Depopulates the heap associated with a pair and places sorted ranks into rank vector.

   @param pop is the number of elements to pop from the heap.

   @return void
*/
void RunSet::DePop(unsigned int pop) {
  return BHeap::Depopulate(heapZero, outZero, pop == 0 ? runCount : pop);
}


/**
   @brief Hammers the pair's run contents with runs selected for
   sampling.  Since the runs are to be read numerous times, performance
   may be benefit from this elimination of a level of indirection.


   @return post-shrink run count.
 */
unsigned int RunSet::DeWide() {
  if (runCount <= maxWidth)
    return runCount;

  HeapRandom();
  FRNode tempRun[maxWidth];
  double *tempSum = new double[ctgWidth * maxWidth];
  // Copies runs referenced by the slot list to a temporary area.
  DePop(maxWidth);
  for (unsigned int i = 0; i < maxWidth; i++) {
    unsigned int outSlot = outZero[i];
    tempRun[i] = runZero[outSlot];
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      tempSum[i * ctgWidth + ctg] = ctgZero[outSlot * ctgWidth + ctg];
    }
  }

  // Overwrites existing runs with the shrunken list
  for (unsigned int i = 0; i < maxWidth; i++) {
    runZero[i] = tempRun[i];
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      ctgZero[i * ctgWidth + ctg] = tempSum[i * ctgWidth + ctg];
    }
  }
  delete [] tempSum;

  return maxWidth;
}


/**
   @brief Decodes bit vector of slot indices and stores LH indices.

   @param lhBits encodes LH/RH slot indices as on/off bits, respectively.

   @param lhSampCt outputs the LHS sample count.

   @return LHS index count.
*/
unsigned int RunSet::LHBits(unsigned int lhBits, unsigned int &lhSampCt) {
  unsigned int lhExtent = 0;
  unsigned int slotSup = EffCount() - 1;
  runsLH = 0;
  lhSampCt = 0;
  if (lhBits != 0) {
    for (unsigned int slot = 0; slot < slotSup; slot++) {
      // If bit # 'slot' set in 'lhBits', then the run at index
      // 'slot' belongs to the left-hand side of the split.  Its
      // sample and index counts are accumulated and its index
      // is recorded in the out-set.
      //
      if ((lhBits & (1 << slot)) != 0) {
	unsigned int sCount;
        lhExtent += LHCounts(slot, sCount);
	lhSampCt += sCount;
	outZero[runsLH++] = slot;
      }
    }
  }

  if (ImplicitLeft()) {
    unsigned int rhIdx = runsLH;
    for (unsigned int slot = 0; slot < EffCount(); slot++) {
      if ((lhBits & (1 << slot)) == 0) {
        outZero[rhIdx++] = slot;
      }
    }
  }

  return lhExtent;
}


/**
   @brief Dereferences out slots and accumulates splitting parameters.

   @param cut is the final out slot of the LHS:  < 0 iff no split.

   @param lhSampCt outputs the LHS sample count.

   @return LHS index count.
*/
unsigned int RunSet::LHSlots(int cut, unsigned int &lhSampCt) {
  unsigned int lhExtent = 0;
  lhSampCt = 0;

  for (int outSlot = 0; outSlot <= cut; outSlot++) {
    unsigned int sCount;
    lhExtent += LHCounts(outZero[outSlot], sCount);
    lhSampCt += sCount;
  }

  runsLH = cut + 1;
  return lhExtent;  
}


// TODO:  Replace with templated versions and place in separate module.
//

/**
   @brief Inserts a key, value pair into the heap at next vacant slot.  Heap
   updates to move element with maximal key to the top.

   @param bhOffset is the cached pair coordinate.

   @param _slot is the slot position.

   @param _key is the associated key.

   @return void.
 */
void BHeap::Insert(BHPair pairVec[], unsigned int _slot, double _key) {
  unsigned int idx = _slot;
  BHPair input;
  input.key = _key;
  input.slot = _slot;
  pairVec[idx] = input;

  int parIdx = Parent(idx);
  while (parIdx >= 0 && pairVec[parIdx].key > _key) {
    pairVec[idx] = pairVec[parIdx];
    pairVec[parIdx] = input;
    idx = parIdx;
    parIdx = Parent(idx);
  }
}


/**
   @brief Empties the slot indices keyed in BHPairs.

   @param pop is the number of elements to pop.  Caller enforces value > 0.

   @param lhOut outputs the popped slots, in increasing order.

   @return void.
*/
void BHeap::Depopulate(BHPair pairVec[], unsigned int lhOut[], unsigned int pop) {
  for (int bot = pop - 1; bot >= 0; bot--) {
    lhOut[pop - (1 + bot)] = SlotPop(pairVec, bot);
  }
}


/**
   @brief Pops value at bottom of heap.

   @return popped value.
 */
unsigned int BHeap::SlotPop(BHPair pairVec[], int bot) {
  unsigned int ret = pairVec[0].slot;
  if (bot == 0)
    return ret;
  
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

  return ret;
}


/**
     @brief Looks up run parameters by indirection through output vector.
     N.B.:  should not be called with a dense run.

     @param start outputs starting index of run.

     @param end outputs ending index of run.

     @return rank of referenced run, plus output reference parameters.
*/
void RunSet::Bounds(unsigned int outSlot, unsigned int &start, unsigned int &extent) const {
  unsigned int slot = outZero[outSlot];
  runZero[slot].ReplayRef(start, extent);
}


unsigned int RunSet::Rank(unsigned int outSlot) const {
  unsigned int slot = outZero[outSlot];
  return runZero[slot].Rank();
}
