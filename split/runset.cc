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
#include "splitfrontier.h"
#include "splitnux.h"
#include "pretree.h"
#include "frontier.h"


IndexT RunSet::noStart = 0;


Run::Run(unsigned int ctgWidth_,
         unsigned int nRow) :
  setCount(0),
  runSet(vector<RunSet>(0)),
  facRun(vector<FRNode>(0)),
  bHeap(vector<BHPair>(0)),
  lhOut(vector<unsigned int>(0)),
  ctgSum(vector<double>(0)),
  rvWide(vector<double>(0)),
  ctgWidth(ctgWidth_) {
  RunSet::noStart = nRow; // Inattainable start value, irrespective of tree.
}


/**
   @brief Initializes the run counts to conservative values.

   @param safeCount is a vector of run counts.

   @return void.
 */
void Run::runSets(const vector<unsigned int> &safeCount) {
  setCount = safeCount.size();
  runSet = vector<RunSet>(setCount);
  for (unsigned int setIdx = 0; setIdx < setCount; setIdx++) {
    setSafeCount(setIdx, safeCount[setIdx]);
  }
}


void Run::offsetsReg(const vector<unsigned int> &safeCount) {
  runSets(safeCount);
  if (setCount == 0)
    return;

  unsigned int runCount = 0;
  for (auto & rs : runSet) {
    rs.offsetCache(runCount, runCount, runCount);
    runCount += rs.getSafeCount();
  }

  facRun = vector<FRNode>(runCount);
  bHeap = vector<BHPair>(runCount);
  lhOut = vector<unsigned int>(runCount);

  reBase();
}


void Run::offsetsCtg(const vector<unsigned int> &safeCount) {
  runSets(safeCount);
  if (setCount == 0)
    return;

  // Running counts:
  unsigned int runCount = 0; // Factor runs.
  unsigned int heapRuns = 0; // Runs subject to sorting.
  unsigned int outRuns = 0; // Sorted runs of interest.
  for (auto & rs : runSet) {
    unsigned int rCount = rs.getSafeCount();
    if (ctgWidth == 2) { // Binary response uses heap for all runs.
      rs.offsetCache(runCount, heapRuns, outRuns);
      heapRuns += rCount;
      outRuns += rCount;
    }
    else if (rCount > RunSet::maxWidth) {
      rs.offsetCache(runCount, heapRuns, outRuns);
      heapRuns += rCount;
      outRuns += RunSet::maxWidth;
    }
    else {
      rs.offsetCache(runCount, 0, outRuns);
      outRuns += rCount;
    }
    runCount += rCount;
  }

  unsigned int boardWidth = runCount * ctgWidth; // Checkerboard.
  ctgSum = vector<double>(boardWidth);
  fill(ctgSum.begin(), ctgSum.end(), 0.0);

  if (ctgWidth > 2 && heapRuns > 0) { // Wide non-binary:  w.o. replacement.
    rvWide = CallBack::rUnif(heapRuns);
  }

  facRun = vector<FRNode>(runCount);
  bHeap = vector<BHPair>(runCount);
  lhOut = vector<unsigned int>(runCount);

  reBase();
}


void Run::reBase() {
  for (auto & rs  : runSet) {
    rs.reBase(facRun, bHeap, lhOut, ctgSum, ctgWidth, rvWide);
  }
}


double Run::branch(const SplitFrontier* splitFrontier,
                   IndexSet* iSet,
                   PreTree* preTree,
		   Replay* replay,
                   vector<SumCount>& ctgCrit,
                   bool& replayLeft) const {
  return runSet[splitFrontier->getSetIdx(iSet)].branch(iSet, preTree, splitFrontier, replay, ctgCrit, replayLeft);
}


double RunSet::branch(IndexSet* iSet,
                      PreTree* preTree,
                      const SplitFrontier* splitFrontier,
                      Replay* replay,
                      vector<SumCount>& ctgCrit,
                      bool& leftExpl) const {
  double sumExpl = 0.0;
  leftExpl = !implicitLeft(); // true iff left-explicit replay indices.
  for (unsigned int outSlot = 0; outSlot < getRunsLH(); outSlot++) {
    preTree->setLeft(iSet, getRank(outSlot));
    if (leftExpl) {
      sumExpl += splitFrontier->blockReplay(iSet, getBounds(outSlot), true, replay, ctgCrit);
    }
  }

  if (!leftExpl) { // Replay indices explicit on right.
    for (auto outSlot = getRunsLH(); outSlot < getRunCount(); outSlot++) {
      sumExpl += splitFrontier->blockReplay(iSet, getBounds(outSlot), false, replay, ctgCrit);
    }
  }

  return sumExpl;
}


void Run::clear() {
  runSet.clear();
  facRun.clear();
  lhOut.clear();
  bHeap.clear();
  ctgSum.clear();
  rvWide.clear();
}


void RunSet::offsetCache(unsigned int _runOff,
                         unsigned int _heapOff,
                         unsigned int _outOff) {
  runOff = _runOff;
  heapOff = _heapOff;
  outOff = _outOff;
}


// N.B.:  Assumes that nonempty vectors have been allocated with
// a conservative length.
//
void RunSet::reBase(vector<FRNode>& runBase,
                    vector<BHPair>& heapBase,
                    vector<unsigned int>& outBase,
                    vector<double>& ctgBase,
                    unsigned int nCtg,
                    vector<double>& rvBase) {
  runZero = &runBase[runOff];
  heapZero = &heapBase[heapOff];
  outZero = &outBase[outOff];
  rvZero = rvBase.size() > 0 ? &rvBase[heapOff] : nullptr;
  ctgZero = ctgBase.size() > 0 ?  &ctgBase[runOff * nCtg] : nullptr;
  runCount = 0;
}


void RunSet::heapRandom() {
  for (unsigned int slot = 0; slot < runCount; slot++) {
    BHeap::insert(heapZero, slot, rvZero[slot]);
  }
}


void RunSet::heapMean() {
  for (unsigned int slot = 0; slot < runCount; slot++) {
    BHeap::insert(heapZero, slot, runZero[slot].sum / runZero[slot].sCount);
  }
}


void RunSet::heapBinary() {
  // Ordering by category probability is equivalent to ordering by
  // concentration, as weighting by priors does not affect order.
  //
  // In the absence of class weighting, numerator can be (integer) slot
  // sample count, instead of slot sum.
  for (unsigned int slot = 0; slot < runCount; slot++) {
    BHeap::insert(heapZero, slot, getSumCtg(slot, 2, 1) / runZero[slot].sum);
  }
}


void RunSet::writeImplicit(const SplitNux* cand, const SplitFrontier* sp,  const vector<double>& ctgSum) {
  IndexT implicit = cand->getImplicitCount();
  if (implicit == 0)
    return;

  IndexT sCount = cand->getSCount();
  double sum = cand->getSum();
  setSumCtg(ctgSum);

  for (unsigned int runIdx = 0; runIdx < runCount; runIdx++) {
    sCount -= runZero[runIdx].sCount;
    sum -= runZero[runIdx].sum;
    residCtg(ctgSum.size(), runIdx);
  }

  write(sp->getDenseRank(cand), sCount, sum, implicit);
}


void RunSet::setSumCtg(const vector<double>& ctgSum) {
  for (unsigned int ctg = 0; ctg < ctgSum.size(); ctg++) {
    ctgZero[runCount * ctgSum.size() + ctg] = ctgSum[ctg];
  }
}


void RunSet::residCtg(unsigned int nCtg, unsigned int runIdx) {
  for (unsigned int ctg = 0; ctg < nCtg; ctg++) {
    ctgZero[runCount * nCtg + ctg] -= getSumCtg(runIdx, nCtg, ctg);
  }
}
  

/**
   @brief Implicit runs are characterized by a start value of 'noStart'.

   @return Whether this run is dense.
 */
bool FRNode::isImplicit() {
  return range.getStart() == RunSet::noStart;
}


bool RunSet::implicitLeft() const {
  if (!hasImplicit)
    return false;

  for (unsigned int runIdx = 0; runIdx < runsLH; runIdx++) {
    unsigned int outSlot = outZero[runIdx];
    if (runZero[outSlot].isImplicit()) {
      return true;
    }
  }

  return false;
}


void RunSet::dePop(unsigned int pop) {
  return BHeap::depopulate(heapZero, outZero, pop == 0 ? runCount : pop);
}


unsigned int RunSet::deWide(unsigned int nCtg) {
  if (runCount <= maxWidth)
    return runCount;

  heapRandom();

  vector<FRNode> tempRun(maxWidth);
  vector<double> tempSum(nCtg * maxWidth); // Accessed as ctg-minor matrix.

  // Copies runs referenced by the slot list to a temporary area.
  dePop(maxWidth);
  unsigned i = 0;
  for (auto & tr : tempRun) {
    unsigned int outSlot = outZero[i];
    for (unsigned int ctg = 0; ctg < nCtg; ctg++) {
      tempSum[i * nCtg + ctg] = ctgZero[outSlot * nCtg + ctg];
    }
    tr = runZero[outSlot];
    i++;
  }

  // Overwrites existing runs with the shrunken list
  i = 0;
  for (auto tr : tempRun) {
    for (unsigned int ctg = 0; ctg < nCtg; ctg++) {
      ctgZero[i * nCtg + ctg] = tempSum[i * nCtg + ctg];
    }
    runZero[i] = tr;
    i++;
  }

  return maxWidth;
}


IndexT RunSet::lHBits(unsigned int lhBits, IndexT& lhSampCt) {
  IndexT lhExtent = 0;
  unsigned int slotSup = effCount() - 1;
  runsLH = 0;
  lhSampCt = 0;
  if (lhBits != 0) {
    for (unsigned int slot = 0; slot < slotSup; slot++) {
      // If bit # 'slot' set in 'lhBits', then the run at index
      // 'slot' belongs to the left-hand side of the split.  Its
      // sample and index counts are accumulated and its index
      // is recorded in the out-set.
      //
      if ((lhBits & (1ul << slot)) != 0) {
        unsigned int sCount;
        lhExtent += lHCounts(slot, sCount);
        lhSampCt += sCount;
        outZero[runsLH++] = slot;
      }
    }
  }

  if (implicitLeft()) {
    unsigned int rhIdx = runsLH;
    for (PredictorT slot = 0; slot < effCount(); slot++) {
      if ((lhBits & (1ul << slot)) == 0) {
        outZero[rhIdx++] = slot;
      }
    }
  }

  return lhExtent;
}


unsigned int RunSet::lHSlots(unsigned int cut, unsigned int &lhSampCt) {
  unsigned int lhExtent = 0;
  lhSampCt = 0;

  // Accumulates LH statistics from leading cut + 1 run entries.
  for (unsigned int outSlot = 0; outSlot <= cut; outSlot++) {
    unsigned int sCount;
    lhExtent += lHCounts(outZero[outSlot], sCount);
    lhSampCt += sCount;
  }

  runsLH = cut + 1;
  return lhExtent;  
}


void BHeap::insert(BHPair pairVec[], unsigned int _slot, double _key) {
  unsigned int idx = _slot;
  BHPair input;
  input.key = _key;
  input.slot = _slot;
  pairVec[idx] = input;

  int parIdx = parent(idx);
  while (parIdx >= 0 && pairVec[parIdx].key > _key) {
    pairVec[idx] = pairVec[parIdx];
    pairVec[parIdx] = input;
    idx = parIdx;
    parIdx = parent(idx);
  }
}


void BHeap::depopulate(BHPair pairVec[], unsigned int lhOut[], unsigned int pop) {
  for (int bot = pop - 1; bot >= 0; bot--) {
    lhOut[pop - (1 + bot)] = slotPop(pairVec, bot);
  }
}


unsigned int BHeap::slotPop(BHPair pairVec[], int bot) {
  unsigned int ret = pairVec[0].slot;
  if (bot == 0)
    return ret;
  
  // Places bottom element at head and refiles.
  unsigned int idx = 0;
  int slotRefile = pairVec[idx].slot = pairVec[bot].slot;
  double keyRefile = pairVec[idx].key = pairVec[bot].key;
  int descL = 1;
  int descR = 2;

    // 'descR' remains the lower of the two descendant indices.
    //  Some short-circuiting below.
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


IndexRange RunSet::getBounds(unsigned int outSlot) const {
  unsigned int slot = outZero[outSlot];
  return runZero[slot].getRange();
}


unsigned int RunSet::getRank(unsigned int outSlot) const {
  unsigned int slot = outZero[outSlot];
  return runZero[slot].getRank();
}
