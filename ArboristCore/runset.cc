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
#include "splitcand.h"
#include "pretree.h"
#include "index.h"

unsigned int RunSet::ctgWidth = 0;
unsigned int RunSet::noStart = 0;


Run::Run(unsigned int ctgWidth_,
         unsigned int nRow,
         unsigned int noCand) :
  noRun(noCand),
  setCount(0),
  runSet(vector<RunSet>(0)),
  facRun(vector<FRNode>(0)),
  bHeap(vector<BHPair>(0)),
  lhOut(vector<unsigned int>(0)),
  ctgSum(vector<double>(0)),
  rvWide(vector<double>(0)),
  ctgWidth(ctgWidth_) {
  RunSet::ctgWidth = ctgWidth;
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

bool Run::isRun(const SplitCand& cand) const {
  return isRun(cand.getSetIdx());
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
    rs.reBase(facRun, bHeap, lhOut, ctgSum, rvWide);
  }
}

bool Run::branchFac(const SplitCand& argMax,
                    IndexSet* iSet,
                    PreTree* preTree,
                    IndexLevel* index) const {
  preTree->branchFac(argMax, iSet->getPTId());
  auto setIdx = argMax.getSetIdx();
  if (runSet[setIdx].implicitLeft()) {// LH holds bits, RH holds replay indices.
    for (unsigned int outSlot = 0; outSlot < getRunCount(setIdx); outSlot++) {
      if (outSlot < getRunsLH(setIdx)) {
        preTree->LHBit(iSet->getPTId(), getRank(setIdx, outSlot));
      }
      else {
        unsigned int runStart, runExtent;
        runBounds(setIdx, outSlot, runStart, runExtent);
        iSet->blockReplay(argMax, runStart, runExtent, index);
      }
    }
    return false;
  }
  else { // LH runs hold both bits and replay indices.
    for (unsigned int outSlot = 0; outSlot < getRunsLH(setIdx); outSlot++) {
      preTree->LHBit(iSet->getPTId(), getRank(setIdx, outSlot));
      unsigned int runStart, runExtent;
      runBounds(setIdx, outSlot, runStart, runExtent);
      iSet->blockReplay(argMax, runStart, runExtent, index);
    }
    return true;
  }
}


void Run::levelClear() {
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
                    vector<double>& rvBase) {
  runZero = &runBase[runOff];
  heapZero = &heapBase[heapOff];
  outZero = &outBase[outOff];
  rvZero = rvBase.size() > 0 ? &rvBase[heapOff] : nullptr;
  ctgZero = ctgBase.size() > 0 ?  &ctgBase[runOff * ctgWidth] : nullptr;
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
    BHeap::insert(heapZero, slot, getSumCtg(slot, 1) / runZero[slot].sum);
  }
}


void RunSet::writeImplicit(unsigned int denseRank, unsigned int sCountTot, double sumTot, unsigned int denseCount, const double nodeSum[]) {
  if (nodeSum != 0) {
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      setSumCtg(ctg, nodeSum[ctg]);
    }
  }

  for (unsigned int runIdx = 0; runIdx < runCount; runIdx++) {
    sCountTot -= runZero[runIdx].sCount;
    sumTot -= runZero[runIdx].sum;
    if (nodeSum != 0) {
      for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
        accumCtg(ctg, -getSumCtg(runIdx, ctg));
      }
    }
  }

  write(denseRank, sCountTot, sumTot, denseCount);
}


/**
   @brief Implicit runs are characterized by a start value of 'noStart'.

   @return Whether this run is dense.
 */
bool FRNode::isImplicit() {
  return start == RunSet::noStart;
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


unsigned int RunSet::deWide() {
  if (runCount <= maxWidth)
    return runCount;

  heapRandom();

  vector<FRNode> tempRun(maxWidth);
  vector<double> tempSum(ctgWidth * maxWidth); // Accessed as matrix.

  // Copies runs referenced by the slot list to a temporary area.
  dePop(maxWidth);
  unsigned i = 0;
  for (auto & tr : tempRun) {
    unsigned int outSlot = outZero[i];
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      tempSum[i * ctgWidth + ctg] = ctgZero[outSlot * ctgWidth + ctg];
    }
    tr = runZero[outSlot];
    i++;
  }

  // Overwrites existing runs with the shrunken list
  i = 0;
  for (auto tr : tempRun) {
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      ctgZero[i * ctgWidth + ctg] = tempSum[i * ctgWidth + ctg];
    }
    runZero[i] = tr;
    i++;
  }

  return maxWidth;
}


unsigned int RunSet::lHBits(unsigned int lhBits, unsigned int &lhSampCt) {
  unsigned int lhExtent = 0;
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
      if ((lhBits & (1 << slot)) != 0) {
        unsigned int sCount;
        lhExtent += lHCounts(slot, sCount);
        lhSampCt += sCount;
        outZero[runsLH++] = slot;
      }
    }
  }

  if (implicitLeft()) {
    unsigned int rhIdx = runsLH;
    for (unsigned int slot = 0; slot < effCount(); slot++) {
      if ((lhBits & (1 << slot)) == 0) {
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


void RunSet::bounds(unsigned int outSlot, unsigned int &start, unsigned int &extent) const {
  unsigned int slot = outZero[outSlot];
  runZero[slot].replayRef(start, extent);
}


unsigned int RunSet::getRank(unsigned int outSlot) const {
  unsigned int slot = outZero[outSlot];
  return runZero[slot].getRank();
}
