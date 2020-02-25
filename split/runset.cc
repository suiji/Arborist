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
#include "obspart.h"
#include "splitnux.h"
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


void Run::setOffsets(const vector<unsigned int>& safeCount, PredictorT nCtg) {
  if (nCtg > 0) {
    offsetsCtg(safeCount);
  }
  else {
    offsetsReg(safeCount);
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


vector<PredictorT> Run::getLHBits(const SplitNux* nux) const {
  PredictorT setIdx = nux->getSetIdx();
  vector<PredictorT> lhBits(runSet[setIdx].getRunsLH());
  PredictorT outSlot = 0;
  for (auto & bit : lhBits) {
    bit = runSet[setIdx].getRank(outSlot++);
  }

  return lhBits;
}


IndexRange Run::getBounds(const SplitNux* nux,
			  PredictorT outSlot) const {
  return runSet[nux->getSetIdx()].getBounds(outSlot);
}


PredictorT Run::getRunsLH(const SplitNux* nux) const {
  return runSet[nux->getSetIdx()].getRunsLH();
}


PredictorT Run::getRunCount(const SplitNux* nux) const {
  return runSet[nux->getSetIdx()].getRunCount();
}


void Run::lHBits(SplitNux* nux, PredictorT lhBits) {
  IndexT lhExtent, lhSampCt, lhImplicit;
  runSet[nux->getSetIdx()].lHBits(lhBits, lhExtent, lhSampCt, lhImplicit);
  nux->writeFac(lhSampCt, lhExtent, lhImplicit);
}


void Run::lHSlots(SplitNux* nux, PredictorT cut) {
  IndexT lhExtent, lhSampCt, lhImplicit;
  runSet[nux->getSetIdx()].lHSlots(cut, lhExtent, lhSampCt, lhImplicit);
  nux->writeFac(lhSampCt, lhExtent, lhImplicit);
}


void Run::appendSlot(SplitNux* nux) {
  IndexT lhExtent, lhSampCt, lhImplicit;
  runSet[nux->getSetIdx()].appendSlot(lhExtent, lhSampCt, lhImplicit);
  nux->writeFac(lhSampCt, lhExtent, lhImplicit);
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
                    PredictorT nCtg,
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


void RunSet::appendImplicit(const SplitNux* cand, const SplitFrontier* sp,  const vector<double>& ctgSum) {
  IndexT implicit = cand->getImplicitCount();
  if (implicit == 0)
    return;

  IndexT sCount = cand->getSCountTrue();
  double sum = cand->getSum();
  setSumCtg(ctgSum);

  for (unsigned int runIdx = 0; runIdx < runCount; runIdx++) {
    sCount -= runZero[runIdx].sCount;
    sum -= runZero[runIdx].sum;
    residCtg(ctgSum.size(), runIdx);
  }

  append(sp->getDenseRank(cand), sCount, sum, implicit);
}


void RunSet::setSumCtg(const vector<double>& ctgSum) {
  for (PredictorT ctg = 0; ctg < ctgSum.size(); ctg++) {
    ctgZero[runCount * ctgSum.size() + ctg] = ctgSum[ctg];
  }
}


void RunSet::residCtg(PredictorT nCtg, PredictorT setIdx) {
  for (unsigned int ctg = 0; ctg < nCtg; ctg++) {
    ctgZero[runCount * nCtg + ctg] -= getSumCtg(setIdx, nCtg, ctg);
  }
}
  

bool FRNode::isImplicit() {
  return range.getStart() == RunSet::noStart;
}


IndexT RunSet::implicitLeft() const {
  if (!hasImplicit)
    return 0;

  IndexT lhImplicit = 0;
  for (PredictorT runIdx = 0; runIdx < runsLH; runIdx++) {
    unsigned int outSlot = outZero[runIdx];
    if (runZero[outSlot].isImplicit()) {
      IndexT dummy;
      lhImplicit += lHCounts(outZero[outSlot], dummy);
    }
  }

  return lhImplicit;
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
    for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
      ctgZero[i * nCtg + ctg] = tempSum[i * nCtg + ctg];
    }
    runZero[i] = tr;
    i++;
  }

  return maxWidth;
}


void RunSet::lHBits(PredictorT lhBits, IndexT& lhExtent, IndexT& lhSampCt, IndexT& lhImplicit) {
  lhExtent = 0;
  lhSampCt = 0;
  unsigned int slotSup = effCount() - 1;
  runsLH = 0;
  if (lhBits != 0) {
    for (unsigned int slot = 0; slot < slotSup; slot++) {
      // If bit # 'slot' set in 'lhBits', then the run at index
      // 'slot' belongs to the left-hand side of the split.  Its
      // sample and index counts are accumulated and its index
      // is recorded in the out-set.
      //
      if ((lhBits & (1ul << slot)) != 0) {
        IndexT sCount;
        lhExtent += lHCounts(slot, sCount);
        lhSampCt += sCount;
        outZero[runsLH++] = slot;
      }
    }
  }

  lhImplicit = implicitLeft();
  if (lhImplicit > 0) {
    unsigned int rhIdx = runsLH;
    for (PredictorT slot = 0; slot < effCount(); slot++) {
      if ((lhBits & (1ul << slot)) == 0) {
        outZero[rhIdx++] = slot;
      }
    }
  }
}


void RunSet::lHSlots(PredictorT cut, IndexT& lhExtent, IndexT& lhSampCt, IndexT& lhImplicit) {
  lhExtent = 0;
  lhSampCt = 0;
  lhImplicit = 0;

  // Accumulates LH statistics from leading cut + 1 run entries.
  for (PredictorT outSlot = 0; outSlot <= cut; outSlot++) {
    IndexT sCount, extent, impCount;
    appendSlot(extent, sCount, impCount);
    lhExtent += extent;
    lhSampCt += sCount;
    lhImplicit += impCount;
  }
}


void RunSet::appendSlot(IndexT& extent, IndexT& sCount, IndexT& implicitCount) {
  extent = lHCounts(outZero[runsLH], sCount);
  implicitCount = runZero[runsLH].isImplicit() ? extent : 0;
  runsLH++;
}


IndexRange RunSet::getBounds(PredictorT outSlot) const {
  PredictorT slot = outZero[outSlot];
  return runZero[slot].getRange();
}


PredictorT RunSet::getRank(PredictorT outSlot) const {
  PredictorT slot = outZero[outSlot];
  return runZero[slot].getRank();
}


void BHeap::insert(BHPair pairVec[], unsigned int slot_, double key_) {
  unsigned int idx = slot_;
  BHPair input;
  input.key = key_;
  input.slot = slot_;
  pairVec[idx] = input;

  int parIdx = parent(idx);
  while (parIdx >= 0 && pairVec[parIdx].key > key_) {
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
