// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file runaccum.cc

   @brief Methods for maintaining runs of factor-valued predictors during splitting.

   @author Mark Seligman
 */

#include "runaccum.h"
#include "callback.h"
#include "branchsense.h"
#include "splitfrontier.h"
#include "obspart.h"
#include "splitnux.h"


IndexT FRNode::noStart = 0;


RunSet::RunSet(const SplitFrontier* splitFrontier,
	       PredictorT nCtg_,
	       IndexT nRow) :
  style(splitFrontier->getFactorStyle()),
  nCtg(nCtg_) {
  FRNode::noStart = nRow; // Inattainable start value, irrespective of tree.
}


IndexT RunSet::addRun(const SplitFrontier* splitFrontier,
		      const SplitNux* cand,
		      PredictorT rc) {
  runAccum.emplace_back(splitFrontier, cand, nCtg, style, rc);
  return runAccum.size() - 1; // Top position.
}


RunAccum::RunAccum(const SplitFrontier* splitFrontier,
		   const SplitNux* cand,
		   PredictorT nCtg,
		   SplitStyle style,
		   PredictorT rcSafe_) :
  Accum(splitFrontier, cand),
  rcSafe(rcSafe_),
  runZero(vector<FRNode>(rcSafe)),
  heapZero(vector<BHPair>((style == SplitStyle::slots || rcSafe > maxWidth) ? rcSafe : 0)),
  idxOrdered(vector<PredictorT>(rcSafe)),
  ctgZero(vector<double>(nCtg * rcSafe)),
  rvZero(nullptr),
  runCount(0),
  runsLH(0),
  implicitTrue(0) {
}


void RunSet::setOffsets() {
  if (runAccum.empty()) {
    return;
  }

  if (nCtg > 0)
    offsetsCtg();
}


void RunSet::offsetsCtg() {
  IndexT rvRuns = 0;
  for (auto accum : runAccum) {
    rvRuns += accum.countWide();
  }
  if (rvRuns == 0) {
    return;
  }

  // Economizes by pre-allocating random variates for entire frontier.
  rvWide = CallBack::rUnif(rvRuns);
  IndexT rvOff = 0;
  for (auto & accum : runAccum) {
    accum.reWide(rvWide, rvOff);
  }
}


// Caches an rvWide pointer iff required by the accumulator.
//
void RunAccum::reWide(vector<double>& rvWide, IndexT& rvOff) {
  if (rcSafe > maxWidth) {
    rvZero = &rvWide[rvOff];
    rvOff += rcSafe;
  }
}


vector<IndexRange> RunSet::getRange(const SplitNux* nux, const CritEncoding& enc) const {
  return runAccum[nux->getAccumIdx()].getRange(enc);
}


vector<IndexRange> RunAccum::getRange(const CritEncoding& enc) const {
  PredictorT slotStart, slotEnd;
  if (enc.trueEncoding()) {
    slotStart = 0;
    slotEnd = runsLH;
  }
  else { // Replay indices explicit on false branch.
    slotStart = runsLH;
    slotEnd = runCount;
  }
  vector<IndexRange> rangeVec(slotEnd - slotStart);
  PredictorT slot = 0;
  for (PredictorT outSlot = slotStart; outSlot != slotEnd; outSlot++) {
    rangeVec[slot++] = getBounds(outSlot);
  }

  return rangeVec;
}


IndexRange RunSet::getTopRange(const SplitNux* nux, const CritEncoding& enc) const {
  return runAccum[nux->getAccumIdx()].getTopRange(enc);
}


IndexRange RunAccum::getTopRange(const CritEncoding& enc) const {
  return IndexRange(getBounds(enc.trueEncoding() ? runsLH - 1 : runCount - 1));
}


vector<PredictorT> RunSet::getTrueBits(const SplitNux* nux) const {
  return runAccum[nux->getAccumIdx()].getTrueBits();
}


vector<PredictorT> RunAccum::getTrueBits() const {
  vector<PredictorT> trueBits(runsLH);
  PredictorT outSlot = 0; // True-branch bits lie to the left;
  for (auto & bit : trueBits) {
    bit = getCode(outSlot++);
  }

  return trueBits;
}


IndexT RunSet::getImplicitTrue(const SplitNux* nux) const {
  return runAccum[nux->getAccumIdx()].getImplicitTrue();
}


PredictorT RunSet::getRunCount(const SplitNux* nux) const {
  return runAccum[nux->getAccumIdx()].getRunCount();
}


void RunSet::resetRunCount(PredictorT accumIdx,
		      PredictorT runCount) {
  runAccum[accumIdx].resetRunCount(runCount);
}


void RunSet::updateAccum(const SplitNux* cand) {
  runAccum[cand->getAccumIdx()].update(style);
}


void RunAccum::update(SplitStyle style) {
  if (style == SplitStyle::slots) {
    leadSlots(splitToken);
  }
  else if (style == SplitStyle::bits) {
    leadBits(splitToken);
  }
  else if (style == SplitStyle::topSlot) {
    topSlot();
  }
}


void RunAccum::topSlot() {
  implicitTrue += getImplicitExtent(runsLH++);
}


void RunAccum::leadSlots(PredictorT cut) {
  runsLH = cut + 1;
  implicitLeft();
}


void RunAccum::implicitLeft() {
  for (PredictorT runIdx = 0; runIdx < runsLH; runIdx++) {
    implicitTrue += getImplicitExtent(runIdx);
  }
}


void RunAccum::leadBits(PredictorT lhBits) {
  //  assert(lhBits != 0); // Argmax'd bits should never get here.

  // runsLH = popcount(lhBits); // Awaits C++20.

  // effCount() sufficient to capture all true bits.
  for (PredictorT slot = 0; slot < effCount(); slot++) {
    runsLH += (lhBits & (1ul << slot)) ? 1 : 0;
  }
  implicitLeft(); // runsLH now final.

  vector<FRNode> frTemp(runCount);
  // Places true-sense runs to the left for range and code capture.
  PredictorT off = 0;
  for (PredictorT runIdx = 0; runIdx < effCount(); runIdx++) {
    if (lhBits & (1ul << runIdx)) {
      frTemp[off++] = runZero[runIdx];
    }
  }

  // Places false-sense runs to the right for range capture only.
  // Can be omitted if the LHS is explicit.
  for (PredictorT runIdx = 0; runIdx < runCount; runIdx++) {
    if (!(lhBits & (1ul << runIdx))) {
      frTemp[off++] = runZero[runIdx];
    }
  }

  for (PredictorT runIdx = 0; runIdx < off; runIdx++) {
    runZero[runIdx] = frTemp[runIdx];
  }
}


/**
   Regression runs always maintained by heap.
*/
void RunAccum::regRuns(const SplitNux* cand) {
  IndexT idxEnd = cand->getIdxEnd();
  IndexT rkRight = sampleRank[idxEnd].getRank();
  IndexT sCountRun = 0;
  double sumRun = 0.0;
  IndexT runLeft = idxEnd;
  IndexT runRight = idxEnd;
  for (int idx = static_cast<int>(idxEnd); idx >= static_cast<int>(cand->getIdxStart()); idx--) {
    IndexT rkThis = sampleRank[idx].getRank();
    IndexT sCountThis = sampleRank[idx].getSCount();
    FltVal ySumThis = sampleRank[idx].getSum();

    if (rkThis == rkRight) { // Same run:  counters accumulate.
      sumRun += ySumThis;
      sCountRun += sCountThis;
    }
    else { // New run:  flush accumulated counters and reset.
      append(rkRight, sCountRun, sumRun, runLeft, runRight - runLeft + 1);
      rkRight = rkThis;
      sumRun = ySumThis;
      sCountRun = sCountThis;
      runRight = idx;
    }
    runLeft = idx;
  }
  
  // Flushes the remaining run and implicit run, if dense.
  //
  append(rkRight, sCountRun, sumRun, runLeft, runRight - runLeft + 1);
  appendImplicit(cand);
}


void RunAccum::appendImplicit(const SplitNux* cand, const vector<double>& ctgSum) {
  if (!cand->getImplicitCount()) {
    return;
  }
  
  IndexT sCount = cand->getSCount();
  double sum = cand->getSum();
  setSumCtg(ctgSum);

  for (PredictorT runIdx = 0; runIdx < runCount; runIdx++) {
    sCount -= runZero[runIdx].sCount;
    sum -= runZero[runIdx].sum;
    residCtg(ctgSum.size(), runIdx);
  }

  append(cand, sCount, sum);
}


void RunAccum::append(const SplitNux* cand,
		      IndexT sCount,
		      double sum) {
  append(rankDense, sCount, sum, FRNode::noStart, cand->getImplicitCount());
}


void RunAccum::regRunsMasked(const SplitNux* cand,
			       const BranchSense* branchSense,
			       IndexT idxEnd,
			       IndexT edgeLeft) {
  IndexT rkRight = sampleRank[idxEnd].getRank();
  IndexT sCountRun = 0;
  double sumRun = 0.0;
  IndexT runLeft = idxEnd; // Leftmost index of run.
  IndexT runRight = idxEnd; // Rightmost index of run.
  for (int idx = static_cast<int>(idxEnd); idx >= static_cast<int>(edgeLeft); idx--) {
    if (!branchSense->isExplicit(sampleIndex[idx])) {
      IndexT rkThis = sampleRank[idx].getRank();
      IndexT sCountThis = sampleRank[idx].getSCount();
      double ySumThis = sampleRank[idx].getSum();
      if (rkThis == rkRight) { // Same run:  counters accumulate.
	sumRun += ySumThis;
	sCountRun += sCountThis;
      }
      else { // New run:  flush accumulated counters and reset.
	append(rkRight, sCountRun, sumRun, runLeft, runRight - runLeft + 1);
	rkRight = rkThis;
	sumRun = ySumThis;
	sCountRun = sCountThis;
	runRight = idx;
      }
      runLeft = idx;
    }
  }

  // Flushes the remaining run.
  //
  append(rkRight, sCountRun, sumRun, runLeft, runRight - runLeft + 1);
  appendImplicit(cand);
}


void RunAccum::ctgRuns(const SplitNux* cand, PredictorT nCtg,
		       const vector<double>& sumSlice) {
  IndexT idxEnd = cand->getIdxEnd();
  IndexT rkRight = sampleRank[idxEnd].getRank();
  IndexT sCountRun = 0;
  double sumRun = 0.0;
  IndexT runLeft = idxEnd;
  IndexT runRight = idxEnd;
  for (int idx = static_cast<int>(idxEnd); idx >= static_cast<int>(cand->getIdxStart()); idx--) {
    IndexT rkThis = sampleRank[idx].getRank();
    IndexT sCountThis = sampleRank[idx].getSCount();
    FltVal ySumThis = sampleRank[idx].getSum();

    if (rkThis == rkRight) { // Current run's counters accumulate.
      sumRun += ySumThis;
      sCountRun += sCountThis;
    }
    else { // Flushes current run and resets counters for next run.
      append(rkRight, sCountRun, sumRun, runLeft, runRight - runLeft + 1);
      rkRight = rkThis;
      sumRun = ySumThis;
      sCountRun = sCountThis;
      runRight = idx;
    }
    runLeft = idx;
    ctgAccum(nCtg, ySumThis, sampleRank[idx].getCtg());
  }

  // Flushes remaining run and implicit blob, if any.
  append(rkRight, sCountRun, sumRun, runLeft, runRight - runLeft + 1);
  appendImplicit(cand, sumSlice);
}


void RunAccum::deWide(PredictorT nCtg) {
  if (runCount > maxWidth) {
    // Randomly samples maxWidth-many runs and reorders.
    orderRandom(maxWidth);

    // Updates the per-category response contributions to reflect the run
    // reordering.
    ctgReorder(maxWidth, nCtg);
  }
}


void RunAccum::ctgReorder(PredictorT leadCount, PredictorT nCtg) {
  vector<double> tempSum(nCtg * leadCount); // Accessed as ctg-minor matrix.
  for (PredictorT slot = 0; slot < leadCount; slot++) {
    PredictorT outSlot = idxOrdered[slot];
    for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
      tempSum[slot * nCtg + ctg] = ctgZero[outSlot * nCtg + ctg];
    }
  }

  // Overwrites existing runs with the shrunken list
  for (PredictorT slot = 0; slot < leadCount; slot++) {
    for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
      ctgZero[slot * nCtg + ctg] = tempSum[slot * nCtg + ctg];
    }
  }
}


void RunAccum::orderRandom(PredictorT leadCount) {
  heapRandom();
  // If an implicit run is present, the entire set of runs must be permuted
  // in order to retain the full complement of right-hand ranges.
  // For now, permutes the entire set regardless whether any runs be implicit.
  slotReorder(runCount);
}


void RunAccum::heapRandom() {
  for (PredictorT slot = 0; slot < runCount; slot++) {
    BHeap::insert(&heapZero[0], slot, rvZero[slot]);
  }
}


void RunAccum::slotReorder(PredictorT leadCount) {
  vector<FRNode> frOrdered(leadCount == 0 ? runCount : leadCount);
  BHeap::depopulate(&heapZero[0], &idxOrdered[0], frOrdered.size());

  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    frOrdered[slot] = runZero[idxOrdered[slot]];
  }
  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    runZero[slot] = frOrdered[slot];
  }
}


void RunAccum::orderMean() {
  heapMean();
  slotReorder();
}


void RunAccum::heapMean() {
  for (PredictorT slot = 0; slot < runCount; slot++) {
    BHeap::insert(&heapZero[0], slot, runZero[slot].sum / runZero[slot].sCount);
  }
}


void RunAccum::orderBinary() {
  heapBinary();
  slotReorder();
}


void RunAccum::heapBinary() {
  // Ordering by category probability is equivalent to ordering by
  // concentration, as weighting by priors does not affect order.
  //
  // In the absence of class weighting, numerator can be (integer) slot
  // sample count, instead of slot sum.
  for (PredictorT slot = 0; slot < runCount; slot++) {
    BHeap::insert(&heapZero[0], slot, getSumCtg(slot, 2, 1) / runZero[slot].sum);
  }
}


void RunAccum::setSumCtg(const vector<double>& ctgSum) {
  for (PredictorT ctg = 0; ctg < ctgSum.size(); ctg++) {
    ctgZero[runCount * ctgSum.size() + ctg] = ctgSum[ctg];
  }
}


void RunAccum::residCtg(PredictorT nCtg, PredictorT accumIdx) {
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    ctgZero[runCount * nCtg + ctg] -= getSumCtg(accumIdx, nCtg, ctg);
  }
}
  

struct RunDump RunAccum::dump() const {
  PredictorT startTrue = implicitTrue ? runsLH : 0;
  PredictorT runsTrue = implicitTrue ? (runCount - runsLH) : runsLH;
  return RunDump(this, startTrue, runsTrue);
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


void BHeap::depopulate(BHPair pairVec[], PredictorT lhOut[], PredictorT pop) {
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

