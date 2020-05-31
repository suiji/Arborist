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
#include "branchsense.h"
#include "splitfrontier.h"
#include "obspart.h"
#include "splitnux.h"

IndexT FRNode::noStart = 0;


RunAccum::RunAccum(const SplitFrontier* splitFrontier,
		   const SplitNux* cand,
		   PredictorT nCtg,
		   SplitStyle style,
		   PredictorT rcSafe_) :
  Accum(splitFrontier, cand),
  rcSafe(rcSafe_),
  runZero(vector<FRNode>(rcSafe)),
  heapZero(vector<BHPair>((style == SplitStyle::slots || rcSafe > maxWidth) ? rcSafe : 0)),
  idxRank(vector<PredictorT>(rcSafe)),
  cellSum(vector<double>(nCtg * rcSafe)),
  rvZero(nullptr),
  implicitSlot(rcSafe), // Inattainable slot index.
  runCount(0),
  runsLH(0),
  implicitTrue(0) {
}


// Caches an rvWide pointer iff required by the accumulator.
//
void RunAccum::reWide(vector<double>& rvWide, IndexT& rvOff) {
  if (rcSafe > maxWidth) {
    rvZero = &rvWide[rvOff];
    rvOff += rcSafe;
  }
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


IndexRange RunAccum::getTopRange(const CritEncoding& enc) const {
  return IndexRange(getBounds(enc.trueEncoding() ? runsLH - 1 : runCount - 1));
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
  implicitTrue = getImplicitLeftSlots(cut);
  runsLH = cut + 1;
}


void RunAccum::implicitLeft() {
  for (PredictorT runIdx = 0; runIdx < runsLH; runIdx++) {
    implicitTrue += getImplicitExtent(runIdx);
  }
}


void RunAccum::leadBits(PredictorT lhBits) {
  //  assert(lhBits != 0); // Argmax'd bits should never get here.

  implicitTrue = getImplicitLeftBits(lhBits);

  // effCount() sufficient to capture all true bits.

  vector<FRNode> frTemp(runCount);
  // Places true-sense runs to the left for range and code capture.
  PredictorT off = 0;
  for (PredictorT runIdx = 0; runIdx < effCount(); runIdx++) {
    if (lhBits & (1ul << runIdx)) {
      frTemp[off++] = runZero[runIdx];
    }
  }
  runsLH = off;

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


vector<PredictorT> RunAccum::getTrueBits() const {
  vector<PredictorT> trueBits(runsLH);
  PredictorT outSlot = 0;
  for (auto & bit : trueBits) {
    bit = getCode(outSlot++);
  }

  return trueBits;
}


/**
   Regression runs always maintained by heap.
*/
void RunAccum::regRuns(const SplitNux* cand) {
  IndexT rkRight = sampleRank[idxEnd].getRank();
  IndexT sCountRun = 0;
  double sumRun = 0.0;
  IndexT runLeft = idxEnd;
  IndexT runRight = idxEnd;
  for (int idx = static_cast<int>(idxEnd); idx >= static_cast<int>(idxStart); idx--) {
    sampleRank[idx].refAccum(codeSR, sCountSR, ySumSR);
    if (codeSR == rkRight) { // Same run:  counters accumulate.
      sumRun += ySumSR;
      sCountRun += sCountSR;
    }
    else { // New run:  flush accumulated counters and reset.
      append(rkRight, sCountRun, sumRun, runLeft, runRight - runLeft + 1);
      rkRight = codeSR;
      sumRun = ySumSR;
      sCountRun = sCountSR;
      runRight = idx;
    }
    runLeft = idx;
  }
  
  // Flushes the remaining run and implicit run, if dense.
  //
  append(rkRight, sCountRun, sumRun, runLeft, runRight - runLeft + 1);
  appendImplicit(cand);
}


void RunAccum::appendImplicit(const SplitNux* cand, const vector<double>& sumSlice) {
  if (!cand->getImplicitCount()) {
    return;
  }
  
  sCount = sCountCand;
  sum = sumCand;
  initCtg(sumSlice);

  for (PredictorT runIdx = 0; runIdx < runCount; runIdx++) {
    sCount -= runZero[runIdx].sCount;
    sum -= runZero[runIdx].sum;
    residCtg(sumSlice.size(), runIdx);
  }

  implicitSlot = runCount;
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


void RunAccum::ctgRuns(const SplitNux* cand,
		       const vector<double>& sumSlice) {
  IndexT codeRight = sampleRank[idxEnd].getRank();
  IndexT sCountRun = 0;
  double sumRun = 0.0;
  IndexT runLeft = idxEnd;
  IndexT runRight = idxEnd;
  for (int idx = static_cast<int>(idxEnd); idx >= static_cast<int>(idxStart); idx--) {
    sampleRank[idx].refAccum(codeSR, sCountSR, ySumSR, ctgSR);
    if (codeSR == codeRight) { // Current run's counters accumulate.
      sumRun += ySumSR;
      sCountRun += sCountSR;
    }
    else { // Flushes current run and resets counters for next run.
      append(codeRight, sCountRun, sumRun, runLeft, runRight - runLeft + 1);
      codeRight = codeSR;
      sumRun = ySumSR;
      sCountRun = sCountSR;
      runRight = idx;
    }
    runLeft = idx;
    ctgAccum(sumSlice.size(), ySumSR, ctgSR);
  }
  
  // Flushes remaining run and implicit blob, if any.
  append(codeRight, sCountRun, sumRun, runLeft, runRight - runLeft + 1);
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
    PredictorT outSlot = idxRank[slot];
    for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
      tempSum[slot * nCtg + ctg] = cellSum[outSlot * nCtg + ctg];
    }
  }

  // Overwrites existing runs with the shrunken list
  for (PredictorT slot = 0; slot < leadCount; slot++) {
    for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
      cellSum[slot * nCtg + ctg] = tempSum[slot * nCtg + ctg];
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
  BHeap::depopulate(&heapZero[0], &idxRank[0], frOrdered.size());

  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    frOrdered[idxRank[slot]] = runZero[slot];
  }
  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    runZero[slot] = frOrdered[slot];
  }
  if (implicitSlot < runCount) {
    implicitSlot = idxRank[implicitSlot];
  }
}


void BHeap::depopulate(BHPair pairVec[], PredictorT idxRank[], PredictorT pop) {
  for (int bot = pop - 1; bot >= 0; bot--) {
    idxRank[slotPop(pairVec, bot)] = pop - (1 + bot);
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
    BHeap::insert(&heapZero[0], slot, getCellSum(slot, 2, 1) / runZero[slot].sum);
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


void RunAccum::maxVar() {
  orderMean();

  IndexT sCountL = 0;
  double sumL = 0.0;
  PredictorT runSlot = getRunCount() - 1;
  for (PredictorT slotTrial = 0; slotTrial < getRunCount() - 1; slotTrial++) {
    sumAccum(slotTrial, sCountL, sumL);
    if (trialSplit(infoVar(sumL, sumCand - sumL, sCountL, sCountCand - sCountL))) {
      runSlot = slotTrial;
    }
  }
  setToken(runSlot);
}


void RunAccum::ctgGini(const vector<double>& sumSlice) {
  deWide(sumSlice.size());

  // Run index subsets as binary-encoded unsigneds.
  // All nontrivial subsets with high bit unset:  complementary subsets yield
  // identical information.
  PredictorT trueSlots = 0; // Slot offsets of codes taking true branch.
  PredictorT leadHigh = 1ul << (effCount() - 1);
  for (unsigned int subset = 1; subset < leadHigh; subset++) {
    if (trialSplit(subsetGini(sumSlice, subset))) {
      trueSlots = subset;
    }
  }

  setToken(trueSlots);
}


// Symmetric w.r.t. complement:  (~subset << (32 - effCount())) >> (32 - effCount()).
double RunAccum::subsetGini(const vector<double>& sumSlice,
			    unsigned int subset) const {
  // sumSlice[ctg] decomposes the 'sumCand' by category.
  // getSum(runIdx) decomposes 'sumCand' by run.
  // getCellSum(..., ctg) decomposes 'sumCand' by category x run.
  PredictorT nCtg = sumSlice.size();
  vector<double> sumSampled(nCtg);
  for (PredictorT runIdx = 0; runIdx < effCount(); runIdx++) {
    if (subset & (1ul << runIdx)) {
      for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
	sumSampled[ctg] += getCellSum(runIdx, nCtg, ctg);
      }
    }
  }

  double ssL = 0.0;
  double sumL = 0.0;
  double ssR = 0.0;
  PredictorT ctg = 0;
  for (auto maskedSum : sumSampled) {
    sumL += maskedSum;
    ssL += maskedSum * maskedSum;
    ssR += (sumSlice[ctg] - maskedSum) * (sumSlice[ctg] - maskedSum);
    ctg++;
  }


  return infoGini(ssL, ssR, sumL, sumCand - sumL);
}


void RunAccum::binaryGini(const vector<double>& sumSlice) {
  orderBinary();

  const double tot0 = sumSlice[0];
  const double tot1 = sumSlice[1];
  double sumL0 = 0.0; // Running left sum at category 0.
  double sumL1 = 0.0; // " " category 1.
  PredictorT argMaxRun = getRunCount() - 1;
  for (PredictorT runIdx = 0; runIdx < getRunCount() - 1; runIdx++) {
    if (accumBinary(runIdx, sumL0, sumL1)) { // Splitable
      // sumR, sumL magnitudes can be ignored if no large case/class weightings.
      FltVal sumL = sumL0 + sumL1;
      double ssL = sumL0 * sumL0 + sumL1 * sumL1;
      double ssR = (tot0 - sumL0) * (tot0 - sumL0) + (tot1 - sumL1) * (tot1 - sumL1);
      if (trialSplit(infoGini(ssL, ssR, sumL, sumCand - sumL))) {
        argMaxRun = runIdx;
      }
    } 
  }
  setToken(argMaxRun);
}
