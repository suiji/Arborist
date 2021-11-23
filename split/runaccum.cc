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


RunAccum::RunAccum(const SplitFrontier* splitFrontier,
		   const SplitNux* cand,
		   SplitStyle style,
		   PredictorT rcSafe_) :
  Accum(splitFrontier, cand),
  nCtg(splitFrontier->getNCtg()),
  rcSafe(rcSafe_),
  runZero(vector<RunNux>(rcSafe)),
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
  return getRange(slotStart, slotEnd);
}


vector<IndexRange> RunAccum::getRange(PredictorT slotStart,
				      PredictorT slotEnd) const {
  vector<IndexRange> rangeVec(slotEnd - slotStart);
  PredictorT slot = 0;
  for (PredictorT outSlot = slotStart; outSlot != slotEnd; outSlot++) {
    rangeVec[slot++] = getBounds(outSlot);
  }

  return rangeVec;
}


vector<IndexRange> RunAccum::getTopRange(const CritEncoding& enc) const {
  vector<IndexRange> rangeVec;
  rangeVec.push_back(IndexRange(getBounds(enc.trueEncoding() ? runsLH - 1 : runCount - 1)));
  return rangeVec;
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


void RunAccum::leadBits(PredictorT lhBits) {
  //  assert(lhBits != 0); // Argmax'd bits should never get here.

  // Places true-sense runs to the left for range and code capture.
  implicitTrue = (lhBits & (1ul << implicitSlot)) == 0 ? 0 : getImplicitExtent(implicitSlot);

  vector<RunNux> frTemp(rcSafe);
  PredictorT off = 0;

  // effCount() captures all true bits.
  for (PredictorT runIdx = 0; runIdx < effCount(); runIdx++) {
    if (lhBits & (1ul << runIdx)) {
      frTemp[off++] = runZero[runIdx];
    }
  }
  runsLH = off;

  for (PredictorT runIdx = 0; runIdx < runCount; runIdx++) {
    if (!(lhBits & (1ul << runIdx))) {
      frTemp[off++] = runZero[runIdx];
    }
  }

  runZero = frTemp;
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
void RunAccum::regRuns() {
  initReg(idxStart);
  for (IndexT idx = idxStart + 1; idx <= idxEnd; idx++) {
    if (!sampleRank[idx].regAccum(runZero[runCount])) {
      endRun(idx - 1);
      initReg(idx);
    }
  }
  
  // Flushes the remaining run and implicit run, if dense.
  //
  endRun(idxEnd);
  appendImplicit();
}


void RunAccum::regRunsMasked(const BranchSense* branchSense,
			     IndexT edgeRight,
			     IndexT edgeLeft,
			     bool maskSense) {
  initReg(edgeLeft);
  IndexT runRight = edgeLeft; // Previous unmasked index.
  for (IndexT idx = edgeLeft + 1; idx <= edgeRight; idx++) {
    if (branchSense->isExplicit(sampleIndex[idx]) == maskSense) {
      if (!sampleRank[idx].regAccum(runZero[runCount])) {
	endRun(runRight);
	initReg(idx);
      }
      runRight = idx;
    }
  }

  // Flushes the remaining run.
  //
  endRun(runRight);
  appendImplicit();
}


void RunAccum::ctgRuns(const SFCtg* sf, const SplitNux* cand) {
  double* sumBase = initCtg(idxStart);
  for (IndexT idx = idxStart + 1; idx <= idxEnd; idx++) {
    if (!sampleRank[idx].ctgAccum(runZero[runCount], sumBase)) {
      endRun(idx - 1);
      sumBase = initCtg(idx);
    }
  }
  endRun(idxEnd); // Flushes remaining run.
  
  // Flushes implicit blob, if any.
  appendImplicit(sf->getSumSlice(cand));
}


double* RunAccum::initCtg(IndexT runLeft) {
  double* sumBase = &cellSum[runCount * nCtg];
  runZero[runCount].startRange(runLeft);
  sampleRank[runLeft].ctgInit(runZero[runCount], sumBase);
  return sumBase;
}


void RunAccum::initReg(IndexT runLeft) {
  runZero[runCount].startRange(runLeft);
  sampleRank[runLeft].regInit(runZero[runCount]);
}


void RunAccum::appendImplicit(const vector<double>& sumSlice) {
  implicitSlot = runCount;
  if (implicitCand) {
    residCtg(sumSlice);
    runZero[runCount++].set(rankDense, sCount, sum, implicitCand);
  }
}


void RunAccum::residCtg(const vector<double>& sumSlice) {
  if (nCtg == 0) {
    return;
  }
  double* ctgBase = &cellSum[runCount * nCtg];
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    ctgBase[ctg] = sumSlice[ctg];
  }
  for (PredictorT runIdx = 0; runIdx < runCount; runIdx++) {
    for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
      ctgBase[ctg] -= cellSum[runIdx * nCtg + ctg];
    }
  }
}


void RunAccum::deWide() {
  if (runCount > maxWidth) {
    // Randomly samples maxWidth-many runs and reorders.
    orderRandom(maxWidth);

    // Updates the per-category response contributions to reflect the run
    // reordering.
    ctgReorder(maxWidth);
  }
}


void RunAccum::ctgReorder(PredictorT leadCount) {
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
  slotReorder(runCount);
}


void RunAccum::heapRandom() {
  for (PredictorT slot = 0; slot < runCount; slot++) {
    BHeap::insert(&heapZero[0], slot, rvZero[slot]);
  }
}


void RunAccum::slotReorder(PredictorT leadCount) {
  vector<RunNux> frOrdered(leadCount == 0 ? runCount : leadCount);
  BHeap::depopulate(&heapZero[0], &idxRank[0], frOrdered.size());

  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    frOrdered[idxRank[slot]] = runZero[slot];
  }
  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    runZero[slot] = frOrdered[slot];
  }
  if (implicitSlot < runCount) { // Tracks movement of implicit slot, if any.
    implicitSlot = idxRank[implicitSlot];
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
    BHeap::insert(&heapZero[0], slot, getCellSum(slot, 1) / runZero[slot].sum);
  }
}


struct RunDump RunAccum::dump() const {
  PredictorT startTrue = implicitTrue ? runsLH : 0;
  PredictorT runsTrue = implicitTrue ? (runCount - runsLH) : runsLH;
  return RunDump(this, startTrue, runsTrue);
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


void RunAccum::split(const SFReg* sf, SplitNux* cand) {
  RunAccum* runAccum = sf->getRunAccum(cand);
  runAccum->splitReg(cand);
}


void RunAccum::splitReg(SplitNux* cand) {
  regRuns();
  maxVar();
  cand->infoGain(this);
}


void RunAccum::split(const SFCtg* sf, SplitNux* cand) {
  RunAccum* runAccum = sf->getRunAccum(cand);
  runAccum->splitCtg(sf, cand);
}


void RunAccum::splitCtg(const SFCtg* sf, SplitNux* cand) {
  ctgRuns(sf, cand);
  if (nCtg == 2)
    binaryGini(sf, cand);
  else
    ctgGini(sf, cand);
  cand->infoGain(this);
}


void RunAccum::ctgGini(const SFCtg* sf, const SplitNux* cand) {
  deWide();

  // Run index subsets as binary-encoded unsigneds.
  PredictorT trueSlots = 0; // Slot offsets of codes taking true branch.

  // High bit unset, remainder set.
  PredictorT lowSet = (1ul << (effCount() - 1)) - 1;

  // Only categories present at this node can be incorporated into the
  // splitting decision.  By convention, the categories resident in 'true'
  // slots will take the true branch during prediction.  All other categories,
  // regardless whether observed at this node, will take the false branch.
  // This includes not only categories eclipsed by bagging or conditioning,
  // but also proxy categories not present during training.

  // Arg-max over all nontrivial subsets, up to complement:
  const vector<double>& sumSlice = sf->getSumSlice(cand);
  for (unsigned int subset = 1; subset <= lowSet; subset++) {
    if (trialSplit(subsetGini(sumSlice, subset))) {
      trueSlots = subset;
    }
  }

  // No slot, whether implicit or explicit, should be assigned a branch
  // sense fixed a priori.  Doing so biases predictions for reasons outlined
  // above.  For this reason the true branch is "randomly" assigned to either
  // the argmax slot subset or its complement.

  if (cand->getNodeIdx() & 1) // Ersatz temporary "randomization".
    trueSlots = slotComplement(trueSlots);

  setToken(trueSlots);
}


double RunAccum::subsetGini(const vector<double>& sumSlice,
			    unsigned int subset) const {
  // getCellSum(..., ctg) decomposes 'sumCand' by category x run.
  // getSum(runIdx) decomposes 'sumCand' by run, so may be used
  // as a cross-check.
  vector<double> sumSampled(nCtg);
  for (PredictorT runIdx = 0; runIdx < effCount() - 1; runIdx++) {
    if (subset & (1ul << runIdx)) {
      for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
	sumSampled[ctg] += getCellSum(runIdx, ctg);
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


void RunAccum::binaryGini(const SFCtg* sf, const SplitNux* cand) {
  const vector<double>& sumSlice = sf->getSumSlice(cand);
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
