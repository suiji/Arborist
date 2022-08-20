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
#include "interlevel.h"
#include "splitfrontier.h"
#include "partition.h"
#include "splitnux.h"
#include "bv.h"

#include <numeric>

RunAccum::RunAccum(const SplitFrontier* splitFrontier,
		   const SplitNux& cand,
		   SplitStyle style) :
  Accum(splitFrontier, cand),
  nCtg(splitFrontier->getNCtg()),
  runCount(cand.getRunCount()),
  sampledRuns(runCount),
  runZero(vector<RunNux>(runCount)),
  heapZero(vector<BHPair<PredictorT>>((style == SplitStyle::slots || runCount > maxWidth) ? runCount : 0)),
  cellSum(vector<double>(nCtg * runCount)),
  rvZero(nullptr),
  implicitSlot(runCount), // Inattainable slot index.
  baseTrue(0),
  runsTrue(0),
  implicitTrue(0) {
  if (nCtg == 0) // CART-style info by default:  refactor.
    info = (sum * sum) / sCount;
  else {
    ctgSum = static_cast<const SFCtg*>(splitFrontier)->ctgNodeSums(cand);
    double sumSquares = static_cast<const SFCtg*>(splitFrontier)->getSumSquares(cand);
    filterMissingCtg(cand, sumSquares, ctgSum);
    info = sumSquares / sum;
  }
}


// Caches an rvWide pointer iff required by the accumulator.
//
void RunAccum::reWide(vector<double>& rvWide, IndexT& rvOff) {
  if (runCount > maxWidth) {
    rvZero = &rvWide[rvOff];
    rvOff += runCount;
  }
}


vector<IndexRange> RunAccum::getRange(const CritEncoding& enc) const {
  PredictorT slotStart, slotEnd;
  if (enc.trueEncoding()) {
    slotStart = baseTrue;
    slotEnd = baseTrue + runsTrue;
  }
  else { // Replay indices explicit on false branch.
    slotStart = baseTrue == 0 ? runsTrue : 0;
    slotEnd = baseTrue == 0 ? sampledRuns : (sampledRuns - runsTrue);
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
  rangeVec.push_back(IndexRange(getBounds(enc.trueEncoding() ? runsTrue - 1 : sampledRuns - 1)));
  return rangeVec;
}


void RunAccum::setTrueBits(const InterLevel* interLevel,
			   const SplitNux& nux,
			   BV* splitBits,
			   size_t bitPos) const {
  for (PredictorT trueIdx = baseTrue; trueIdx < baseTrue + runsTrue; trueIdx++) {
    IndexT code = interLevel->getCode(nux, getObs(trueIdx), isImplicit(runZero[trueIdx]));
    splitBits->setBit(bitPos + code);
  }
}


void RunAccum::setObservedBits(const InterLevel* interLevel,
			       const SplitNux& nux,
			       BV* observedBits,
			       size_t bitPos) const {
  for (PredictorT runIdx = 0; runIdx != sampledRuns; runIdx++) {
    IndexT code = interLevel->getCode(nux, getObs(runIdx), isImplicit(runZero[runIdx]));
    observedBits->setBit(bitPos + code);
  }
}


void RunAccum::split(const SFReg* sf, SplitNux& cand) {
  RunAccum* runAccum = sf->getRunAccum(cand);
  runAccum->splitReg(sf, cand);
}


void RunAccum::splitReg(const SFReg* sf, SplitNux& cand) {
  double infoCell = info;
  regRuns(sf, cand);
  maxVar();
  cand.setInfo(info - infoCell);
}


/**
   Regression runs always maintained by heap.
*/
void RunAccum::regRuns(const SFReg* sf, const SplitNux& cand) {
  if (implicitCand) {
    regRunsImplicit(sf, cand);
    return;
  }
  
  PredictorT runIdx = 0;
  initReg(obsStart, runIdx);
  for (IndexT idx = obsStart + 1; idx != obsEnd; idx++) {
    if (!obsCell[idx].regAccum(runZero[runIdx])) {
      endRun(runZero[runIdx], idx - 1);
      initReg(idx, ++runIdx);
    }
  }
  
  // Flushes the remaining run.
  endRun(runZero[runIdx], obsEnd-1);
}


void RunAccum::regRunsImplicit(const SFReg* sf, const SplitNux& cand) {
  PredictorT runIdx = 0;
  if (cutResidual == obsStart)
    implicitSlot = runIdx++;
  initReg(obsStart, runIdx);
  for (IndexT obsIdx = obsStart + 1; obsIdx != obsEnd; obsIdx++) {
    if (!obsCell[obsIdx].regAccum(runZero[runIdx])) {
      endRun(runZero[runIdx], obsIdx - 1);
      if (cutResidual == obsIdx)
	implicitSlot = ++runIdx;
      initReg(obsIdx, ++runIdx);
    }
  }
  endRun(runZero[runIdx], obsEnd-1);
  if (cutResidual == obsEnd)
    implicitSlot = ++runIdx;

  applyResidual();
}


void RunAccum::regRunsMasked(const SFReg* sf,
			     const SplitNux& cand,
			     const BranchSense* branchSense,
			     IndexT edgeRight,
			     IndexT edgeLeft,
			     bool maskSense) {
  PredictorT runIdx = 0;
  initReg(edgeLeft, runIdx);
  IndexT runRight = edgeLeft; // Previous unmasked index.
  for (IndexT idx = edgeLeft + 1; idx <= edgeRight; idx++) {
    if (branchSense->isExplicit(sampleIndex[idx]) == maskSense) {
      if (!obsCell[idx].regAccum(runZero[runIdx])) {
	endRun(runZero[runIdx], runRight);
	initReg(idx, ++runIdx);
      }
      runRight = idx;
    }
  }

  // Flushes the remaining run.
  //
  endRun(runZero[runIdx], runRight);
  if (implicitCand) {
    implicitSlot = ++runIdx;
    applyResidual();
  }
}


void RunAccum::initReg(IndexT runLeft,
		       PredictorT runIdx) {
  runZero[runIdx].startRange(runLeft);
  obsCell[runLeft].regInit(runZero[runIdx]);
}


void RunAccum::maxVar() {
  orderMean();

  IndexT sCountL = 0;
  double sumL = 0.0;
  PredictorT runSlot = sampledRuns - 1;
  for (PredictorT slotTrial = 0; slotTrial < sampledRuns - 1; slotTrial++) {
    sumAccum(slotTrial, sCountL, sumL);
    if (trialSplit(infoVar(sumL, sumCount.sum - sumL, sCountL, sumCount.sCount - sCountL))) {
      runSlot = slotTrial;
    }
  }
  setToken(runSlot);
}


void RunAccum::orderMean() {
  heapMean();
  slotReorder();
}


void RunAccum::heapMean() {
  for (PredictorT slot = 0; slot < sampledRuns; slot++) {
    PQueue::insert<PredictorT>(&heapZero[0], runZero[slot].sum / runZero[slot].sCount, slot);
  }
}


void RunAccum::slotReorder(PredictorT leadCount) {
  vector<RunNux> frOrdered(leadCount == 0 ? sampledRuns : leadCount);
  idxRank = PQueue::depopulate<PredictorT>(&heapZero[0], frOrdered.size());

  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    frOrdered[idxRank[slot]] = runZero[slot];
  }
  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    runZero[slot] = frOrdered[slot];
  }
  if (implicitSlot < sampledRuns) { // Tracks movement of implicit slot, if any.
    implicitSlot = idxRank[implicitSlot];
  }
}


void RunAccum::split(const SFCtg* sf, SplitNux& cand) {
  RunAccum* runAccum = sf->getRunAccum(cand);
  runAccum->splitCtg(sf, cand);
}


void RunAccum::splitCtg(const SFCtg* sf, SplitNux& cand) {
  double infoCell = info;
  ctgRuns(sf, cand);
  if (nCtg == 2) {
    binaryGini(sf, cand);
  }
  else {
    sampledRuns = deWide();
    ctgGini(sf, cand);
  }
  cand.setInfo(info - infoCell);
}


void RunAccum::ctgRuns(const SFCtg* sf, const SplitNux& cand) {
  if (implicitCand) {
    ctgRunsImplicit(sf, cand);
    return;
  }
  
  PredictorT runIdx = 0;
  double* sumBase = initCtg(obsStart, runIdx);
  for (IndexT obsIdx = obsStart + 1; obsIdx != obsEnd; obsIdx++) {
    if (!obsCell[obsIdx].ctgAccum(runZero[runIdx], sumBase)) {
      endRun(runZero[runIdx], obsIdx - 1);
      sumBase = initCtg(obsIdx, ++runIdx);
    }
  }
  endRun(runZero[runIdx], obsEnd-1); // Flushes remaining run.
}


void RunAccum::ctgRunsImplicit(const SFCtg* sf, const SplitNux& cand) {
  // Cut position yields the run index at which to place the residual.
  // Observation at this position must not marked as tied.
  PredictorT runIdx = 0;
  if (cutResidual == obsStart)
    implicitSlot = runIdx++;
  double* sumBase = initCtg(obsStart, runIdx);
  for (IndexT obsIdx = obsStart + 1; obsIdx != obsEnd; obsIdx++) {
    if (!obsCell[obsIdx].ctgAccum(runZero[runIdx], sumBase)) {
      endRun(runZero[runIdx], obsIdx - 1);
      if (cutResidual == obsIdx) {
	implicitSlot = ++runIdx;
      }
      sumBase = initCtg(obsIdx, ++runIdx);
    }
  }
  endRun(runZero[runIdx], obsEnd-1); // Flushes remaining run.
  if (cutResidual == obsEnd)
    implicitSlot = ++runIdx;
  
  applyResidual(sf->ctgNodeSums(cand));
}


double* RunAccum::initCtg(IndexT obsLeft,
			  PredictorT runIdx) {
  double* sumBase = &cellSum[runIdx * nCtg];
  runZero[runIdx].startRange(obsLeft);
  obsCell[obsLeft].ctgInit(runZero[runIdx], sumBase);
  return sumBase;
}


void RunAccum::applyResidual(const vector<double>& sumSlice) {
  residCtg(sumSlice);
  runZero[implicitSlot].setResidual(sCount, sum, obsEnd, implicitCand);
}


void RunAccum::residCtg(const vector<double>& sumSlice) {
  if (nCtg == 0) {
    return;
  }
  double* ctgBase = &cellSum[implicitSlot * nCtg];
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    ctgBase[ctg] = sumSlice[ctg];
  }
  for (PredictorT idx = 0; idx != runCount; idx++) {
    if (idx != implicitSlot) {
      for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
	ctgBase[ctg] -= cellSum[idx * nCtg + ctg];
      }
    }
  }
}


PredictorT RunAccum::deWide() {
  if (runCount > maxWidth) {
    // Randomly samples maxWidth-many runs and reorders.
    orderRandom(maxWidth);

    // Updates the per-category response contributions to reflect the run
    // reordering.
    ctgReorder(maxWidth);
  }
  return min(runCount, maxWidth);
}


void RunAccum::orderRandom(PredictorT leadCount) {
  heapRandom();
  slotReorder(leadCount);
}


void RunAccum::heapRandom() {
  for (PredictorT slot = 0; slot < runCount; slot++) {
    PQueue::insert<PredictorT>(&heapZero[0], rvZero[slot], slot);
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


void RunAccum::ctgGini(const SFCtg* sf, const SplitNux& cand) {
  // Run index subsets as binary-encoded unsigneds.
  PredictorT trueSlots = 0; // Slot offsets of codes taking true branch.

  // High bit unset, remainder set.
  PredictorT lowSet = (1ul << (sampledRuns - 1)) - 1;

  // Arg-max over all nontrivial subsets, up to complement:
  for (unsigned int subset = 1; subset <= lowSet; subset++) {
    if (trialSplit(subsetGini(subset))) {
      trueSlots = subset;
    }
  }

  setToken(trueSlots);
}


double RunAccum::subsetGini(unsigned int subset) const {
  // getCellSum(..., ctg) decomposes 'sumCand' by category x run.
  // getSum(runIdx) decomposes 'sumCand' by run, so may be used
  // as a cross-check.
  vector<double> sumSampled(nCtg);
  for (PredictorT runIdx = 0; runIdx != sampledRuns - 1; runIdx++) {
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
    ssR += (ctgSum[ctg] - maskedSum) * (ctgSum[ctg] - maskedSum);
    ctg++;
  }

  return infoGini(ssL, ssR, sumL, sumCount.sum - sumL);
}


void RunAccum::binaryGini(const SFCtg* sf, const SplitNux& cand) {
  orderBinary();

  const double tot0 = ctgSum[0];
  const double tot1 = ctgSum[1];
  double sumL0 = 0.0; // Running left sum at category 0.
  double sumL1 = 0.0; // " " category 1.
  PredictorT argMaxRun = sampledRuns - 1;
  for (PredictorT runIdx = 0; runIdx != sampledRuns - 1; runIdx++) {
    if (accumBinary(runIdx, sumL0, sumL1)) { // Splitable
      // sumR, sumL magnitudes can be ignored if no large case/class weightings.
      FltVal sumL = sumL0 + sumL1;
      double ssL = sumL0 * sumL0 + sumL1 * sumL1;
      double ssR = (tot0 - sumL0) * (tot0 - sumL0) + (tot1 - sumL1) * (tot1 - sumL1);
      if (trialSplit(infoGini(ssL, ssR, sumL, sumCount.sum - sumL))) {
        argMaxRun = runIdx;
      }
    } 
  }
  setToken(argMaxRun);
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
  for (PredictorT slot = 0; slot < sampledRuns; slot++) {
    PQueue::insert<PredictorT>(&heapZero[0], getCellSum(slot, 1) / runZero[slot].sum, slot);
  }
}


void RunAccum::update(const SplitNux& cand, SplitStyle style) {
  if (style == SplitStyle::slots) {
    leadSlots(cand.invertTest());
  }
  else if (style == SplitStyle::bits) {
    leadBits(cand.invertTest());
  }
  else if (style == SplitStyle::topSlot) {
    topSlot();
  }
}


void RunAccum::topSlot() {
  implicitTrue += getImplicitExtent(runsTrue++);
}


void RunAccum::leadSlots(bool invertTest) {
  // 'splitToken' is the index of the cut, or highest left slot.
  PredictorT runsLeft = splitToken + 1;
  if (invertTest) {
    baseTrue = runsLeft;
    runsTrue = sampledRuns - runsLeft;
  }
  else {
    runsTrue = runsLeft;
  }
  implicitTrue = getImplicitCut();
}


void RunAccum::leadBits(bool invertTest) {
  PredictorT lhBits = splitToken;
  //  assert(lhBits != 0); // Argmax'd bits should never get here.

  // Only categories visible to this node can be incorporated into the
  // splitting decision.  By convention, the categories resident in 'true'
  // slots will take the true branch during prediction.  All other categories,
  // regardless whether visible, will take the false branch.  This includes not
  // only categories eclipsed by bagging or conditioning, but also proxy
  // categories not encountered during training, as well as NA.

  // No slot, whether implicit or explicit, should be assigned a branch
  // sense fixed a priori.  Doing so biases predictions for reasons outlined
  // above.  For this reason the true branch is randomly assigned to either
  // the argmax slot subset or its complement.  Because factor splitting is
  // expressed via set membership, the randomization can be performed during
  // training.

  if (invertTest)
    lhBits = slotComplement(lhBits);

  implicitTrue = (lhBits & (1ul << implicitSlot)) == 0 ? 0 : getImplicitExtent(implicitSlot);

  // Places true-sense runs to the left for range and code capture.
  // sampledRuns captures all factor levels visible to the cell.
  vector<RunNux> frTemp;
  for (PredictorT runIdx = 0; runIdx != sampledRuns; runIdx++) {
    if (lhBits & (1ul << runIdx)) {
      frTemp.emplace_back(runZero[runIdx]);
    }
  }
  runsTrue = frTemp.size();
  for (PredictorT runIdx = 0; runIdx != sampledRuns; runIdx++) {
    if (!(lhBits & (1ul << runIdx))) {
      frTemp.emplace_back(runZero[runIdx]);
    }
  }

  runZero = frTemp;
}
