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


RunAccum::RunAccum(const SplitFrontier* splitFrontier,
		   const SplitNux* cand,
		   SplitStyle style,
		   PredictorT rcSafe_) :
  Accum(splitFrontier, cand),
  rankResidual(cand->getRankResidual()),
  nCtg(splitFrontier->getNCtg()),
  rcSafe(rcSafe_),
  runZero(vector<RunNux>(rcSafe)),
  heapZero(vector<BHPair<PredictorT>>((style == SplitStyle::slots || rcSafe > maxWidth) ? rcSafe : 0)),
  cellSum(vector<double>(nCtg * rcSafe)),
  rvZero(nullptr),
  implicitSlot(rcSafe), // Inattainable slot index.
  baseTrue(0),
  runsTrue(0),
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
    slotStart = baseTrue;
    slotEnd = baseTrue + runsTrue;
  }
  else { // Replay indices explicit on false branch.
    slotStart = baseTrue == 0 ? runsTrue : 0;
    slotEnd = baseTrue == 0 ? obsCount : (obsCount - runsTrue);
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
  rangeVec.push_back(IndexRange(getBounds(enc.trueEncoding() ? runsTrue - 1 : obsCount - 1)));
  return rangeVec;
}


void RunAccum::setTrueBits(const InterLevel* interLevel,
			   const SplitNux& nux,
			   BV* splitBits,
			   size_t bitPos) const {
  for (PredictorT trueIdx = baseTrue; trueIdx < baseTrue + runsTrue; trueIdx++) {
    IndexT code = isImplicit(runZero[trueIdx]) ? rankResidual : interLevel->getCode(nux, getObs(trueIdx));
    splitBits->setBit(bitPos + code);
  }
}


void RunAccum::setObservedBits(const InterLevel* interLevel,
			       const SplitNux& nux,
			       BV* observedBits,
			       size_t bitPos) const {
  for (PredictorT runIdx = 0; runIdx != obsCount; runIdx++) {
    IndexT code = isImplicit(runZero[runIdx]) ? rankResidual : interLevel->getCode(nux, getObs(runIdx));
    observedBits->setBit(bitPos + code);
  }
}


void RunAccum::split(const SFReg* sf, SplitNux* cand) {
  RunAccum* runAccum = sf->getRunAccum(cand);
  runAccum->splitReg(sf, cand);
}


void RunAccum::splitReg(const SFReg* sf, SplitNux* cand) {
  obsCount = regRuns(sf, cand);
  maxVar();
  cand->infoGain(this);
}


/**
   Regression runs always maintained by heap.
*/
PredictorT RunAccum::regRuns(const SFReg* sf, const SplitNux* cand) {
  PredictorT runIdx = 0;
  initReg(obsStart, runIdx);
  for (IndexT idx = obsStart + 1; idx != obsEnd; idx++) {
    if (!obsCell[idx].regAccum(runZero[runIdx])) {
      endRun(runZero[runIdx], idx - 1);
      initReg(idx, ++runIdx);
    }
  }
  
  // Flushes the remaining run and residual, if any.
  //
  endRun(runZero[runIdx], obsEnd-1);
  return appendImplicit(++runIdx);
}


PredictorT RunAccum::regRunsMasked(const SFReg* sf,
				   const SplitNux* cand,
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
  return appendImplicit(++runIdx);
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
  PredictorT runSlot = obsCount - 1;
  for (PredictorT slotTrial = 0; slotTrial < obsCount - 1; slotTrial++) {
    sumAccum(slotTrial, sCountL, sumL);
    if (trialSplit(infoVar(sumL, sumCand - sumL, sCountL, sCountCand - sCountL))) {
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
  for (PredictorT slot = 0; slot < obsCount; slot++) {
    PQueue::insert<PredictorT>(&heapZero[0], runZero[slot].sum / runZero[slot].sCount, slot);
  }
}


void RunAccum::slotReorder(PredictorT leadCount) {
  vector<RunNux> frOrdered(leadCount == 0 ? obsCount : leadCount);
  idxRank = PQueue::depopulate<PredictorT>(&heapZero[0], frOrdered.size());

  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    frOrdered[idxRank[slot]] = runZero[slot];
  }
  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    runZero[slot] = frOrdered[slot];
  }
  if (implicitSlot < obsCount) { // Tracks movement of implicit slot, if any.
    implicitSlot = idxRank[implicitSlot];
  }
}


void RunAccum::split(const SFCtg* sf, SplitNux* cand) {
  RunAccum* runAccum = sf->getRunAccum(cand);
  runAccum->splitCtg(sf, cand);
}


void RunAccum::splitCtg(const SFCtg* sf, SplitNux* cand) {
  PredictorT runCount = ctgRuns(sf, cand);
  if (nCtg == 2) {
    obsCount = runCount;
    binaryGini(sf, cand);
  }
  else {
    obsCount = deWide(runCount);
    ctgGini(sf, cand);
  }
  cand->infoGain(this);
}


PredictorT RunAccum::ctgRuns(const SFCtg* sf, const SplitNux* cand) {
  PredictorT runIdx = 0;
  double* sumBase = initCtg(obsStart, runIdx);
  for (IndexT obsIdx = obsStart + 1; obsIdx != obsEnd; obsIdx++) {
    if (!obsCell[obsIdx].ctgAccum(runZero[runIdx], sumBase)) {
      endRun(runZero[runIdx], obsIdx - 1);
      sumBase = initCtg(obsIdx, ++runIdx);
    }
  }
  endRun(runZero[runIdx], obsEnd-1); // Flushes remaining run.
  
  // Flushes residual, if any.
  return appendImplicit(++runIdx, sf->getSumSlice(cand));
}


double* RunAccum::initCtg(IndexT runLeft,
			  PredictorT runIdx) {
  double* sumBase = &cellSum[runIdx * nCtg];
  runZero[runIdx].startRange(runLeft);
  obsCell[runLeft].ctgInit(runZero[runIdx], sumBase);
  return sumBase;
}


PredictorT RunAccum::appendImplicit(PredictorT runIdx,
			      const vector<double>& sumSlice) {
  implicitSlot = runIdx;
  if (implicitCand) {
    residCtg(sumSlice, runIdx);
    runZero[runIdx].setResidual(rankResidual, sCount, sum, obsEnd, implicitCand);
    return runIdx + 1;
  }
  return runIdx;
}


void RunAccum::residCtg(const vector<double>& sumSlice,
			PredictorT runIdx) {
  if (nCtg == 0) {
    return;
  }
  double* ctgBase = &cellSum[runIdx * nCtg];
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    ctgBase[ctg] = sumSlice[ctg];
  }
  for (PredictorT idx = 0; idx < runIdx; idx++) {
    for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
      ctgBase[ctg] -= cellSum[idx * nCtg + ctg];
    }
  }
}


PredictorT RunAccum::deWide(PredictorT runCount) {
  if (runCount > maxWidth) {
    // Randomly samples maxWidth-many runs and reorders.
    orderRandom(runCount, maxWidth);

    // Updates the per-category response contributions to reflect the run
    // reordering.
    ctgReorder(maxWidth);
  }
  return min(runCount, maxWidth);
}


void RunAccum::orderRandom(PredictorT runCount,
			   PredictorT leadCount) {
  heapRandom(runCount);
  slotReorder(leadCount);
}


void RunAccum::heapRandom(PredictorT runCount) {
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


void RunAccum::ctgGini(const SFCtg* sf, const SplitNux* cand) {
  // Run index subsets as binary-encoded unsigneds.
  PredictorT trueSlots = 0; // Slot offsets of codes taking true branch.

  // High bit unset, remainder set.
  PredictorT lowSet = (1ul << (obsCount - 1)) - 1;

  // Arg-max over all nontrivial subsets, up to complement:
  const vector<double>& sumSlice = sf->getSumSlice(cand);
  for (unsigned int subset = 1; subset <= lowSet; subset++) {
    if (trialSplit(subsetGini(sumSlice, subset))) {
      trueSlots = subset;
    }
  }

  setToken(trueSlots);
}


double RunAccum::subsetGini(const vector<double>& sumSlice,
			    unsigned int subset) const {
  // getCellSum(..., ctg) decomposes 'sumCand' by category x run.
  // getSum(runIdx) decomposes 'sumCand' by run, so may be used
  // as a cross-check.
  vector<double> sumSampled(nCtg);
  for (PredictorT runIdx = 0; runIdx != obsCount - 1; runIdx++) {
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
  PredictorT argMaxRun = obsCount - 1;
  for (PredictorT runIdx = 0; runIdx != obsCount - 1; runIdx++) {
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
  for (PredictorT slot = 0; slot < obsCount; slot++) {
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
    runsTrue = obsCount - runsLeft;
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
  // obsCount captures all factor levels visible to the cell.
  vector<RunNux> frTemp;
  for (PredictorT runIdx = 0; runIdx != obsCount; runIdx++) {
    if (lhBits & (1ul << runIdx)) {
      frTemp.emplace_back(runZero[runIdx]);
    }
  }
  runsTrue = frTemp.size();
  for (PredictorT runIdx = 0; runIdx != obsCount; runIdx++) {
    if (!(lhBits & (1ul << runIdx))) {
      frTemp.emplace_back(runZero[runIdx]);
    }
  }

  runZero = frTemp;
}
