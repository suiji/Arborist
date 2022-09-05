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
#include "runset.h"
#include "bv.h"

#include <numeric>

RunAccum::RunAccum(const SplitFrontier* splitFrontier,
		   const SplitNux& cand,
		   const RunSet* runSet) :
  Accum(splitFrontier, cand),
  runNux(vector<RunNux>(cand.getRunCount())),
  runSup(runNux.size()),
  heapZero(vector<BHPair<PredictorT>>((runSet->style == SplitStyle::slots || runNux.size() > maxWidth) ? runNux.size() : 0)),
  implicitSlot(runNux.size()), // Inattainable slot index.
  baseTrue(0),
  runsTrue(0),
  implicitTrue(0) {
}


RunAccumReg::RunAccumReg(const SFReg* sfReg,
			 const SplitNux& cand,
			 const RunSet* runSet) : RunAccum(sfReg, cand, runSet) {
  regRuns(cand);
  info = (sumCount.sum * sumCount.sum) / sumCount.sCount;
}


RunAccumCtg::RunAccumCtg(const SFCtg* sfCtg,
			 const SplitNux& cand,
			 const RunSet* runSet) : RunAccum(sfCtg, cand, runSet),
						 nCtg(sfCtg->getNCtg()),
						 ctgNux(filterMissingCtg(sfCtg, cand)),
						 runSum(vector<double>(nCtg * runNux.size())) {
  ctgRuns(runSet, cand);
  info = ctgNux.sumSquares / sumCount.sum;
}


vector<IndexRange> RunAccum::getRange(const CritEncoding& enc) const {
  PredictorT slotStart, slotEnd;
  if (enc.trueEncoding()) {
    slotStart = baseTrue;
    slotEnd = baseTrue + runsTrue;
  }
  else { // Replay indices explicit on false branch.
    slotStart = baseTrue == 0 ? runsTrue : 0;
    slotEnd = baseTrue == 0 ? runNux.size() : (runNux.size() - runsTrue);
  }
  return getRange(slotStart, slotEnd);
}


bool RunAccum::ctgWide(const SplitFrontier* sf,
		       const SplitNux& cand) {
  return (sf->getNCtg() > 2) && (cand.getRunCount() > maxWidth);
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
  rangeVec.push_back(IndexRange(getBounds(enc.trueEncoding() ? runsTrue - 1 : runNux.size() - 1)));
  return rangeVec;
}


void RunAccum::setTrueBits(const InterLevel* interLevel,
			   const SplitNux& nux,
			   BV* splitBits,
			   size_t bitPos) const {
  for (PredictorT trueIdx = baseTrue; trueIdx < baseTrue + runsTrue; trueIdx++) {
    IndexT code = interLevel->getCode(nux, getObs(trueIdx), isImplicit(runNux[trueIdx]));
    splitBits->setBit(bitPos + code);
  }
}


void RunAccum::setObservedBits(const InterLevel* interLevel,
			       const SplitNux& nux,
			       BV* observedBits,
			       size_t bitPos) const {
  for (PredictorT runIdx = 0; runIdx != runNux.size(); runIdx++) {
    IndexT code = interLevel->getCode(nux, getObs(runIdx), isImplicit(runNux[runIdx]));
    observedBits->setBit(bitPos + code);
  }
}


void RunAccumReg::split(const SFReg* sf, RunSet* runSet, SplitNux& cand) {
  unique_ptr<RunAccumReg> runAccum = make_unique<RunAccumReg>(sf, cand, runSet);
  cand.setInfo(runAccum->split());
  runSet->addRun(move(runAccum), cand);
}


void RunAccumCtg::split(const SFCtg* sf, RunSet* runSet, SplitNux& cand) {
  unique_ptr<RunAccumCtg> runAccum = make_unique<RunAccumCtg>(sf, cand, runSet);
  cand.setInfo(runAccum->split());
  runSet->addRun(move(runAccum), cand);
}


double RunAccumReg::split() {
  return maxVar();
}


/**
   Regression runs always maintained by heap.
*/
void RunAccum::regRuns(const SplitNux& cand) {
  if (implicitCand) {
    regRunsImplicit(cand);
    return;
  }

  PredictorT runIdx = 0;
  initReg(obsStart, runIdx);
  for (IndexT idx = obsStart + 1; idx != obsEnd; idx++) {
    if (!obsCell[idx].regAccum(runNux[runIdx])) {
      runNux[runIdx].endRange(idx-1);
      initReg(idx, ++runIdx);
    }
  }
  
  // Flushes the remaining run.
  runNux[runIdx].endRange(obsEnd-1);
}


void RunAccum::regRunsImplicit(const SplitNux& cand) {
  SumCount scExplicit(sumCount);
  PredictorT runIdx = 0;
  if (cutResidual == obsStart)
    implicitSlot = runIdx++;
  initReg(obsStart, runIdx);
  for (IndexT obsIdx = obsStart + 1; obsIdx != obsEnd; obsIdx++) {
    if (!obsCell[obsIdx].regAccum(runNux[runIdx])) {
      endRun(runNux[runIdx], scExplicit, obsIdx - 1);
      if (cutResidual == obsIdx)
	implicitSlot = ++runIdx;
      initReg(obsIdx, ++runIdx);
    }
  }
  endRun(runNux[runIdx], scExplicit, obsEnd-1);
  if (cutResidual == obsEnd)
    implicitSlot = ++runIdx;

  applyResidual(scExplicit);
}


void RunAccum::regRunsMasked(const SplitNux& cand,
			     const BranchSense* branchSense,
			     IndexT edgeRight,
			     IndexT edgeLeft,
			     bool maskSense) {
  SumCount scExplicit(sumCount);
  PredictorT runIdx = 0;
  initReg(edgeLeft, runIdx);
  IndexT runRight = edgeLeft; // Previous unmasked index.
  for (IndexT idx = edgeLeft + 1; idx <= edgeRight; idx++) {
    if (branchSense->isExplicit(sampleIndex[idx]) == maskSense) {
      if (!obsCell[idx].regAccum(runNux[runIdx])) {
	endRun(runNux[runIdx], scExplicit, runRight);
	initReg(idx, ++runIdx);
      }
      runRight = idx;
    }
  }

  // Flushes the remaining run.
  //
  endRun(runNux[runIdx], scExplicit, runRight);
  if (implicitCand) {
    implicitSlot = ++runIdx;
    applyResidual(scExplicit);
  }
}


void RunAccum::initReg(IndexT runLeft,
		       PredictorT runIdx) {
  runNux[runIdx].startRange(runLeft);
  obsCell[runLeft].regInit(runNux[runIdx]);
}


double RunAccum::maxVar() {
  double infoCell = info;
  orderMean();

  SumCount scAccum;
  PredictorT runSlot = runNux.size() - 1;
  for (PredictorT slotTrial = 0; slotTrial < runNux.size() - 1; slotTrial++) {
    sumAccum(slotTrial, scAccum);
    if (trialSplit(infoVar(scAccum, sumCount))) {
      runSlot = slotTrial;
    }
  }
  setToken(runSlot);
  return info - infoCell;
}


void RunAccum::orderMean() {
  heapMean();
  slotReorder();
}


void RunAccum::heapMean() {
  for (PredictorT slot = 0; slot < runNux.size(); slot++) {
    PQueue::insert<PredictorT>(&heapZero[0], runNux[slot].sum / runNux[slot].sCount, slot);
  }
}


void RunAccum::slotReorder() {
  vector<RunNux> frOrdered(runNux.size());
  vector<PredictorT> idxRank = PQueue::depopulate<PredictorT>(&heapZero[0], frOrdered.size());

  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    frOrdered[idxRank[slot]] = runNux[slot];
  }
  runNux = frOrdered;

  if (implicitSlot < runNux.size()) { // Tracks movement of implicit slot, if any.
    implicitSlot = idxRank[implicitSlot];
  }
}


void RunAccumCtg::ctgRuns(const RunSet* runSet, const SplitNux& cand) {
  if (implicitCand)
    runsImplicit(cand);
  else
    runsExplicit(cand);

  if (nCtg > 2 && runNux.size() > maxWidth)
    sampleRuns(runSet, cand);
}


void RunAccumCtg::runsExplicit(const SplitNux& cand) {
  PredictorT runIdx = 0;
  double* sumBase = initCtg(obsStart, runIdx);
  for (IndexT obsIdx = obsStart + 1; obsIdx != obsEnd; obsIdx++) {
    if (!obsCell[obsIdx].ctgAccum(runNux[runIdx], sumBase)) {
      runNux[runIdx].endRange(obsIdx - 1);
      sumBase = initCtg(obsIdx, ++runIdx);
    }
  }
  runNux[runIdx].endRange(obsEnd-1); // Flushes remaining run.
}


void RunAccumCtg::runsImplicit(const SplitNux& cand) {
  // Cut position yields the run index at which to place the residual.
  // Observation at this position must not marked as tied.
  SumCount scExplicit(sumCount);
  PredictorT runIdx = 0;
  if (cutResidual == obsStart)
    implicitSlot = runIdx++;
  double* sumBase = initCtg(obsStart, runIdx);
  for (IndexT obsIdx = obsStart + 1; obsIdx != obsEnd; obsIdx++) {
    if (!obsCell[obsIdx].ctgAccum(runNux[runIdx], sumBase)) {
      endRun(runNux[runIdx], scExplicit, obsIdx - 1);
      if (cutResidual == obsIdx) {
	implicitSlot = ++runIdx;
      }
      sumBase = initCtg(obsIdx, ++runIdx);
    }
  }
  endRun(runNux[runIdx], scExplicit, obsEnd-1); // Flushes remaining run.
  if (cutResidual == obsEnd)
    implicitSlot = ++runIdx;

  residCtg();
  applyResidual(scExplicit);
}


void RunAccumCtg::sampleRuns(const RunSet* runSet, const SplitNux& cand) {
  vector<PredictorT> runIdx(runNux.size());
  iota(runIdx.begin(), runIdx.end(), 0);
  bool implicitSampled = false;
  const double* rvAccum = runSet->rvSlice(cand.getAccumIdx());
  vector<double> tempCtgSum(nCtg);
  vector<double> tempSum(maxWidth * nCtg);
  vector<RunNux> rvNux(maxWidth);
  for (unsigned int idx = 0; idx < maxWidth; idx++) {
    unsigned int runRandom = rvAccum[idx] * (runNux.size() - idx);
    runIdx[runRandom] = runIdx[runNux.size() - idx - 1];
    rvNux[idx] = runNux[runRandom];
    for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
      double sumCtg = runSum[runRandom * nCtg + ctg];
      tempCtgSum[ctg] += sumCtg;
      tempSum[idx * nCtg + ctg] = sumCtg;
    }
    if (implicitSlot == runRandom) {
      implicitSampled = true;
      implicitSlot = idx;
    }
  }

  double tempSS = 0;
  for (PredictorT ctg = 0; ctg < nCtg; ctg++)
    tempSS += tempCtgSum[ctg] * tempCtgSum[ctg];

  ctgNux = CtgNux(tempCtgSum, tempSS);
  runSum = tempSum;
  runNux = rvNux;
  runSup = maxWidth;
  if (!implicitSampled)
    implicitSlot = runNux.size();
}


double* RunAccumCtg::initCtg(IndexT obsLeft,
			  PredictorT runIdx) {
  double* sumBase = &runSum[runIdx * nCtg];
  runNux[runIdx].startRange(obsLeft);
  obsCell[obsLeft].ctgInit(runNux[runIdx], sumBase);
  return sumBase;
}


void RunAccum::applyResidual(const SumCount& scResidual) {
  runNux[implicitSlot].setResidual(scResidual, obsEnd, implicitCand);
}


void RunAccumCtg::residCtg() {
  double* ctgBase = &runSum[implicitSlot * nCtg];
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    ctgBase[ctg] = ctgNux.ctgSum[ctg];
  }
  for (PredictorT idx = 0; idx != runNux.size(); idx++) {
    if (idx != implicitSlot) {
      for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
	ctgBase[ctg] -= runSum[idx * nCtg + ctg];
      }
    }
  }
}


double RunAccumCtg::split() {
  if (nCtg == 2)
    return binaryGini();
  else
    return ctgGini();
}


double RunAccumCtg::ctgGini() {
  double infoCell = info;
  // Run index subsets as binary-encoded unsigneds.
  PredictorT trueSlots = 0; // Slot offsets of codes taking true branch.

  // High bit unset, remainder set.
  PredictorT lowSet = (1ul << (runNux.size() - 1)) - 1;

  // Arg-max over all nontrivial subsets, up to complement:
  for (unsigned int subset = 1; subset <= lowSet; subset++) {
    if (trialSplit(subsetGini(subset))) {
      trueSlots = subset;
    }
  }

  setToken(trueSlots);
  return info - infoCell;
}


double RunAccumCtg::subsetGini(unsigned int subset) const {
  // getRunSum(..., ctg) decomposes 'sumCand' by category x run.
  // getSum(runIdx) decomposes 'sumCand' by run, so may be used
  // as a cross-check.
  vector<double> sumSampled(nCtg);
  for (PredictorT runIdx = 0; runIdx != runNux.size() - 1; runIdx++) {
    if (subset & (1ul << runIdx)) {
      for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
	sumSampled[ctg] += getRunSum(runIdx, ctg);
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
    ssR += (ctgNux.ctgSum[ctg] - maskedSum) * (ctgNux.ctgSum[ctg] - maskedSum);
    ctg++;
  }

  return infoGini(ssL, ssR, sumL, sumCount.sum - sumL);
}


double RunAccumCtg::binaryGini() {
  double infoCell = info;
  orderBinary();

  const double tot0 = ctgNux.ctgSum[0];
  const double tot1 = ctgNux.ctgSum[1];
  double sumL0 = 0.0; // Running left sum at category 0.
  double sumL1 = 0.0; // " " category 1.
  PredictorT argMaxRun = runNux.size() - 1;
  for (PredictorT runIdx = 0; runIdx != runNux.size() - 1; runIdx++) {
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
  return info - infoCell;
}


void RunAccumCtg::orderBinary() {
  heapBinary();
  slotReorder();
}


void RunAccumCtg::heapBinary() {
  // Ordering by category probability is equivalent to ordering by
  // concentration, as weighting by priors does not affect order.
  //
  // In the absence of class weighting, numerator can be (integer) slot
  // sample count, instead of slot sum.
  for (PredictorT slot = 0; slot < runNux.size(); slot++) {
    PQueue::insert<PredictorT>(&heapZero[0], getRunSum(slot, 1) / runNux[slot].sum, slot);
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
    runsTrue = runNux.size() - runsLeft;
  }
  else {
    runsTrue = runsLeft;
  }
  implicitTrue = getImplicitCut();
}


void RunAccum::leadBits(bool invertTest) {
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

  PredictorT lhBits = invertTest ? slotComplement(splitToken) : splitToken;
  implicitTrue = (lhBits & (1ul << implicitSlot)) == 0 ? 0 : getImplicitExtent(implicitSlot);

  // Places true-sense runs to the left for range and code capture.
  // runNux.size() captures all factor levels visible to the cell.
  vector<RunNux> frTemp;
  for (PredictorT runIdx = 0; runIdx != runNux.size(); runIdx++) {
    if (lhBits & (1ul << runIdx)) {
      frTemp.emplace_back(runNux[runIdx]);
    }
  }
  runsTrue = frTemp.size();
  for (PredictorT runIdx = 0; runIdx != runNux.size(); runIdx++) {
    if (!(lhBits & (1ul << runIdx))) {
      frTemp.emplace_back(runNux[runIdx]);
    }
  }

  runNux = frTemp;
}
