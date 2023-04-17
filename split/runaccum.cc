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
#include "splitnux.h"
#include "runfrontier.h"
#include "obs.h"

#include <numeric>

RunAccum::RunAccum(const SplitFrontier* sf,
		   const SplitNux& cand) :
  Accum(sf, cand),
  heapZero(vector<BHPair<PredictorT>>((sf->getRunSet()->style == SplitStyle::slots || cand.getRunCount() > maxWidth) ? cand.getRunCount() : 0)) {
}


RunAccumReg::RunAccumReg(const SFReg* sfReg,
			 const SplitNux& cand) : RunAccum(sfReg, cand) {
}


RunAccumCtg::RunAccumCtg(const SFCtg* sfCtg,
			 const SplitNux& cand) : RunAccum(sfCtg, cand),
						 nCtg(sfCtg->getNCtg()),
						 sampling(nCtg > 2 && cand.getRunCount() > maxWidth),
						 sampleCount(sampling ? maxWidth : cand.getRunCount()),
						 ctgNux(filterMissingCtg(sfCtg, cand)),
						 runSum(vector<double>(nCtg * cand.getRunCount())) {
}


bool RunAccum::ctgWide(const SplitFrontier* sf,
		       const SplitNux& cand) {
  return (sf->getNCtg() > 2) && (cand.getRunCount() > maxWidth);
}


/**
   Regression runs always maintained by heap.
*/
vector<RunNux> RunAccum::regRuns(const SplitNux& cand) {
  if (implicitCand) {
    return regRunsImplicit(cand);
  }
  else {
    return regRunsExplicit(cand);
  }
}


vector<RunNux> RunAccum::regRunsExplicit(const SplitNux& cand) {
  vector<RunNux> runNux(cand.getRunCount());
  PredictorT runIdx = 0;
  initReg(obsStart, runNux[runIdx]);
  for (IndexT idx = obsStart + 1; idx != obsEnd; idx++) {
    if (!obsCell[idx].regAccum(runNux[runIdx])) {
      runNux[runIdx++].endRange(idx-1);
      initReg(idx, runNux[runIdx]);
    }
  }
  
  // Flushes the remaining run.
  runNux[runIdx].endRange(obsEnd-1);

  return runNux;
}


vector<RunNux> RunAccum::regRunsImplicit(const SplitNux& cand) {
  vector<RunNux> runNux(cand.getRunCount());
  SumCount scExplicit(sumCount);
  PredictorT runIdx = 0;
  PredictorT implicitSlot = runNux.size(); // Inattainable.
  if (cutResidual == obsStart)
    implicitSlot = runIdx++;
  initReg(obsStart, runNux[runIdx]);
  for (IndexT obsIdx = obsStart + 1; obsIdx != obsEnd; obsIdx++) {
    if (!obsCell[obsIdx].regAccum(runNux[runIdx])) {
      runNux[runIdx].endRun(scExplicit, obsIdx-1);
      if (cutResidual == obsIdx)
	implicitSlot = ++runIdx;
      initReg(obsIdx, runNux[++runIdx]);
    }
  }
  runNux[runIdx].endRun(scExplicit, obsEnd-1);
  if (cutResidual == obsEnd)
    implicitSlot = ++runIdx;

  runNux[implicitSlot].setResidual(scExplicit, obsEnd, implicitCand);

  return runNux;
}


vector<RunNux> RunAccum::regRunsMasked(const SplitNux& cand,
				       const BranchSense* branchSense,
				       bool maskSense) {
  IndexRange unmaskedRange = findUnmaskedRange(branchSense, maskSense);
  IndexT edgeLeft = unmaskedRange.getStart();
  vector<RunNux> runNux(cand.getRunCount());
  SumCount scExplicit(sumCount);
  PredictorT runIdx = 0;
  initReg(edgeLeft, runNux[runIdx]);
  IndexT runRight = edgeLeft; // Previous unmasked index.
  for (IndexT idx = edgeLeft + 1; idx != unmaskedRange.getEnd(); idx++) {
    if (branchSense->isExplicit(sampleIndex[idx]) == maskSense) {
      if (!obsCell[idx].regAccum(runNux[runIdx])) {
	runNux[runIdx++].endRun(scExplicit, runRight);
	initReg(idx, runNux[runIdx]);
      }
      runRight = idx;
    }
  }

  // Flushes the remaining run.
  //
  runNux[runIdx].endRun(scExplicit, runRight);
  if (implicitCand) {
    PredictorT implicitSlot = ++runIdx;
    runNux[implicitSlot].setResidual(scExplicit, obsEnd, implicitCand);
  }

  return runNux;
}


void RunAccum::initReg(IndexT runLeft,
		       RunNux& nux) const {
  nux.startRange(runLeft);
  obsCell[runLeft].regInit(nux);
}


vector<RunNux> RunAccum::orderMean(const vector<RunNux>& runNux) {
  heapMean(runNux);
  return slotReorder(runNux);
}


void RunAccum::heapMean(const vector<RunNux>& runNux) {
  for (PredictorT slot = 0; slot < runNux.size(); slot++) {
    PQueue::insert<PredictorT>(&heapZero[0], runNux[slot].sumCount.mean(), slot);
  }
}


vector<RunNux> RunAccumCtg::ctgRuns(RunSet* runSet, const SplitNux& cand) {
  vector<RunNux> runNux;
  if (implicitCand)
    runNux = runsImplicit(cand);
  else
    runNux = runsExplicit(cand);

  if (nCtg == 2)
    runNux = orderBinary(runNux);
  else if (sampling) {
    runNux = sampleRuns(runSet, cand, runNux);
  }

  return runNux;
}


vector<RunNux> RunAccumCtg::orderBinary(const vector<RunNux>& runNux) {
  heapBinary(runNux);
  return slotReorder(runNux);
}


void RunAccumCtg::heapBinary(const vector<RunNux>& runNux) {
  // Ordering by category probability is equivalent to ordering by
  // concentration, as weighting by priors does not affect order.
  //
  // In the absence of class weighting, numerator can be (integer) slot
  // sample count, instead of slot sum.
  for (PredictorT slot = 0; slot < runNux.size(); slot++) {
    PQueue::insert<PredictorT>(&heapZero[0], getRunSum(slot, 1) / runNux[slot].sumCount.sum, slot);
  }
}


vector<RunNux> RunAccumCtg::sampleRuns(const RunSet* runSet,
				       const SplitNux& cand,
				       const vector<RunNux>& runNux) {
  const double* rvAccum = runSet->rvSlice(cand.getSigIdx());
  vector<PredictorT> idxSample(runNux.size());
  iota(idxSample.begin(), idxSample.end(), 0);

  BV runRandom(runNux.size());
  PredictorT choiceSize = runNux.size();
  for (unsigned int idx = 0; idx < sampleCount; idx++) {
    PredictorT rvIdx = rvAccum[idx] * choiceSize;
    runRandom.setBit(idxSample[rvIdx]);
    idxSample[rvIdx] = idxSample[--choiceSize];
  }

  vector<double> tempCtgSum(nCtg);
  vector<double> tempSum(sampleCount * nCtg);
  vector<RunNux> nuxSampled(sampleCount);
  PredictorT idxSampled = 0;
  PredictorT idxUnsampled = sampleCount;
  for (PredictorT idx = 0; idx < runNux.size(); idx++) {
    if (runRandom.testBit(idx)) {
      for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
	double sumCtg = runSum[idx * nCtg + ctg];
	tempCtgSum[ctg] += sumCtg;
	tempSum[idxSampled * nCtg + ctg] = sumCtg;
      }
      nuxSampled[idxSampled++] = runNux[idx];
    }
    else {
      nuxSampled[idxUnsampled++] = runNux[idx];
    }
  }

  double tempSS = 0;
  for (PredictorT ctg = 0; ctg < nCtg; ctg++)
    tempSS += tempCtgSum[ctg] * tempCtgSum[ctg];

  ctgNux = CtgNux(tempCtgSum, tempSS);
  runSum = tempSum;

  return nuxSampled;
}


vector<RunNux> RunAccum::slotReorder(const vector<RunNux>& runNux) {
  vector<RunNux> frOrdered(runNux.size());
  vector<PredictorT> idxRank = PQueue::depopulate<PredictorT>(&heapZero[0], frOrdered.size());

  for (PredictorT slot = 0; slot < frOrdered.size(); slot++) {
    frOrdered[idxRank[slot]] = runNux[slot];
  }

  return frOrdered;
}


vector<RunNux> RunAccumCtg::runsExplicit(const SplitNux& cand) {
  vector<RunNux> runNux(cand.getRunCount());
  PredictorT runIdx = 0;
  double* sumBase = initCtg(obsStart, runNux[runIdx], runIdx);
  for (IndexT obsIdx = obsStart + 1; obsIdx != obsEnd; obsIdx++) {
    if (!obsCell[obsIdx].ctgAccum(runNux[runIdx], sumBase)) {
      runNux[runIdx++].endRange(obsIdx - 1);
      sumBase = initCtg(obsIdx, runNux[runIdx], runIdx);
    }
  }
  runNux[runIdx].endRange(obsEnd-1); // Flushes remaining run.

  return runNux;
}


vector<RunNux> RunAccumCtg::runsImplicit(const SplitNux& cand) {
  vector<RunNux> runNux(cand.getRunCount());
  // Cut position yields the run index at which to place the residual.
  // Observation at this position must not marked as tied.
  SumCount scExplicit(sumCount);
  PredictorT runIdx = 0;
  PredictorT implicitSlot = runNux.size(); // Inattainable.
  if (cutResidual == obsStart)
    implicitSlot = runIdx++;
  double* sumBase = initCtg(obsStart, runNux[runIdx], runIdx);
  for (IndexT obsIdx = obsStart + 1; obsIdx != obsEnd; obsIdx++) {
    if (!obsCell[obsIdx].ctgAccum(runNux[runIdx], sumBase)) {
      runNux[runIdx].endRun(scExplicit, obsIdx-1);
      if (cutResidual == obsIdx) {
	implicitSlot = ++runIdx;
      }
      runIdx++;
      sumBase = initCtg(obsIdx, runNux[runIdx], runIdx);
    }
  }
  runNux[runIdx].endRun(scExplicit, obsEnd-1);
  if (cutResidual == obsEnd)
    implicitSlot = ++runIdx;

  residualSums(runNux, implicitSlot);

  runNux[implicitSlot].setResidual(scExplicit, obsEnd, implicitCand);

  return runNux;
}


void RunAccumCtg::residualSums(const vector<RunNux>& runNux,
			   PredictorT implicitSlot) {
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


double* RunAccumCtg::initCtg(IndexT obsLeft,
			     RunNux& nux,
			     PredictorT runIdx) {
  nux.startRange(obsLeft);
  double* sumBase = &runSum[runIdx * nCtg];
  obsCell[obsLeft].ctgInit(nux, sumBase);
  return sumBase;
}


void RunAccumReg::split(const SFReg* sfReg, RunSet* runSet, SplitNux& cand) {
  RunAccumReg runAccum(sfReg, cand);
  vector<RunNux> runNux = runAccum.initRuns(runSet, cand);
  SplitRun splitRun = runAccum.split(runNux);
  runSet->setSplit(cand, std::move(runNux), splitRun);
}


void RunAccumCtg::split(const SFCtg* sfCtg, RunSet* runSet, SplitNux& cand) {
  RunAccumCtg runAccum(sfCtg, cand);
  vector<RunNux> runNux = runAccum.initRuns(runSet, cand);
  SplitRun splitRun = runAccum.split(runNux);
  runSet->setSplit(cand, std::move(runNux), splitRun);
}


vector<RunNux> RunAccum::initRuns(RunSet* runSet,
				  const SplitNux& cand) {
  vector<RunNux> runNux = regRuns(cand);
  info = (sumCount.sum * sumCount.sum) / sumCount.sCount;
  return runNux;
};


vector<RunNux> RunAccumCtg::initRuns(RunSet* runSet,
				     const SplitNux& cand) {
  vector<RunNux> runNux = ctgRuns(runSet, cand);
  info = ctgNux.sumSquares / sumCount.sum;
  return runNux;
};


SplitRun RunAccumReg::split(const vector<RunNux>& runNux) {
  return maxVar(runNux);
}


SplitRun RunAccumCtg::split(const vector<RunNux>& runNux) {
  if (nCtg == 2) {
    return binaryGini(runNux);
  }
  else
    return ctgGini(runNux);
}


SplitRun RunAccum::maxVar(const vector<RunNux>& runNux) {
  double infoCell = info;
  SumCount scAccum;
  PredictorT runSlot = runNux.size() - 1;
  for (PredictorT slotTrial = 0; slotTrial < runNux.size() - 1; slotTrial++) {
    runNux[slotTrial].accum(scAccum);
    if (trialSplit(infoVar(scAccum, sumCount))) {
      runSlot = slotTrial;
    }
  }
  return SplitRun(info - infoCell, runSlot, runNux.size());
}


SplitRun RunAccumCtg::ctgGini(const vector<RunNux>& runNux) {
  double infoCell = info;
  // Run index subsets as binary-encoded unsigneds.
  PredictorT trueSlots = 0; // Slot offsets of codes taking true branch.

  // High bit unset, remainder set.
  PredictorT lowSet = (1ul << (sampleCount - 1)) - 1;

  // Arg-max over all nontrivial subsets, up to complement:
  for (unsigned int subset = 1; subset <= lowSet; subset++) {
    if (trialSplit(subsetGini(runNux, subset))) {
      trueSlots = subset;
    }
  }

  return SplitRun(info - infoCell, trueSlots, sampleCount);
}


double RunAccumCtg::subsetGini(const vector<RunNux>& runNux,
			       unsigned int subset) const {
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


SplitRun RunAccumCtg::binaryGini(const vector<RunNux>& runNux) {
  double infoCell = info;
  const double tot0 = ctgNux.ctgSum[0];
  const double tot1 = ctgNux.ctgSum[1];
  double sumL0 = 0.0; // Running left sum at category 0.
  double sumL1 = 0.0; // " " category 1.
  PredictorT argMaxRun = runNux.size() - 1;
  for (PredictorT runIdx = 0; runIdx != runNux.size() - 1; runIdx++) {
    if (accumBinary(runNux, runIdx, sumL0, sumL1)) { // Splitable
      // sumR, sumL magnitudes can be ignored if no large case/class weightings.
      FltVal sumL = sumL0 + sumL1;
      double ssL = sumL0 * sumL0 + sumL1 * sumL1;
      double ssR = (tot0 - sumL0) * (tot0 - sumL0) + (tot1 - sumL1) * (tot1 - sumL1);
      if (trialSplit(infoGini(ssL, ssR, sumL, sumCount.sum - sumL))) {
        argMaxRun = runIdx;
      }
    } 
  }
  return SplitRun(info - infoCell, argMaxRun, runNux.size());
}
