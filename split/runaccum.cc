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
#include "runset.h"

#include <numeric>

RunAccum::RunAccum(const SplitFrontier* splitFrontier,
		   const SplitNux& cand,
		   const RunSet* runSet) :
  Accum(splitFrontier, cand),
  heapZero(vector<BHPair<PredictorT>>((runSet->style == SplitStyle::slots || cand.getRunCount() > maxWidth) ? cand.getRunCount() : 0)) {
}


RunAccumReg::RunAccumReg(const SFReg* sfReg,
			 const SplitNux& cand,
			 const RunSet* runSet) : RunAccum(sfReg, cand, runSet) {
}


RunAccumCtg::RunAccumCtg(const SFCtg* sfCtg,
			 const SplitNux& cand,
			 const RunSet* runSet) : RunAccum(sfCtg, cand, runSet),
						 nCtg(sfCtg->getNCtg()),
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
void RunAccum::regRuns(RunSet* runSet,
		       const SplitNux& cand) {
  vector<RunNux> runNux;
  if (implicitCand) {
    runNux = regRunsImplicit(cand);
  }
  else {
    runNux = regRunsExplicit(cand);
  }

  runSet->setRuns(cand, orderMean(runNux));
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
			     IndexT edgeRight,
			     IndexT edgeLeft,
			     bool maskSense) {
  vector<RunNux> runNux(cand.getRunCount());
  SumCount scExplicit(sumCount);
  PredictorT runIdx = 0;
  initReg(edgeLeft, runNux[runIdx]);
  IndexT runRight = edgeLeft; // Previous unmasked index.
  for (IndexT idx = edgeLeft + 1; idx <= edgeRight; idx++) {
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
    PQueue::insert<PredictorT>(&heapZero[0], runNux[slot].sumCount.sum / runNux[slot].sumCount.sCount, slot);
  }
}


void RunAccumCtg::ctgRuns(RunSet* runSet, const SplitNux& cand) {
  vector<RunNux> runNux;
  if (implicitCand)
    runNux = runsImplicit(cand);
  else
    runNux = runsExplicit(cand);

  if (nCtg > 2) {
    if (runNux.size() > maxWidth)
      runNux = sampleRuns(runSet, cand, runNux);
  }
  else
    runNux = orderBinary(runNux);

  runSet->setRuns(cand, move(runNux));
}


vector<RunNux> RunAccumCtg::orderBinary(const vector<RunNux>& runNux) {
  heapBinary(runNux);
  return slotReorder(runNux);
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

  residCtg(runNux, implicitSlot);

  runNux[implicitSlot].setResidual(scExplicit, obsEnd, implicitCand);

  return runNux;
}


vector<RunNux> RunAccumCtg::sampleRuns(const RunSet* runSet,
				       const SplitNux& cand,
				       const vector<RunNux>& runNux) {
  vector<PredictorT> runIdx(runNux.size());
  iota(runIdx.begin(), runIdx.end(), 0);
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
  }

  double tempSS = 0;
  for (PredictorT ctg = 0; ctg < nCtg; ctg++)
    tempSS += tempCtgSum[ctg] * tempCtgSum[ctg];

  ctgNux = CtgNux(tempCtgSum, tempSS);
  runSum = tempSum;

  return rvNux;
}


double* RunAccumCtg::initCtg(IndexT obsLeft,
			     RunNux& nux,
			     PredictorT runIdx) {
  nux.startRange(obsLeft);
  double* sumBase = &runSum[runIdx * nCtg];
  obsCell[obsLeft].ctgInit(nux, sumBase);
  return sumBase;
}


void RunAccumCtg::residCtg(const vector<RunNux>& runNux,
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


void RunAccumReg::split(const SFReg* sfReg, RunSet* runSet, SplitNux& cand) {
  RunAccumReg runAccum(sfReg, cand, runSet);
  runAccum.initRuns(runSet, cand);
  cand.setInfo(runAccum.split(runSet, cand));
  runSet->setToken(cand, runAccum.splitToken);
}


void RunAccumCtg::split(const SFCtg* sfCtg, RunSet* runSet, SplitNux& cand) {
  RunAccumCtg runAccum(sfCtg, cand, runSet);
  runAccum.initRuns(runSet, cand);
  cand.setInfo(runAccum.split(runSet, cand));
  runSet->setToken(cand, runAccum.splitToken);
}


void RunAccum::initRuns(RunSet* runSet,
			const SplitNux& cand) {
  regRuns(runSet, cand);
  info = (sumCount.sum * sumCount.sum) / sumCount.sCount;
};


void RunAccumCtg::initRuns(RunSet* runSet,
			   const SplitNux& cand) {
  ctgRuns(runSet, cand);
  info = ctgNux.sumSquares / sumCount.sum;
};


double RunAccumReg::split(const RunSet* runSet,
			  const SplitNux& cand) {
  return maxVar(runSet->getRunNux(cand));
}


double RunAccumCtg::split(const RunSet* runSet,
			  const SplitNux& cand) {
  if (nCtg == 2)
    return binaryGini(runSet->getRunNux(cand));
  else
    return ctgGini(runSet->getRunNux(cand));
}


double RunAccum::maxVar(const vector<RunNux>& runNux) {
  double infoCell = info;
  SumCount scAccum;
  PredictorT runSlot = runNux.size() - 1;
  for (PredictorT slotTrial = 0; slotTrial < runNux.size() - 1; slotTrial++) {
    runNux[slotTrial].accum(scAccum);//sumAccum(slotTrial, scAccum);
    if (trialSplit(infoVar(scAccum, sumCount))) {
      runSlot = slotTrial;
    }
  }
  setToken(runSlot);
  return info - infoCell;
}


double RunAccumCtg::ctgGini(const vector<RunNux>& runNux) {
  double infoCell = info;
  // Run index subsets as binary-encoded unsigneds.
  PredictorT trueSlots = 0; // Slot offsets of codes taking true branch.

  // High bit unset, remainder set.
  PredictorT lowSet = (1ul << (runNux.size() - 1)) - 1;

  // Arg-max over all nontrivial subsets, up to complement:
  for (unsigned int subset = 1; subset <= lowSet; subset++) {
    if (trialSplit(subsetGini(runNux, subset))) {
      trueSlots = subset;
    }
  }

  setToken(trueSlots);
  return info - infoCell;
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


double RunAccumCtg::binaryGini(const vector<RunNux>& runNux) {
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
  setToken(argMaxRun);
  return info - infoCell;
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
