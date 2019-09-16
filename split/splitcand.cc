// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitcand.cc

   @brief Methods to implement splitting candidates.

   @author Mark Seligman
 */

#include "splitcand.h"
#include "splitaccum.h"
#include "frontier.h"
#include "splitfrontier.h"
#include "sfcart.h" // Temporary.
#include "level.h"
#include "runset.h"
#include "samplenux.h"

SplitCand::SplitCand(const SplitFrontier* splitNode,
                     const Frontier* frontier,
                     const SplitCoord& splitCoord,
                     unsigned int bufIdx,
                     IndexT noSet) :
  sCount(frontier->getSCount(splitCoord.nodeIdx)),
  sum(frontier->getSum(splitCoord.nodeIdx)),
  splitNux(SplitNux(splitCoord, noSet, bufIdx, splitNode->getPrebias(splitCoord))) {
}


bool SplitCand::schedule(const Level* levelFront,
                         const Frontier* frontier,
                         vector<unsigned int>& runCount) {
  return levelFront->scheduleSplit(frontier, runCount, splitNux, implicitCount);
}


/**
   @brief  Regression splitting based on type:  numeric or factor.
 */
void SplitCand::split(const SFReg *spReg) {
  if (spReg->isFactor(splitNux.splitCoord)) {
    splitFac(spReg);
  }
  else {
    splitNum(spReg);
  }
}


/**
   @brief Categorical splitting based on type:  numeric or factor.
 */
void SplitCand::split(SFCtg *spCtg) {
  if (spCtg->isFactor(splitNux.splitCoord)) {
    splitFac(spCtg);
  }
  else {
    splitNum(spCtg);
  }
}


void SplitCand::splitFac(SFCtg *spCtg) {
  buildRuns(spCtg);

  if (spCtg->getNCtg() == 2) {
    splitBinary(spCtg);
  }
  else {
    splitRuns(spCtg);
  }
}


/**
   @brief Main entry for numerical split.
*/
void SplitCand::splitNum(const SFReg* spReg) {
  SampleRank* spn = spReg->getPredBase(this);
  SplitAccumReg numPersist(this, spn, spReg);
  numPersist.split(spReg, spn, this);
  writeNum(spReg, numPersist);
}


void SplitCand::splitNum(SFCtg* spCtg) {
  SampleRank* spn = spCtg->getPredBase(this);
  SplitAccumCtg numPersist(this, spn, spCtg);
  numPersist.split(spCtg, spn, this);
  writeNum(spCtg, numPersist);
}


void SplitCand::writeNum(const SplitFrontier* spNode,
                         const SplitAccum& accum) {
  splitNux.info = accum.info;
  if (infoGain(spNode)) {
    splitNux.rankRange.set(accum.rankLH, accum.rankRH - accum.rankLH);
    splitNux.lhSCount = accum.lhSCount;
    splitNux.lhImplicit = accum.lhDense() ? implicitCount : 0;
    splitNux.lhExtent = splitNux.lhImplicit + (accum.rhMin - getIdxStart());
  }
}


/**
   Regression runs always maintained by heap.
*/
void SplitCand::splitFac(const SFReg *spReg) {
  RunSet *runSet = spReg->rSet(splitNux.setIdx);
  SampleRank* spn = spReg->getPredBase(this);
  double sumHeap = 0.0;
  IndexT sCountHeap = 0;
  IndexT idxEnd = getIdxEnd();
  IndexT rkThis = spn[idxEnd].getRank();
  IndexT frEnd = idxEnd;
  for (int i = static_cast<int>(idxEnd); i >= static_cast<int>(splitNux.idxRange.getStart()); i--) {
    IndexT rkRight = rkThis;
    IndexT sampleCount;
    FltVal ySum;
    rkThis = spn[i].regFields(ySum, sampleCount);

    if (rkThis == rkRight) { // Same run:  counters accumulate.
      sumHeap += ySum;
      sCountHeap += sampleCount;
    }
    else { // New run:  flush accumulated counters and reset.
      runSet->write(rkRight, sCountHeap, sumHeap, frEnd - i, i+1);

      sumHeap = ySum;
      sCountHeap = sampleCount;
      frEnd = i;
    }
  }
  
  // Flushes the remaining run and implicit run, if dense.
  //
  runSet->write(rkThis, sCountHeap, sumHeap, frEnd - splitNux.idxRange.getStart() + 1, splitNux.idxRange.getStart());
  runSet->writeImplicit(this, spReg);

  PredictorT runSlot = heapSplit(runSet);
  writeSlots(spReg, runSet, runSlot);
}


PredictorT SplitCand::heapSplit(RunSet *runSet) {
  runSet->heapMean();
  runSet->dePop();

  IndexT sCountL = 0;
  double sumL = 0.0;
  PredictorT runSlot = runSet->getRunCount() - 1;
  for (PredictorT slotTrial = 0; slotTrial < runSet->getRunCount() - 1; slotTrial++) {
    runSet->sumAccum(slotTrial, sCountL, sumL);
    if (SplitAccumReg::infoSplit(sumL, sum - sumL, sCountL, sCount - sCountL, splitNux.info)) {
      runSlot = slotTrial;
    }
  }

  return runSlot;
}


void SplitCand::writeSlots(const SplitFrontier* splitNode,
                           RunSet* runSet,
                           PredictorT cutSlot) {
  if (infoGain(splitNode)) {
    splitNux.lhExtent = runSet->lHSlots(cutSlot, splitNux.lhSCount);
  }
}


bool SplitCand::infoGain(const SplitFrontier* splitNode) {
  splitNux.info -= splitNode->getPrebias(splitNux.splitCoord);
  return splitNux.info > 0.0;
}


void SplitCand::buildRuns(SFCtg *spCtg) const {
  SampleRank* spn = spCtg->getPredBase(this);
  double sumLoc = 0.0;
  IndexT sCountLoc = 0;
  IndexT idxEnd = getIdxEnd();
  IndexT rkThis = spn[idxEnd].getRank();
  auto runSet = spCtg->rSet(splitNux.setIdx);

  IndexT frEnd = idxEnd;
  for (int i = static_cast<int>(idxEnd); i >= static_cast<int>(getIdxStart()); i--) {
    IndexT rkRight = rkThis;
    PredictorT yCtg;
    IndexT sampleCount;
    FltVal ySum;
    rkThis = spn[i].ctgFields(ySum, sampleCount, yCtg);

    if (rkThis == rkRight) { // Current run's counters accumulate.
      sumLoc += ySum;
      sCountLoc += sampleCount;
    }
    else { // Flushes current run and resets counters for next run.
      runSet->write(rkRight, sCountLoc, sumLoc, frEnd - i, i + 1);

      sumLoc = ySum;
      sCountLoc = sampleCount;
      frEnd = i;
    }
    runSet->accumCtg(spCtg->getNCtg(), ySum, yCtg);
  }

  
  // Flushes remaining run and implicit blob, if any.
  runSet->write(rkThis, sCountLoc, sumLoc, frEnd - getIdxStart() + 1, getIdxStart());
  runSet->writeImplicit(this, spCtg, spCtg->getSumSlice(this));
}


void SplitCand::splitRuns(SFCtg *spCtg) {
  RunSet *runSet = spCtg->rSet(splitNux.setIdx);
  const vector<double> ctgSum(spCtg->getSumSlice(this));
  const PredictorT slotSup = runSet->deWide(ctgSum.size()) - 1;// Uses post-shrink value.
  PredictorT lhBits = 0;

  // Nonempty subsets as binary-encoded unsigneds.
  unsigned int leftFull = (1 << slotSup) - 1;
  for (unsigned int subset = 1; subset <= leftFull; subset++) {
    double sumL = 0.0;
    double ssL = 0.0;
    double ssR = 0.0;
    PredictorT yCtg = 0;
    for (auto nodeSum : ctgSum) {
      double slotSum = 0.0; // Sum at category 'yCtg' over subset slots.
      for (PredictorT slot = 0; slot < slotSup; slot++) {
	if ((subset & (1ul << slot)) != 0) {
	  slotSum += runSet->getSumCtg(slot, ctgSum.size(), yCtg);
	}
      }
      yCtg++;
      sumL += slotSum;
      ssL += slotSum * slotSum;
      ssR += (nodeSum - slotSum) * (nodeSum - slotSum);
    }
    if (SplitAccumCtg::infoSplit(ssL, ssR, sumL, sum - sumL, splitNux.info)) {
      lhBits = subset;
    }
  }

  writeBits(spCtg, lhBits);
}

void SplitCand::writeBits(const SplitFrontier* splitNode,
                          PredictorT lhBits) {
  if (infoGain(splitNode)) {
    RunSet *runSet = splitNode->rSet(splitNux.setIdx);
    splitNux.lhExtent = runSet->lHBits(lhBits, splitNux.lhSCount);
  }
}


void SplitCand::splitBinary(SFCtg *spCtg) {
  RunSet *runSet = spCtg->rSet(splitNux.setIdx);
  runSet->heapBinary();
  runSet->dePop();

  const vector<double> ctgSum(spCtg->getSumSlice(this));
  const double tot0 = ctgSum[0];
  const double tot1 = ctgSum[1];
  double sumL0 = 0.0; // Running left sum at category 0.
  double sumL1 = 0.0; // " " category 1.
  PredictorT runSlot = runSet->getRunCount() - 1;
  for (PredictorT slotTrial = 0; slotTrial < runSet->getRunCount() - 1; slotTrial++) {
    if (runSet->accumBinary(slotTrial, sumL0, sumL1)) { // Splitable
      // sumR, sumL magnitudes can be ignored if no large case/class weightings.
      FltVal sumL = sumL0 + sumL1;
      double ssL = sumL0 * sumL0 + sumL1 * sumL1;
      double ssR = (tot0 - sumL0) * (tot0 - sumL0) + (tot1 - sumL1) * (tot1 - sumL1);
      if (SplitAccumCtg::infoSplit(ssL, ssR, sumL, sum - sumL, splitNux.info)) {
        runSlot = slotTrial;
      }
    } 
  }

  writeSlots(spCtg, runSet, runSlot);
}
