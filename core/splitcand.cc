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
#include "index.h"
#include "splitnode.h"
#include "level.h"
#include "runset.h"
#include "samplenux.h"
#include "samplepred.h"

double SplitCand::minRatio = minRatioDefault;

void SplitCand::immutables(double minRatio) {
  SplitCand::minRatio = minRatio;
}

void SplitCand::deImmutables() {
  minRatio = minRatioDefault;
}


SplitCand::SplitCand(const SplitCoord& splitCoord_,
                     unsigned int bufIdx_,
                     unsigned int noSet) :
  splitCoord(splitCoord_),
  info(0.0),
  setIdx(noSet),
  bufIdx(bufIdx_),
  lhSCount(0),
  lhImplicit(0) {
}


bool SplitCand::schedule(const SplitNode* splitNode,
                         const Level* levelFront,
                         const IndexLevel* iLevel,
                         vector<unsigned int>& runCount) {
  unsigned int rCount;
  if (levelFront->scheduleSplit(splitCoord, rCount)) {
    initLate(splitNode, levelFront, iLevel, runCount, rCount);
    return true;
  }
  return false;
}


void SplitCand::initLate(const SplitNode *splitNode,
                         const Level* levelFront,
                         const IndexLevel* ilevel,
                         vector<unsigned int>& runCount,
                         unsigned int rCount) {
  if (rCount > 1) {
    setIdx = runCount.size();
    runCount.push_back(rCount);
  }
  info = splitNode->getPrebias(splitCoord);
  indexInit(levelFront, ilevel);
}


void SplitCand::indexInit(const Level* levelFront, const IndexLevel* iLevel) {
  IndexSet iSet = iLevel->getISet(splitCoord);
  sCount = iSet.getSCount();
  sum = iSet.getSum();

  IndexRange idxRange(levelFront->adjustRange(splitCoord, iSet, implicit));
  idxStart = idxRange.getStart();
  idxEnd = idxRange.getEnd() - 1;// Singletons invalid:  idxEnd < idxStart.
}


/**
   @brief  Regression splitting based on type:  numeric or factor.
 */
void SplitCand::split(const SPReg *spReg,
                      const SamplePred *samplePred) {
  if (spReg->isFactor(splitCoord)) {
    splitFac(spReg, samplePred->PredBase(splitCoord, bufIdx));
  }
  else {
    splitNum(spReg, samplePred->PredBase(splitCoord, bufIdx));
  }
}


/**
   @brief Categorical splitting based on type:  numeric or factor.
 */
void SplitCand::split(SPCtg *spCtg,
                      const SamplePred *samplePred) {
  if (spCtg->isFactor(splitCoord)) {
    splitFac(spCtg, samplePred->PredBase(splitCoord, bufIdx));
  }
  else {
    splitNum(spCtg, samplePred->PredBase(splitCoord, bufIdx));
  }
}


void SplitCand::splitFac(SPCtg *spCtg,
                         const SampleRank spn[]) {
  buildRuns(spCtg, spn);

  if (spCtg->getNCtg() == 2) {
    splitBinary(spCtg);
  }
  else {
    splitRuns(spCtg);
  }
}


/**
   @brief Main entry for numerical split.

   @return void.
*/
void SplitCand::splitNum(const SPReg *spReg,
                         const SampleRank spn[]) {
  SplitAccumReg numPersist(this, spn, spReg);
  numPersist.split(spn, idxEnd, idxStart);
  numPersist.write(this);
}


void SplitCand::splitNum(SPCtg *spCtg,
                         const SampleRank spn[]) {
  SplitAccumCtg numPersist(this, spn, spCtg);
  numPersist.split(spn, idxEnd, idxStart);
  numPersist.write(this);
}


/**
   Regression runs always maintained by heap.
*/
void SplitCand::splitFac(const SPReg *spReg,
                         const SampleRank spn[]) {
  RunSet *runSet = spReg->rSet(setIdx);
  
  double sumHeap = 0.0;
  unsigned int sCountHeap = 0;
  unsigned int rkThis = spn[idxEnd].getRank();
  unsigned int frEnd = idxEnd;
  for (int i = static_cast<int>(idxEnd); i >= static_cast<int>(idxStart); i--) {
    unsigned int rkRight = rkThis;
    unsigned int sampleCount;
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
  runSet->write(rkThis, sCountHeap, sumHeap, frEnd - idxStart + 1, idxStart);
  runSet->writeImplicit(this, spReg);

  unsigned int runSlot = heapSplit(runSet);
  writeSlots(spReg, runSet, runSlot);
}


unsigned SplitCand::heapSplit(RunSet *runSet) {
  runSet->heapMean();
  runSet->dePop();

  unsigned int sCountL = 0;
  double sumL = 0.0;
  unsigned int runSlot = runSet->getRunCount() - 1;
  for (unsigned int slotTrial = 0; slotTrial < runSet->getRunCount() - 1; slotTrial++) {
    runSet->sumAccum(slotTrial, sCountL, sumL);
    if (SplitAccumReg::infoSplit(sumL, sum - sumL, sCountL, sCount - sCountL, info)) {
      runSlot = slotTrial;
    }
  }

  return runSlot;
}

void SplitCand::writeSlots(const SplitNode *splitNode,
                           RunSet *runSet,
                           unsigned int cut) {
  if (infoGain(splitNode)) {
    lhExtent = runSet->lHSlots(cut, lhSCount);
  }
}


bool SplitCand::infoGain(const SplitNode* splitNode) {
  info -= splitNode->getPrebias(splitCoord);
  return info > 0.0;
}



void SplitCand::buildRuns(SPCtg *spCtg,
                          const SampleRank spn[]) const {
  double sumLoc = 0.0;
  unsigned int sCountLoc = 0;
  unsigned int rkThis = spn[idxEnd].getRank();
  auto runSet = spCtg->rSet(setIdx);

  unsigned int frEnd = idxEnd;
  for (int i = static_cast<int>(idxEnd); i >= static_cast<int>(idxStart); i--) {
    unsigned int rkRight = rkThis;
    unsigned int yCtg, sampleCount;
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
  runSet->write(rkThis, sCountLoc, sumLoc, frEnd - idxStart + 1, idxStart);
  runSet->writeImplicit(this, spCtg, spCtg->getSumSlice(this));
}


void SplitCand::splitRuns(SPCtg *spCtg) {
  RunSet *runSet = spCtg->rSet(setIdx);
  const vector<double> ctgSum(spCtg->getSumSlice(this));
  const unsigned int slotSup = runSet->deWide(ctgSum.size()) - 1;// Uses post-shrink value.
  unsigned int lhBits = 0;

  // Nonempty subsets as binary-encoded unsigneds.
  unsigned int leftFull = (1 << slotSup) - 1;
  for (unsigned int subset = 1; subset <= leftFull; subset++) {
    double sumL = 0.0;
    double ssL = 0.0;
    double ssR = 0.0;
    unsigned int yCtg = 0;
    for (auto nodeSum : ctgSum) {
      double slotSum = 0.0; // Sum at category 'yCtg' over subset slots.
      for (unsigned int slot = 0; slot < slotSup; slot++) {
	if ((subset & (1ul << slot)) != 0) {
	  slotSum += runSet->getSumCtg(slot, ctgSum.size(), yCtg);
	}
      }
      yCtg++;
      sumL += slotSum;
      ssL += slotSum * slotSum;
      ssR += (nodeSum - slotSum) * (nodeSum - slotSum);
    }
    if (SplitAccumCtg::infoSplit(ssL, ssR, sumL, sum - sumL, info)) {
      lhBits = subset;
    }
  }

  writeBits(spCtg, lhBits);
}

void SplitCand::writeBits(const SplitNode* splitNode,
                          unsigned int lhBits) {
  if (infoGain(splitNode)) {
    RunSet *runSet = splitNode->rSet(setIdx);
    lhExtent = runSet->lHBits(lhBits, lhSCount);
  }
}


void SplitCand::splitBinary(SPCtg *spCtg) {
  RunSet *runSet = spCtg->rSet(setIdx);
  runSet->heapBinary();
  runSet->dePop();

  const vector<double> ctgSum(spCtg->getSumSlice(this));
  const double tot0 = ctgSum[0];
  const double tot1 = ctgSum[1];
  double sumL0 = 0.0; // Running left sum at category 0.
  double sumL1 = 0.0; // ibid., category 1.
  unsigned int runSlot = runSet->getRunCount() - 1;
  for (unsigned int slotTrial = 0; slotTrial < runSet->getRunCount() - 1; slotTrial++) {
    if (runSet->accumBinary(slotTrial, sumL0, sumL1)) { // Splitable
      // sumR, sumL magnitudes can be ignored if no large case/class weightings.
      FltVal sumL = sumL0 + sumL1;
      double ssL = sumL0 * sumL0 + sumL1 * sumL1;
      double ssR = (tot0 - sumL0) * (tot0 - sumL0) + (tot1 - sumL1) * (tot1 - sumL1);
      if (SplitAccumCtg::infoSplit(ssL, ssR, sumL, sum - sumL, info)) {
        runSlot = slotTrial;
      }
    } 
  }

  writeSlots(spCtg, runSet, runSlot);
}
