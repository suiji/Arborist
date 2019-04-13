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


SplitCand::SplitCand(unsigned int splitIdx_,
                     unsigned int predIdx_,
                     unsigned int bufIdx_,
                     unsigned int noSet) :
  info(0.0),
  splitIdx(splitIdx_),
  predIdx(predIdx_),
  setIdx(noSet),
  bufIdx(bufIdx_),
  lhSCount(0),
  lhImplicit(0) {
}


bool SplitCand::schedule(const SplitNode* splitNode,
                         const Level* levelFront,
                         const IndexLevel* index,
                         vector<unsigned int>& runCount) {
  unsigned int rCount;
  if (levelFront->scheduleSplit(splitIdx, predIdx, rCount)) {
    initLate(splitNode, levelFront, index->getISet(splitIdx), runCount, rCount);
    return true;
  }
  return false;
}


/**
   @brief Initializes field values known only following restaging.  Entry
   singletons should not reach here.

   @return void
 */
void SplitCand::initLate(const SplitNode *splitNode,
                         const Level *levelFront,
                         const IndexSet& iSet,
                         vector<unsigned int>& runCount,
                         unsigned int rCount) {
  if (rCount > 1) {
    setIdx = runCount.size();
    runCount.push_back(rCount);
  }
  info = splitNode->getPrebias(splitIdx);
  indexInit(levelFront, iSet);
}


void SplitCand::indexInit(const Level* levelFront, const IndexSet &iSet) {
  idxStart = iSet.getStart();
  sCount = iSet.getSCount();
  sum = iSet.getSum();

  unsigned int extent = iSet.getExtent();
  implicit = levelFront->adjustDense(splitIdx, predIdx, idxStart, extent);
  idxEnd = idxStart + extent - 1; // Singletons invalid:  idxEnd < idxStart.
}


/**
   @brief  Regression splitting based on type:  numeric or factor.
 */
void SplitCand::split(const SPReg *spReg,
                      const SamplePred *samplePred) {
  if (spReg->isFactor(predIdx)) {
    splitFac(spReg, samplePred->PredBase(predIdx, bufIdx));
  }
  else {
    splitNum(spReg, samplePred->PredBase(predIdx, bufIdx));
  }
}


/**
   @brief Categorical splitting based on type:  numeric or factor.
 */
void SplitCand::split(SPCtg *spCtg,
                      const SamplePred *samplePred) {
  if (spCtg->isFactor(predIdx)) {
    splitFac(spCtg, samplePred->PredBase(predIdx, bufIdx));
  }
  else {
    splitNum(spCtg, samplePred->PredBase(predIdx, bufIdx));
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
  NumPersistReg numPersist(this, spn, spReg);
  numPersist.split(spn, idxEnd, idxStart);
  numPersist.write(this);
}


NumPersist::NumPersist(const SplitCand* cand,
                       unsigned int rankDense_) :
  sCount(cand->getSCount()),
  sum(cand->getSum()),
  rankDense(rankDense_),
  sCountL(sCount),
  sumL(sum),
  cutDense(cand->getIdxEnd() + 1),
  info(cand->getInfo()) {
}

NumPersistReg::NumPersistReg(const SplitCand* cand,
                             const SampleRank spn[],
                             const SPReg* spReg) :
  NumPersist(cand, spReg->getDenseRank(cand)),
  monoMode(spReg->getMonoMode(cand)),
  resid(makeResidual(cand, spn)) {
}

void NumPersistReg::split(const SampleRank spn[],
                          unsigned int idxEnd,
                          unsigned int idxStart) {
  if (resid != nullptr) {
    splitImpl(spn, idxEnd, idxStart);
  }
  else {
    unsigned int rkThis = spn[idxEnd].regFields(ySum, sCountThis);
    splitExpl(spn, rkThis, idxEnd-1, idxStart);
  }
}

void NumPersistReg::splitImpl(const SampleRank spn[],
                              unsigned int idxEnd,
                              unsigned int idxStart) {
  if (cutDense > idxEnd) {
    // Checks resid/idxEnd, ..., idxStart+1/idxStart.
    resid->apply(ySum, sCountThis);
    splitExpl(spn, rankDense, idxEnd, idxStart);
  }
  else {
    // Checks idxEnd/idxEnd-1, ..., denseCut+1/denseCut.
    unsigned int rkThis = spn[idxEnd].regFields(ySum, sCountThis);
    splitExpl(spn, rkThis, idxEnd-1, cutDense);
    leftResidual(spn[cutDense].getRank()); // Checks denseCut/resid.

    // Checks resid/denseCut-1, ..., idxStart+1/idxStart, if applicable.
    if (cutDense > 0) {
      splitExpl(spn, rankDense, cutDense - 1, idxStart);
    }
  }
}


void NumPersistReg::leftResidual(unsigned int rkThis) {
  // Rank exposed from previous invocation of splitExpl():
  sumL -= ySum;
  sCountL -= sCountThis;
  resid->apply(ySum, sCountThis);

  unsigned int sCountR = sCount - sCountL;
  double sumR = sum - sumL;
  double infoTrial = infoSplit(sumL, sumR, sCountL, sCountR);
  if (infoTrial > info) {
    bool up = (sumL * sCountR <= sumR * sCountL);
    if (monoMode == 0 || (monoMode >0 && up) || (monoMode < 0 && !up)) {
      lhSCount = sCountL;
      rankRH = rkThis;
      rankLH = rankDense;
      rhMin = cutDense;
      info = infoTrial;
    }
  }
}


void NumPersistReg::splitExpl(const SampleRank spn[],
                              unsigned int rkThis,
                              unsigned int idxInit,
                              unsigned int idxFinal) {
  // Per-sample monotonicity constraint confined to specialized method:
  if (monoMode != 0) {
    splitMono(spn, rkThis, idxInit, idxFinal);
    return;
  }

  for (int idx = static_cast<int>(idxInit); idx >= static_cast<int>(idxFinal); idx--) {
    unsigned int rkRight = rkThis;
    sumL -= ySum;
    sCountL -= sCountThis;
    rkThis = spn[idx].regFields(ySum, sCountThis);

    double infoTrial = infoSplit(sumL, sum - sumL, sCountL, sCount - sCountL);
    if (infoTrial > info && rkThis != rkRight) {
      info = infoTrial;
      lhSCount = sCountL;
      rankRH = rkRight;
      rankLH = rkThis;
      rhMin = rkRight == rankDense ? cutDense : idx + 1;
    }
  }
}

/**
   @brief As above, but checks monotonicity at every index.
 */
void NumPersistReg::splitMono(const SampleRank spn[],
                              unsigned int rkThis,
                              unsigned int idxInit,
                              unsigned int idxFinal) {
  bool nonDecreasing = monoMode > 0;
  for (int idx = static_cast<int>(idxInit); idx >= static_cast<int>(idxFinal); idx--) {
    unsigned int rkRight = rkThis;
    sumL -= ySum;
    sCountL -= sCountThis;
    rkThis = spn[idx].regFields(ySum, sCountThis);

    //    localMax(nonDecreasing);
    unsigned int sCountR = sCount - sCountL;
    double sumR = sum - sumL;
    double infoTrial = infoSplit(sumL, sumR, sCountL, sCountR);
    if (infoTrial > info && rkThis != rkRight) {
      bool up = (sumL * sCountR <= sumR * sCountL);
      if (nonDecreasing ? up : !up) {
        info = infoTrial;
        lhSCount = sCountL;
        rankRH = rkRight;
        rankLH = rkThis;
        rhMin = rkRight == rankDense ? cutDense : idx + 1;
      }
    }
  }
}


void NumPersist::write(SplitCand* cand) {
  cand->writeNum(info, lhSCount, rankLH, rankRH, rankDense <= rankLH, rhMin);
}

void SplitCand::writeNum(double splitInfo,
                         unsigned int lhSCount,
                         unsigned int rankLH,
                         unsigned int rankRH,
                         bool lhDense,
                         unsigned int rhMin) {
  info = splitInfo - info;
  if (info > 0.0) {
    rankRange.set(rankLH, rankRH);
    this->lhSCount = lhSCount;
    lhImplicit = lhDense ? implicit : 0;
    lhExtent = lhImplicit + (rhMin - idxStart);
  }
}


void SplitCand::splitNum(SPCtg *spCtg,
                         const SampleRank spn[]) {
  NumPersistCtg numPersist(this, spn, spCtg);
  numPersist.split(spn, idxEnd, idxStart);
  numPersist.write(this);
}

NumPersistCtg::NumPersistCtg(const SplitCand* cand,
                             const SampleRank spn[],
                             SPCtg* spCtg) :
  NumPersist(cand, spCtg->getDenseRank(cand)),
  nCtg(spCtg->getNCtg()),
  resid(makeResidual(cand, spn, spCtg)),
  ctgSum(spCtg->getSumSlice(cand)),
  ctgAccum(spCtg->getAccumSlice(cand)),
  ssL(spCtg->getSumSquares(cand)),
  ssR(0.0) {
}


// Initializes from final index and loops over remaining indices.
void NumPersistCtg::split(const SampleRank spn[],
                          unsigned int idxEnd,
                          unsigned int idxStart) {
  if (resid != nullptr) {
    splitImpl(spn, idxEnd, idxStart);
  }
  else {
    unsigned int rkThis = stateNext(spn, idxEnd);
    splitExpl(spn, rkThis, idxEnd-1, idxStart);
  }
}


inline unsigned int NumPersistCtg::stateNext(const SampleRank spn[],
                       unsigned int idx) {
  unsigned int yCtg;
  unsigned int rkThis = spn[idx].ctgFields(ySum, sCountThis, yCtg);

  sumL -= ySum;
  sCountL -= sCountThis;
  double sumRCtg = accumCtgSum(yCtg, ySum);
  ssR += ySum * (ySum + 2.0 * sumRCtg);
  double sumLCtg = ctgSum[yCtg] - sumRCtg;
  ssL += ySum * (ySum - 2.0 * sumLCtg);

  return rkThis;
}

// Initializes from final index and loops over remaining indices.
void NumPersistCtg::splitExpl(const SampleRank spn[],
                              unsigned int rkThis,
                              unsigned int idxInit,
                              unsigned int idxFinal) {
  for (int idx = static_cast<int>(idxInit); idx >= static_cast<int>(idxFinal); idx--) {
    // Applies upward-exposed or wraparound state:
    unsigned int rkRight = rkThis;
    rkThis = spn[idx].getRank();

    double infoTrial = infoSplit(ssL, ssR, sumL, sum - sumL);
    if (infoTrial > info && rkThis != rkRight) {
      info = infoTrial;
      lhSCount = sCountL;
      rankRH = rkRight;
      rankLH = rkThis;
      rhMin = rkRight == rankDense ? cutDense : idx + 1;
    }
    (void) stateNext(spn, idx);
  }
}


void NumPersistCtg::splitImpl(const SampleRank spn[],
                              unsigned int idxStart,
                              unsigned int idxEnd) {
  if (cutDense > idxEnd) {
    resid->apply(ySum, sCountThis, ssR, ssL, this);
    splitExpl(spn, rankDense, idxEnd, idxStart);
  }
  else {
    unsigned int rkThis = stateNext(spn, idxEnd);
    splitExpl(spn, rkThis, idxEnd-1, cutDense);
    resid->apply(ySum, sCountThis, ssR, ssL, this);
    if (cutDense > 0) {
      splitExpl(spn, rankDense, cutDense - 1, idxStart);
    }
  }
}


void ResidualCtg::apply(FltVal& ySum,
                        unsigned int& sCount,
                        double& ssR,
                        double& ssL,
                        NumPersistCtg* np) {
  ySum = this->sum;
  sCount = this->sCount;
  for (unsigned int ctg = 0; ctg < ctgImpl.size(); ctg++) {
    double sumCtg = ctgImpl[ctg];
    double sumRCtg = np->accumCtgSum(ctg, sumCtg);
    ssR += sumCtg * (sumCtg + 2.0 * sumRCtg);
    double sumLCtg = np->getCtgSum(ctg) - sumRCtg;
    ssL += sumCtg * (sumCtg - 2.0 * sumLCtg);
  }
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
    unsigned int sCountRun;
    sumL += runSet->sumHeap(slotTrial, sCountRun);
    sCountL += sCountRun;
    double infoTrial = NumPersistReg::infoSplit(sumL, sum - sumL, sCountL, sCount - sCountL);
    if (infoTrial > info) {
      info = infoTrial;
      runSlot = slotTrial;
    }
  }

  return runSlot;
}

void SplitCand::writeSlots(const SplitNode *splitNode,
                           RunSet *runSet,
                           unsigned int cut) {
  info -= splitNode->getPrebias(splitIdx);
  if (info > 0.0) {
    lhExtent = runSet->lHSlots(cut, lhSCount);
  }
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
  unsigned int leftFull = (1 << slotSup) - 1;

  // Nonempty subsets as binary-encoded unsigneds.
  for (unsigned int subset = 1; subset <= leftFull; subset++) {
    double sumL = 0.0;
    double ssL = 0.0;
    double ssR = 0.0;
    for (unsigned int yCtg = 0; yCtg < ctgSum.size(); yCtg++) {
      double sumCtg = 0.0; // Sum at category 'yCtg' over subset slots.
      for (unsigned int slot = 0; slot < slotSup; slot++) {
	if ((subset & (1ul << slot)) != 0) {
	  sumCtg += runSet->getSumCtg(slot, ctgSum.size(), yCtg);
	}
      }
      const double nodeSumCtg = ctgSum[yCtg];
      sumL += sumCtg;
      ssL += sumCtg * sumCtg;
      ssR += (nodeSumCtg - sumCtg) * (nodeSumCtg - sumCtg);
    }
    double infoTrial = NumPersistCtg::infoSplit(ssL, ssR, sumL, sum - sumL);
    if (infoTrial > info) {
      info = infoTrial;
      lhBits = subset;
    }
  }

  writeBits(spCtg, lhBits);
}

void SplitCand::writeBits(const SplitNode* splitNode,
                          unsigned int lhBits) {
  info -= splitNode->getPrebias(splitIdx);
  if (info > 0.0) {
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
      double infoTrial = NumPersistCtg::infoSplit(ssL, ssR, sumL, sum - sumL);
      if (infoTrial > info) {
        info = infoTrial;
        runSlot = slotTrial;
      }
    } 
  }

  writeSlots(spCtg, runSet, runSlot);
}


shared_ptr<Residual> NumPersistReg::makeResidual(const SplitCand* cand,
                                                 const SampleRank spn[]) {
  if (cand->getImplicit() == 0) {
    return nullptr;
  }
  double sumExpl = 0.0;
  unsigned int sCountExpl = 0;
  for (int idx = static_cast<int>(cand->getIdxEnd()); idx >= static_cast<int>(cand->getIdxStart()); idx--) {
    unsigned int rkThis = spn[idx].regFields(ySum, sCountThis);
    if (rkThis > rankDense) {
      cutDense = idx;
    }
    sCountExpl += sCountThis;
    sumExpl += ySum;
  }
  
  return make_shared<Residual>(sum - sumExpl, sCount - sCountExpl);
}

Residual::Residual(double sum_,
                   unsigned int sCount_) :
  sum(sum_),
  sCount(sCount_) {
}


shared_ptr<ResidualCtg>
NumPersistCtg::makeResidual(const SplitCand* cand,
                            const SampleRank spn[],
                            SPCtg* spCtg) {
  if (cand->getImplicit() == 0) {
    return nullptr;
  }

  vector<double> ctgImpl(spCtg->getSumSlice(cand));//nCtg);
  //  ctgImpl.assign(spCtg->getSumSlice(cand), spCtg->getSumSlice(cand) + ctgImpl.size());

  double sumExpl = 0.0;
  unsigned int sCountExpl = 0;
  for (int idx = static_cast<int>(cand->getIdxEnd()); idx >= static_cast<int>(cand->getIdxStart()); idx--) {
    unsigned int yCtg;
    unsigned int rkThis = spn[idx].ctgFields(ySum, sCountThis, yCtg);
    if (rkThis > rankDense) {
      cutDense = idx;
    }
    sCountExpl += sCountThis;
    ctgImpl[yCtg] -= ySum;
    sumExpl += ySum;
  }

  return make_shared<ResidualCtg>(sum - sumExpl, sCount - sCountExpl, ctgImpl);
}

ResidualCtg::ResidualCtg(double sum_,
                         unsigned int sCount_,
                         const vector<double>& ctgImpl_) :
  Residual(sum_, sCount_),
  ctgImpl(ctgImpl_) {
}
