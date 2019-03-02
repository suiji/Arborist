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
                       unsigned int bufIdx_) :
  info(0.0),
  splitIdx(splitIdx_),
  predIdx(predIdx_),
  bufIdx(bufIdx_),
  lhSCount(0),
  lhImplicit(0) {
}


/**
   @brief Initializes field values known only following restaging.  Entry
   singletons should not reach here.

   @return void
 */
void SplitCand::initLate(const SplitNode *splitNode,
                         const Level *levelFront,
                         const IndexLevel *index,
                         unsigned int vecIdx,
                         unsigned int setIdx) {
  this->vecIdx = vecIdx,
  this->setIdx = setIdx;
  unsigned int extent = index->setCand(this);
  info = splitNode->getPrebias(splitIdx);
  implicit = levelFront->adjustDense(splitIdx, predIdx, idxStart, extent);
  idxEnd = idxStart + extent - 1; // Singletons invalid:  idxEnd < idxStart.
}

bool SplitCand::schedule(const SplitNode *splitNode,
                         const Level *levelFront,
                         const IndexLevel *index,
                         vector<unsigned int> &runCount,
                         vector<SplitCand> &sc2) {
  unsigned int rCount;
  if (levelFront->scheduleSplit(splitIdx, predIdx, rCount)) {
    initLate(splitNode, levelFront, index, sc2.size(), rCount > 1 ? runCount.size() : splitNode->getNoSet());
    if (rCount > 1) {
      runCount.push_back(rCount);
    }
    sc2.push_back(*this);
    return true;
  }
  return false;
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
  NumPersist(cand, spReg->denseRank(cand)),
  monoMode(spReg->getMonoMode(cand)),
  resid(cand->getImplicit() > 0 ? makeResidual(cand, spn) : nullptr) {
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
  NumPersist(cand, spCtg->denseRank(cand)),
  nCtg(spCtg->getNCtg()),
  resid(cand->getImplicit() > 0 ? makeResidual(cand, spn, spCtg) : nullptr),
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
  
  // Flushes the remaining run.  Also flushes the implicit run, if dense.
  //
  runSet->write(rkThis, sCountHeap, sumHeap, frEnd - idxStart + 1, idxStart);
  if (implicit > 0) {
    runSet->writeImplicit(spReg->denseRank(this), sCount, sum, implicit);
  }

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
    runSet->accumCtg(yCtg, ySum);
  }

  
  // Flushes remaining run.
  runSet->write(rkThis, sCountLoc, sumLoc, frEnd - idxStart + 1, idxStart);
  if (implicit > 0) {
    runSet->writeImplicit(spCtg->denseRank(this), sCount, sum, implicit, spCtg->getSumSlice(this));
  }
}


void SplitCand::splitRuns(SPCtg *spCtg) {
  RunSet *runSet = spCtg->rSet(setIdx);
  const double *ctgSum = spCtg->getSumSlice(this);
  const unsigned int slotSup = runSet->deWide() - 1;// Uses post-shrink value.
  unsigned int lhBits = 0;
  unsigned int leftFull = (1 << slotSup) - 1;

  // Nonempty subsets as binary-encoded unsigneds.
  for (unsigned int subset = 1; subset <= leftFull; subset++) {
    double sumL = 0.0;
    double ssL = 0.0;
    double ssR = 0.0;
    for (unsigned int yCtg = 0; yCtg < spCtg->getNCtg(); yCtg++) {
      double sumCtg = 0.0; // Sum at category 'yCtg' over subset slots.
      for (unsigned int slot = 0; slot < slotSup; slot++) {
	if ((subset & (1 << slot)) != 0) {
	  sumCtg += runSet->getSumCtg(slot, yCtg);
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

  const double* ctgSum = spCtg->getSumSlice(this);
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
  unsigned int cut = cand->getIdxEnd() + 1; // Unreachable position for cell.
  double sumTot = 0.0;
  unsigned int sCountTot = 0;
  for (int idx = static_cast<int>(cand->getIdxEnd()); idx >= static_cast<int>(cand->getIdxStart()); idx--) {
    unsigned int sampleCount, rkThis;
    FltVal ySum;
    rkThis = spn[idx].regFields(ySum, sampleCount);
    if (rkThis > rankDense) {
      cut = idx;
    }
    sCountTot += sampleCount;
    sumTot += ySum;
  }
  cutDense = cut;
  
  return make_shared<Residual>(cand, sumTot, sCountTot);
}

Residual::Residual(const SplitCand* cand,
                   double sumTot,
                   unsigned int sCountTot) :
  sum(cand->getSum() - sumTot),
  sCount(cand->getSCount() - sCountTot) {
}


shared_ptr<ResidualCtg>
NumPersistCtg::makeResidual(const SplitCand* cand,
                            const SampleRank spn[],
                            SPCtg* spCtg) {
  vector<double> ctgExpl(nCtg);
  ctgExpl.assign(spCtg->getSumSlice(cand), spCtg->getSumSlice(cand) + ctgExpl.size());

  unsigned int cut = cand->getIdxEnd() + 1; // Defaults to highest index.
  double sumTot = 0.0;
  unsigned int sCountTot = 0;
  for (int idx = static_cast<int>(cand->getIdxEnd()); idx >= static_cast<int>(cand->getIdxStart()); idx--) {
    unsigned int yCtg;
    FltVal ySum;
    unsigned int rkThis = spn[idx].ctgFields(ySum, sCountThis, yCtg);
    sCountTot += sCountThis;
    ctgExpl[yCtg] -= ySum;
    if (rkThis > rankDense) {
      cut = idx;
    }
    sumTot += ySum;
  }
  cutDense = cut;

  return make_shared<ResidualCtg>(cand, sumTot, sCountTot, ctgExpl);
}

ResidualCtg::ResidualCtg(const SplitCand *cand,
                         double sumTot,
                         unsigned int sCountTot,
                         const vector<double>& ctgExpl) :
  Residual(cand, sumTot, sCountTot),
  ctgImpl(ctgExpl) {
}
