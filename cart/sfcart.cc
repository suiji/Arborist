// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitfrontier.cc

   @brief Methods to implement CART-specific splitting of frontier.

   @author Mark Seligman
 */


#include "accumcart.h"
#include "frontier.h"
#include "sfcart.h"
#include "splitnux.h"
#include "runset.h"
#include "samplenux.h"
#include "obspart.h"
#include "callback.h"
#include "summaryframe.h"
#include "rankedframe.h"
#include "sample.h"
#include "ompthread.h"

// Post-split consumption:
#include "pretree.h"


SFCart::SFCart(const class Cand* cand,
	       const SummaryFrame* frame,
	       Frontier* frontier,
	       const Sample* sample) :
  SplitFrontier(cand, frame, frontier, sample) {
}


unique_ptr<SplitFrontier>
SFCart::splitFactory(const class Cand* cand,
		     const SummaryFrame* frame,
		     Frontier* frontier,
		     const Sample* sample,
		     PredictorT nCtg) {
  if (nCtg > 0) {
    return make_unique<SFCartCtg>(cand, frame, frontier, sample, nCtg);
  }
  else {
    return make_unique<SFCartReg>(cand, frame, frontier, sample);
  }
}


vector<double> SFCartReg::mono; // Numeric monotonicity constraints.


void
SFCartReg::immutables(const SummaryFrame* frame,
		      const vector<double>& bridgeMono) {
  auto numFirst = frame->getNumFirst();
  auto numExtent = frame->getNPredNum();
  auto monoCount = count_if(bridgeMono.begin() + numFirst, bridgeMono.begin() + numExtent, [] (double prob) { return prob != 0.0; });
  if (monoCount > 0) {
    mono = vector<double>(frame->getNPredNum());
    mono.assign(bridgeMono.begin() + frame->getNumFirst(), bridgeMono.begin() + frame->getNumFirst() + frame->getNPredNum());
  }
}


void SFCartReg::deImmutables() {
  mono.clear();
}


SFCartReg::SFCartReg(const class Cand* cand,
		     const SummaryFrame* frame,
		     Frontier* frontier,
		     const Sample* sample) :
  SFCart(cand, frame, frontier, sample),
  ruMono(vector<double>(0)) {
  run = make_unique<Run>(0, frame->getNRow());
}


/**
   @brief Constructor.
 */
SFCartCtg::SFCartCtg(const class Cand* cand,
		     const SummaryFrame* frame,
		     Frontier* frontier,
		     const Sample* sample,
		     PredictorT nCtg_):
  SFCart(cand, frame, frontier, sample),
  nCtg(nCtg_) {
  run = make_unique<Run>(nCtg, frame->getNRow());
}


/**
   @brief Sets quick lookup offets for Run object.

   @return void.
 */
void SFCartReg::setRunOffsets(const vector<unsigned int>& runCount) {
  run->offsetsReg(runCount);
}


/**
   @brief Sets quick lookup offsets for Run object.
 */
void SFCartCtg::setRunOffsets(const vector<unsigned int>& runCount) {
  run->offsetsCtg(runCount);
}


double SFCartCtg::getSumSquares(const SplitNux *cand) const {
  return sumSquares[cand->getNodeIdx()];
}


const vector<double>& SFCartCtg::getSumSlice(const SplitNux* cand) const {
  return ctgSum[cand->getNodeIdx()];
}


double* SFCartCtg::getAccumSlice(const SplitNux* cand) {
  return &ctgSumAccum[getNumIdx(cand->getPredIdx()) * nSplit * nCtg + cand->getNodeIdx() * nCtg];
}

/**
   @brief Run objects should not be deleted until after splits have been consumed.
 */
void SFCartReg::clear() {
  SplitFrontier::clear();
}


SFCartReg::~SFCartReg() {
}


SFCartCtg::~SFCartCtg() {
}


void SFCartCtg::clear() {
  SplitFrontier::clear();
}


/**
   @brief Sets layer-specific values for the subclass.
*/
void SFCartReg::layerPreset() {
  if (!mono.empty()) {
    ruMono = CallBack::rUnif(nSplit * mono.size());
  }
}


void SFCartCtg::layerPreset() {
  layerInitSumR(frame->getNPredNum());
  ctgSum = vector<vector<double> >(nSplit);

  sumSquares = frontier->sumsAndSquares(ctgSum);
}


void SFCartCtg::layerInitSumR(PredictorT nPredNum) {
  if (nPredNum > 0) {
    ctgSumAccum = vector<double>(nPredNum * nCtg * nSplit);
    fill(ctgSumAccum.begin(), ctgSumAccum.end(), 0.0);
  }
}


int SFCartReg::getMonoMode(const SplitNux* cand) const {
  if (mono.empty())
    return 0;

  PredictorT numIdx = getNumIdx(cand->getSplitCoord().predIdx);
  double monoProb = mono[numIdx];
  double prob = ruMono[cand->getSplitCoord().nodeIdx * mono.size() + numIdx];
  if (monoProb > 0 && prob < monoProb) {
    return 1;
  }
  else if (monoProb < 0 && prob < -monoProb) {
    return -1;
  }
  else {
    return 0;
  }
}


void
SFCart::split(vector<SplitNux>& sc) {
  OMPBound splitTop = sc.size();
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitPos = 0; splitPos < splitTop; splitPos++) {
      split(&sc[splitPos]);
    }
  }

  nuxMax = maxCandidates(sc);
}


void SFCartCtg::split(SplitNux* cand) {
  if (isFactor(cand->getSplitCoord())) {
      splitFac(cand);
  }
  else {
    splitNum(cand);
  }
}


void SFCartReg::split(SplitNux* cand) {
  if (isFactor(cand->getSplitCoord())) {
    splitFac(cand);
  }
  else {
    splitNum(cand);
  }
}

void SFCartReg::splitNum(SplitNux* cand) const {
  SampleRank* spn = getPredBase(cand);
  AccumCartReg numPersist(cand, spn, this);
  numPersist.split(this, spn, cand);
}


/**
   Regression runs always maintained by heap.
*/
void SFCartReg::splitFac(SplitNux* cand) const {
  RunSet *runSet = rSet(cand->getSetIdx());
  SampleRank* spn = getPredBase(cand);
  double sumHeap = 0.0;
  IndexT sCountHeap = 0;
  IndexT idxEnd = cand->getIdxEnd();
  IndexT rkThis = spn[idxEnd].getRank();
  IndexT frEnd = idxEnd;
  for (int i = static_cast<int>(idxEnd); i >= static_cast<int>(cand->getIdxStart()); i--) {
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
  runSet->write(rkThis, sCountHeap, sumHeap, frEnd - cand->getIdxStart() + 1, cand->getIdxStart());
  runSet->writeImplicit(cand, this);

  PredictorT runSlot = heapSplit(runSet, cand);
  cand->writeSlots(this, runSlot);
}


PredictorT SFCartReg::heapSplit(RunSet *runSet,
				SplitNux* cand) const {
  runSet->heapMean();
  runSet->dePop();

  const double sum = cand->getSum();
  const IndexT sCount = cand->getSCount();
  IndexT sCountL = 0;
  double sumL = 0.0;
  PredictorT runSlot = runSet->getRunCount() - 1;
  for (PredictorT slotTrial = 0; slotTrial < runSet->getRunCount() - 1; slotTrial++) {
    runSet->sumAccum(slotTrial, sCountL, sumL);
    if (AccumCartReg::infoSplit(sumL, sum - sumL, sCountL, sCount - sCountL, cand->refInfo())) {
      runSlot = slotTrial;
    }
  }

  return runSlot;
}


void SFCartCtg::splitNum(SplitNux* cand) {
  SampleRank* spn = getPredBase(cand);
  AccumCartCtg numPersist(cand, spn, this);
  numPersist.split(this, spn, cand);
}


void SFCartCtg::splitFac(SplitNux* cand) const {
  buildRuns(cand);

  if (nCtg == 2) {
    splitBinary(cand);
  }
  else {
    splitRuns(cand);
  }
}


void SFCartCtg::buildRuns(SplitNux* cand) const {
  SampleRank* spn = getPredBase(cand);
  double sumLoc = 0.0;
  IndexT sCountLoc = 0;
  IndexT idxEnd = cand->getIdxEnd();
  IndexT rkThis = spn[idxEnd].getRank();
  auto runSet = rSet(cand->getSetIdx());

  IndexT frEnd = idxEnd;
  for (int i = static_cast<int>(idxEnd); i >= static_cast<int>(cand->getIdxStart()); i--) {
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
    runSet->accumCtg(nCtg, ySum, yCtg);
  }

  
  // Flushes remaining run and implicit blob, if any.
  runSet->write(rkThis, sCountLoc, sumLoc, frEnd - cand->getIdxStart() + 1, cand->getIdxStart());
  runSet->writeImplicit(cand, this, getSumSlice(cand));
}


void SFCartCtg::splitBinary(SplitNux* cand) const {
  RunSet *runSet = rSet(cand->getSetIdx());
  runSet->heapBinary();
  runSet->dePop();

  const vector<double> ctgSum(getSumSlice(cand));
  const double tot0 = ctgSum[0];
  const double tot1 = ctgSum[1];
  double sum = cand->getSum();
  double sumL0 = 0.0; // Running left sum at category 0.
  double sumL1 = 0.0; // " " category 1.
  PredictorT runSlot = runSet->getRunCount() - 1;
  for (PredictorT slotTrial = 0; slotTrial < runSet->getRunCount() - 1; slotTrial++) {
    if (runSet->accumBinary(slotTrial, sumL0, sumL1)) { // Splitable
      // sumR, sumL magnitudes can be ignored if no large case/class weightings.
      FltVal sumL = sumL0 + sumL1;
      double ssL = sumL0 * sumL0 + sumL1 * sumL1;
      double ssR = (tot0 - sumL0) * (tot0 - sumL0) + (tot1 - sumL1) * (tot1 - sumL1);
      if (AccumCartCtg::infoSplit(ssL, ssR, sumL, sum - sumL, cand->refInfo())) {
        runSlot = slotTrial;
      }
    } 
  }

  cand->writeSlots(this, runSlot);
}


void SFCartCtg::splitRuns(SplitNux* cand) const {
  RunSet *runSet = rSet(cand->getSetIdx());
  const vector<double> ctgSum(getSumSlice(cand));
  const PredictorT slotSup = runSet->deWide(ctgSum.size()) - 1;// Uses post-shrink value.
  PredictorT lhBits = 0;

  // Nonempty subsets as binary-encoded unsigneds.
  double sum = cand->getSum();
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
    if (AccumCartCtg::infoSplit(ssL, ssR, sumL, sum - sumL, cand->refInfo())) {
      lhBits = subset;
    }
  }

  cand->writeBits(this, lhBits);
}
