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


#include "train.h"
#include "accumcart.h"
#include "frontier.h"
#include "sfcart.h"
#include "splitnux.h"
#include "runaccum.h"
#include "samplenux.h"
#include "callback.h"
#include "summaryframe.h"
#include "rankedframe.h"
#include "sample.h"
#include "ompthread.h"

// Post-split consumption:
#include "pretree.h"


SFCart::SFCart(const SummaryFrame* frame,
	       Frontier* frontier,
	       const Sample* sample) :
  SplitFrontier(frame, frontier, sample) {
}


unique_ptr<SplitFrontier>
SFCart::factory(const SummaryFrame* frame,
		Frontier* frontier,
		const Sample* sample,
		PredictorT nCtg) {
  if (nCtg > 0) {
    return make_unique<SFCartCtg>(frame, frontier, sample, nCtg);
  }
  else {
    return make_unique<SFCartReg>(frame, frontier, sample);
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


SFCartReg::SFCartReg(const SummaryFrame* frame,
		     Frontier* frontier,
		     const Sample* sample) :
  SFCart(frame, frontier, sample),
  ruMono(vector<double>(0)) {
}


/**
   @brief Constructor.
 */
SFCartCtg::SFCartCtg(const SummaryFrame* frame,
		     Frontier* frontier,
		     const Sample* sample,
		     PredictorT nCtg_):
  SFCart(frame, frontier, sample),
  nCtg(nCtg_) {
}


double SFCartCtg::getSumSquares(const SplitNux *cand) const {
  return sumSquares[cand->getNodeIdx()];
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


SplitStyle SFCartCtg::getFactorStyle() const {
  return nCtg == 2 ? SplitStyle::slots : SplitStyle::bits;
}

  
  /**
     @return enumeration indicating slot-style encoding.
   */
SplitStyle SFCartReg::getFactorStyle() const {
  return SplitStyle::slots;
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


void SFCart::split(vector<IndexSet>& indexSet,
		   vector<SplitNux>& sc,
		   class BranchSense* branchSense) {
  OMPBound splitTop = sc.size();
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitPos = 0; splitPos < splitTop; splitPos++) {
      split(&sc[splitPos]);
    }
  }

  nuxMax = maxCandidates(sc);
  for (auto & iSet : indexSet) {
    SplitNux* nux = nuxMax[iSet.getSplitIdx()].get();
    if (iSet.isInformative(nux)) {
      encodeCriterion(&iSet, nux, branchSense);
    }
  }
}


void SFCart::encodeCriterion(IndexSet* iSet,
			     SplitNux* nux,
			     BranchSense* branchSense) const {
  accumUpdate(nux);
  CritEncoding enc(this, nux, frontier->getNCtg(), true, false);
  nuxEncode(nux, branchSense, enc, false);
  iSet->updateTrue(this, nux, enc);
}


vector<unique_ptr<SplitNux> >
SFCart::maxCandidates(const vector<SplitNux>& sc) {
  vector<unique_ptr<SplitNux> > nuxMax(nSplit); // Info initialized to zero.

  OMPBound splitTop = nSplit;
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitIdx = 0; splitIdx < splitTop; splitIdx++) {
      nuxMax[splitIdx] = maxSplit(sc, candOff[splitIdx], nCand[splitIdx]);
    }
  }

  return nuxMax;
}


unique_ptr<SplitNux>
SFCart::maxSplit(const vector<SplitNux>& cand,
		 IndexT splitBase,
		 IndexT nCandSplit) const {
  IndexT argMax = splitBase + nCandSplit;
  double runningMax = 0.0;
  for (IndexT splitOff = splitBase; splitOff < splitBase + nCandSplit; splitOff++) {
    if (cand[splitOff].maxInfo(runningMax)) {
      argMax = splitOff;
    }
  }

  return runningMax > 0.0 ? make_unique<SplitNux>(cand[argMax]) : make_unique<SplitNux>();
}


void SFCart::consumeNodes(PreTree* pretree) const {
  for (auto & nux : nuxMax) {
    if (!nux->noNux()) {
      pretree->nonterminal(nux.get());
      consumeCriterion(pretree, nux.get());
    }
  }
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
  AccumCartReg numPersist(cand, this);
  numPersist.split(this, cand);
}


/**
   Regression runs always maintained by heap.
*/
void SFCartReg::splitFac(SplitNux* cand) const {
  RunAccum *runAccum = getRunAccum(cand->getAccumIdx());
  runAccum->regRuns(cand);
  splitMean(cand);
}


void SFCartReg::splitMean(SplitNux* cand) const {
  RunAccum *runAccum = getRunAccum(cand->getAccumIdx());
  runAccum->orderMean();

  const double sum = cand->getSum();
  const IndexT sCount = cand->getSCount();
  IndexT sCountL = 0;
  double sumL = 0.0;
  PredictorT runSlot = runAccum->getRunCount() - 1;
  for (PredictorT slotTrial = 0; slotTrial < runAccum->getRunCount() - 1; slotTrial++) {
    runAccum->sumAccum(slotTrial, sCountL, sumL);
    double infoTemp = Accum::infoVar(sumL, sum - sumL, sCountL, sCount - sCountL);
    if (AccumCartReg::infoSplit(infoTemp, cand->refInfo())) {
      runSlot = slotTrial;
    }
  }
  runAccum->setToken(runSlot);
  cand->infoGain(this);
}


void SFCartCtg::splitNum(SplitNux* cand) {
  AccumCartCtg numPersist(cand, this);
  numPersist.split(this, cand);
}


void SFCartCtg::splitFac(SplitNux* cand) const {
  RunAccum* runAccum = getRunAccum(cand->getAccumIdx());
  runAccum->ctgRuns(cand, nCtg, getSumSlice(cand));

  if (nCtg == 2) {
    splitBinary(cand);
  }
  else {
    splitRuns(cand);
  }
}


const vector<double>& SFCartCtg::getSumSlice(const SplitNux* cand) const {
  return ctgSum[cand->getNodeIdx()];
}


void SFCartCtg::splitBinary(SplitNux* cand) const {
  const vector<double> ctgSum(getSumSlice(cand));
  const double sum = cand->getSum();

  RunAccum *runAccum = getRunAccum(cand->getAccumIdx());
  runAccum->orderBinary();

  const double tot0 = ctgSum[0];
  const double tot1 = ctgSum[1];
  double sumL0 = 0.0; // Running left sum at category 0.
  double sumL1 = 0.0; // " " category 1.
  PredictorT runSlot = runAccum->getRunCount() - 1;
  for (PredictorT slotTrial = 0; slotTrial < runAccum->getRunCount() - 1; slotTrial++) {
    if (runAccum->accumBinary(slotTrial, sumL0, sumL1)) { // Splitable
      // sumR, sumL magnitudes can be ignored if no large case/class weightings.
      FltVal sumL = sumL0 + sumL1;
      double ssL = sumL0 * sumL0 + sumL1 * sumL1;
      double ssR = (tot0 - sumL0) * (tot0 - sumL0) + (tot1 - sumL1) * (tot1 - sumL1);
      double infoTemp = AccumCartCtg::infoGini(ssL, ssR, sumL, sum - sumL);
      if (AccumCartCtg::infoSplit(infoTemp, cand->refInfo())) {
        runSlot = slotTrial;
      }
    } 
  }
  runAccum->setToken(runSlot);
  cand->infoGain(this);
}


void SFCartCtg::splitRuns(SplitNux* cand) const {
  RunAccum *runAccum = getRunAccum(cand->getAccumIdx());
  const vector<double> ctgSum(getSumSlice(cand));
  PredictorT trueBits = 0;

  // Nonempty subsets as binary-encoded unsigneds.
  runAccum->deWide(ctgSum.size());
  double sum = cand->getSum();
  PredictorT nBits = runAccum->effCount() - 1; // Highest bit implicitly zero.
  unsigned int allHigh = (1ul << nBits) - 1; // Low-order 'nBits' all high.
  for (unsigned int subset = 1; subset <= allHigh; subset++) { // All nonzero subsets.
    double sumL = 0.0;
    double ssL = 0.0;
    double ssR = 0.0;
    PredictorT yCtg = 0;
    for (auto nodeSum : ctgSum) {
      double slotSum = 0.0; // Sum at category 'yCtg' over subset slots.
      for (PredictorT slot = 0; slot < nBits; slot++) {
	if ((subset & (1ul << slot)) != 0) {
	  slotSum += runAccum->getSumCtg(slot, ctgSum.size(), yCtg);
	}
      }
      yCtg++;
      sumL += slotSum;
      ssL += slotSum * slotSum;
      ssR += (nodeSum - slotSum) * (nodeSum - slotSum);
    }
    double infoTemp = AccumCartCtg::infoGini(ssL, ssR, sumL, sum - sumL);
    if (AccumCartCtg::infoSplit(infoTemp, cand->refInfo())) {
      trueBits = subset;
    }
  }
  runAccum->setToken(trueBits);
  cand->infoGain(this);
}
