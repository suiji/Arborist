// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitfrontier.cc

   @brief Methods to implement splitting of frontier.

   @author Mark Seligman
 */


#include "frontier.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "defmap.h"
#include "runset.h"
#include "cutset.h"
#include "samplenux.h"
#include "obspart.h"
#include "summaryframe.h"
#include "rankedframe.h"
#include "sample.h"
#include "ompthread.h"
#include "callback.h"

// Post-split consumption:
#include "pretree.h"


vector<double> SFReg::mono; // Numeric monotonicity constraints.


SplitFrontier::SplitFrontier(const SummaryFrame* frame_,
                             Frontier* frontier_,
                             const Sample* sample,
			     bool compoundCriteria_,
			     EncodingStyle encodingStyle_) :
  frame(frame_),
  rankedFrame(frame->getRankedFrame()),
  frontier(frontier_),
  nPred(frame->getNPred()),
  obsPart(sample->predictors()),
  compoundCriteria(compoundCriteria_),
  encodingStyle(encodingStyle_) {
}


SplitFrontier::~SplitFrontier() {
}


CritEncoding::CritEncoding(const SplitFrontier* sf, const SplitNux* nux, PredictorT nCtg, bool excl, bool incr) :
  sum(0.0), sCount(0), extent(0), scCtg(vector<SumCount>(nCtg)), implicitTrue(sf->getImplicitTrue(nux)), increment(incr), exclusive(excl), style(sf->getEncodingStyle()) {
}


IndexT CritEncoding::getSCountTrue(const SplitNux* nux) const {
  return implicitTrue == 0 ? sCount : (nux->getSCount() - sCount); 
}


double CritEncoding::getSumTrue(const SplitNux* nux) const {
  return implicitTrue == 0 ? sum : (nux->getSum() - sum);
}


IndexT CritEncoding::getExtentTrue(const SplitNux* nux) const {
  return implicitTrue == 0 ? extent : (implicitTrue + nux->getExtent() - extent);
}


IndexT* SplitFrontier::getBufferIndex(const SplitNux* nux) const {
  return obsPart->getBufferIndex(nux);
}


RunAccumT* SplitFrontier::getRunAccum(PredictorT accumIdx) const {
  return runSet->getAccumulator(accumIdx);
}


SampleRank* SplitFrontier::getPredBase(const SplitNux* nux) const {
  return obsPart->getPredBase(nux->getDefCoord());
}


IndexT SplitFrontier::getDenseRank(const SplitNux* nux) const {
  return rankedFrame->getDenseRank(nux->getPredIdx());
}


vector<DefCoord> SplitFrontier::precandidates(const DefMap* defMap) {
  return CandType::precandidates(this, defMap);
}


void SplitFrontier::preschedule(const DefCoord& defCoord,
				vector<DefCoord>& preCand) const {
  preCand.emplace_back(defCoord);
}


void SplitFrontier::setPrebias() {
  for (IndexT splitIdx = 0; splitIdx < nSplit; splitIdx++) {
    setPrebias(splitIdx, frontier->getSum(splitIdx), frontier->getSCount(splitIdx));
  }
}


bool SplitFrontier::isFactor(const SplitCoord& splitCoord) const {
  return frame->isFactor(splitCoord.predIdx);
}


PredictorT SplitFrontier::getNumIdx(PredictorT predIdx) const {
  return frame->getNumIdx(predIdx);
}


vector<StageCount> SplitFrontier::stage(const Sample* sample) {
  return sample->stage(obsPart.get());
}


void SplitFrontier::accumUpdate(const SplitNux* cand) const {
  if (cand->isFactor(frame)) { // Only factor accumulators currently require an update.
    runSet->updateAccum(cand);
  }
}


CritEncoding SplitFrontier::nuxEncode(const SplitNux* nux,
			      BranchSense* branchSense,
			      const IndexRange& range) const {
  CritEncoding enc(this, nux, frontier->getNCtg(), compoundCriteria);
  if (!range.empty()) {
    obsPart->branchUpdate(nux, range, branchSense, enc);
  }
  else {
    if (nux->isFactor(frame)) {
      if (getFactorStyle() == SplitStyle::topSlot) {
	obsPart->branchUpdate(nux, runSet->getTopRange(nux, enc), branchSense, enc);
      }
      else {
	obsPart->branchUpdate(nux, runSet->getRange(nux, enc), branchSense, enc);
      }
    }
    else {
      obsPart->branchUpdate(nux, getCutRange(nux, enc), branchSense, enc);
    }
  }

  return enc;
}


void SplitFrontier::encodeCriterion(IndexSet* iSet,
			     SplitNux* nux,
			     BranchSense* branchSense) const {
  accumUpdate(nux);
  CritEncoding enc = nuxEncode(nux, branchSense);
  iSet->update(this, nux, enc);
}


IndexRange SplitFrontier::getCutRange(const SplitNux* nux, const CritEncoding& enc) const {
// Returns the left range iff (BOTH left cut AND true encoding or NEITHER left cut NOR true encoding).
  return nux->cutRange(cutSet.get(), !(leftCut(nux) ^ enc.trueEncoding()));
}



void CritEncoding::getISetVals(const SplitNux* nux,
			 IndexT& sCountTrue,
			 double& sumTrue,
			 IndexT& extentTrue) const {
  style == EncodingStyle::direct ? accumDirect(sCountTrue, sumTrue, extentTrue) : accumTrue(nux, sCountTrue, sumTrue, extentTrue);
}


void CritEncoding::accumDirect(IndexT& sCountTrue,
			      double& sumTrue,
			      IndexT& extentTrue) const {
  if (increment) {
    sCountTrue += sCount;
    extentTrue += extent;
    sumTrue += sum;
  }
  else {
    sCountTrue -= sCount;
    extentTrue -= extent;
    sumTrue -= sum;
  }
}


void CritEncoding::accumTrue(const SplitNux* nux,
			     IndexT& sCountTrue,
			     double& sumTrue,
			     IndexT& extentTrue) const {
  if (increment) {
    sCountTrue += getSCountTrue(nux);
    sumTrue += getSumTrue(nux);
    extentTrue += getExtentTrue(nux);
  }
  else {
    sCountTrue -= getSCountTrue(nux);
    sumTrue -= getSumTrue(nux);
    extentTrue -= getExtentTrue(nux);
  }
}


void SplitFrontier::restageAndSplit(vector<IndexSet>& indexSet, DefMap* defMap, BranchSense* branchSense, PreTree* pretree) {
  init(branchSense);
  unsigned int flushCount = defMap->flushRear(this);
  vector<DefCoord> preCand = precandidates(defMap);

  defMap->backdate();
  restage(defMap);
  defMap->eraseLayers(flushCount);

  cutSet = make_unique<CutSet>();
  runSet = make_unique<RunSet>(this, frontier->getNCtg(), frame->getNRow());
  vector<SplitNux> postCand = postSchedule(defMap, preCand);
  setOffsets(postCand);

  split(indexSet, postCand, branchSense);
  consumeFrontier(pretree);
}


/**
   @brief Initializes frontier about to be split
 */
void SplitFrontier::init(BranchSense* branchSense) {
  branchSense->frontierReset();
  nSplit = frontier->getNSplit();
  prebias = vector<double>(nSplit);

  layerPreset(); // virtual
  setPrebias();
}


vector<SplitNux> SplitFrontier::postSchedule(class DefMap* defMap, vector<DefCoord>& preCand) {
  vector<SplitNux> postCand;
  for (auto pc : preCand) {
    PredictorT runCount;
    if (!defMap->isSingleton(pc, runCount)) {
      postCand.emplace_back(pc, this, defMap, runCount);
    }
  }
  return postCand;
}


void SplitFrontier::setOffsets(const vector<SplitNux>& sched) {
  runSet->setOffsets();

  nCand = vector<PredictorT>(nSplit);
  fill(nCand.begin(), nCand.end(), 0);
  for (auto nux : sched) {
    nCand[nux.getNodeIdx()]++;
  }

  candOff = vector<IndexT>(nCand.size());
  IndexT tot = 0;
  IndexT i = 0;
  for (auto nc : nCand) { // Exclusive partial sum.
    candOff[i++] = tot;
    tot += nc;
  }
}


IndexT SplitFrontier::addAccumulator(const SplitNux* cand,
				     PredictorT runCount) const {
  return runCount > 1 ? runSet->addRun(this, cand, runCount) : cutSet->addCut(this, cand);
}


bool SplitFrontier::leftCut(const SplitNux* cand) const {
  return cutSet->leftCut(cand);
}


RunDump SplitFrontier::dumpRun(PredictorT accumIdx) const {
  return runSet->dumpRun(accumIdx);
}


void SplitFrontier::writeCut(const SplitNux* nux,
			     const CutAccum* accum) const {
  cutSet->write(nux, accum);
}


void SplitFrontier::restage(const DefMap* defMap) {
  OMPBound idxTop = restageCoord.size();
  
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound nodeIdx = 0; nodeIdx < idxTop; nodeIdx++) {
      defMap->restage(obsPart.get(), restageCoord[nodeIdx]);
    }
  }

  restageCoord.clear();
}


void SplitFrontier::consumeFrontier(PreTree* pretree) {
  compoundCriteria ? consumeCompound(nuxCompound, pretree) : consumeSimple(nuxMax, pretree);
  clear();
}


double SplitFrontier::getQuantRank(const SplitNux* nux) const {
  return cutSet->getQuantRank(nux);
}


IndexT SplitFrontier::getIdxRight(const SplitNux* nux) const {
  return cutSet->getIdxRight(nux);
}


IndexT SplitFrontier::getIdxLeft(const SplitNux* nux) const {
  return cutSet->getIdxLeft(nux);
}


IndexT SplitFrontier::getImplicitTrue(const SplitNux* cand) const {
  return cand->isFactor(frame) ? runSet->getImplicitTrue(cand) : cutSet->getImplicitTrue(cand);
}


void SplitFrontier::consumeCriterion(PreTree* pretree,
				     const SplitNux* nux) const {
  if (nux->isFactor(frame)) {
    pretree->critBits(nux, nux->getCardinality(frame), runSet->getTrueBits(nux));
  }
  else {
    pretree->critCut(nux, this);
  }
}


void SplitFrontier::clear() {
  prebias.clear();
}


bool SplitFrontier::isUnsplitable(IndexT splitIdx) const {
  return frontier->isUnsplitable(splitIdx);
}


IndexRange SplitFrontier::getBufRange(const DefCoord& preCand) const {
  return frontier->getBufRange(preCand);
}


double SplitFrontier::getSum(const SplitCoord& splitCoord) const {
  return frontier->getSum(splitCoord);
}


IndexT SplitFrontier::getSCount(const SplitCoord& splitCoord) const {
  return frontier->getSCount(splitCoord);
}


IndexT SplitFrontier::getPTId(const SplitCoord& splitCoord) const {
  return frontier->getPTId(splitCoord);
}


SFReg::SFReg(const class SummaryFrame* frame,
	     class Frontier* frontier,
	     const class Sample* sample,
	     bool compoundCriteria,
	     EncodingStyle encodingStyle):
  SplitFrontier(frame, frontier, sample, compoundCriteria, encodingStyle),
  ruMono(vector<double>(0)) {
}


SFReg::~SFReg() {
}


void SFReg::immutables(const SummaryFrame* frame,
		      const vector<double>& bridgeMono) {
  auto numFirst = frame->getNumFirst();
  auto numExtent = frame->getNPredNum();
  auto monoCount = count_if(bridgeMono.begin() + numFirst, bridgeMono.begin() + numExtent, [] (double prob) { return prob != 0.0; });
  if (monoCount > 0) {
    mono = vector<double>(frame->getNPredNum());
    mono.assign(bridgeMono.begin() + frame->getNumFirst(), bridgeMono.begin() + frame->getNumFirst() + frame->getNPredNum());
  }
}


void SFReg::deImmutables() {
  mono.clear();
}


int SFReg::getMonoMode(const SplitNux* cand) const {
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


void SFReg::layerPreset() {
  if (!mono.empty()) {
    ruMono = CallBack::rUnif(nSplit * mono.size());
  }
}


SFCtg::SFCtg(const class SummaryFrame* frame,
	     class Frontier* frontier,
	     const class Sample* sample,
	     bool compoundCriteria,
	     EncodingStyle encodingStyle,
	     PredictorT nCtg_):
  SplitFrontier(frame, frontier, sample, compoundCriteria, encodingStyle),
  nCtg(nCtg_) {
}




const vector<double>& SFCtg::getSumSlice(const SplitNux* cand) const {
  return ctgSum[cand->getNodeIdx()];
}


vector<SplitNux> SplitFrontier::maxCandidates(const vector<SplitNux>& sc) {
  vector<SplitNux> nuxMax(nSplit); // Info initialized to zero.

  OMPBound splitTop = nSplit;
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitIdx = 0; splitIdx < splitTop; splitIdx++) {
      nuxMax[splitIdx] = candMax(sc, candOff[splitIdx], nCand[splitIdx]);
    }
  }

  return nuxMax;
}


SplitNux SplitFrontier::candMax(const vector<SplitNux>& cand,
			  IndexT splitBase,
			  IndexT nCandSplit) const {
  IndexT argMax = splitBase + nCandSplit;
  double runningMax = 0.0;
  for (IndexT splitOff = splitBase; splitOff < splitBase + nCandSplit; splitOff++) {
    if (cand[splitOff].maxInfo(runningMax)) {
      argMax = splitOff;
    }
  }

  return runningMax > 0.0 ? cand[argMax] : SplitNux();
}


void SplitFrontier::consumeSimple(const vector<SplitNux>& nuxSimple,
				  PreTree* pretree) const {
  for (auto nux : nuxSimple) {
    if (!nux.noNux()) {
      pretree->nonterminal(&nux);
      consumeCriterion(pretree, &nux);
    }
  }
}


void SplitFrontier::consumeCompound(const vector<vector<SplitNux>>& nuxMax,
				    PreTree* pretree) const {
  for (auto & nuxCrit : nuxMax) {
    consumeCriteria(pretree, nuxCrit);
  }
}


void SplitFrontier::consumeCriteria(PreTree* pretree,
				    const vector<SplitNux>& nuxCrit) const {
  if (nuxCrit.empty()) {
    return;
  }
  
  // True branches target box exterior.
  // False branches target next criterion or box terminal.
  pretree->offspring(nuxCrit.size());
  for (auto & nux : nuxCrit) {
    pretree->nonterminalInc(&nux);
    consumeCriterion(pretree, &nux);
  }
}


double SFCtg::getSumSquares(const SplitNux *cand) const {
  return sumSquares[cand->getNodeIdx()];
}


double* SFCtg::getAccumSlice(const SplitNux* cand) {
  return &ctgSumAccum[getNumIdx(cand->getPredIdx()) * nSplit * nCtg + cand->getNodeIdx() * nCtg];
}

