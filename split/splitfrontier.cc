// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitfrontier.cc

   @brief Methods to implement splitting at frontier.

   @author Mark Seligman
 */


#include "indexset.h"
#include "frontier.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "obsfrontier.h"
#include "interlevel.h"
#include "runset.h"
#include "cutset.h"
#include "trainframe.h"
#include "ompthread.h"
#include "prng.h"
#include "algsf.h"


vector<double> SFReg::mono; // Numeric monotonicity constraints.


SplitFrontier::SplitFrontier(Frontier* frontier_,
			     bool compoundCriteria_,
			     EncodingStyle encodingStyle_,
			     SplitStyle splitStyle_,
			     void (SplitFrontier::* splitter_)(vector<SplitNux>, BranchSense&)) :
  frame(frontier_->getFrame()),
  frontier(frontier_),
  interLevel(frontier->getInterLevel()),
  ofFront(interLevel->getFront()),
  nPred(frame->getNPred()),
  compoundCriteria(compoundCriteria_),
  encodingStyle(encodingStyle_),
  splitStyle(splitStyle_),
  nSplit(frontier->getNSplit()),
  splitter(splitter_),
  runSet(make_unique<RunSet>(this, frame->getNRow())),
  cutSet(make_unique<CutSet>()),
  nodeInfo(vector<double>(nSplit)) {
}


void SplitFrontier::split(CandType& cand,
			  BranchSense& branchSense) {
  runSet->setOffsets(this);
  frontierPreset(); // virtual.
  (this->*splitter)(cand.getCandidates(interLevel, this), branchSense);
}


PredictorT SplitFrontier::getNCtg() const {
  return frontier->getNCtg();
}



const ObsPart* SplitFrontier::getPartition() const {
  return interLevel->getObsPart();
}


IndexT* SplitFrontier::getIdxBuffer(const SplitNux* nux) const {
  return interLevel->getIdxBuffer(nux);
}


Obs* SplitFrontier::getPredBase(const SplitNux* nux) const {
  return interLevel->getPredBase(nux);
}


RunAccumT* SplitFrontier::getRunAccum(const SplitNux* nux) const {
  return runSet->getAccumulator(nux->getAccumIdx());
}


bool SplitFrontier::isFactor(PredictorT predIdx) const {
  return frame->isFactor(predIdx);
}


PredictorT SplitFrontier::getNumIdx(PredictorT predIdx) const {
  return frame->getNumIdx(predIdx);
}


IndexT SplitFrontier::addAccumulator(const SplitNux* cand) const {
  PredictorT runCount = frame->isFactor(cand->getPredIdx()) ? cand->getRunCount() : 0;
  return runCount > 1 ? runSet->addRun(this, cand, runCount) : cutSet->addCut(this, cand);
}


bool SplitFrontier::leftCut(const SplitNux* cand) const {
  return cutSet->leftCut(cand);
}


void SplitFrontier::writeCut(const SplitNux* nux,
			     const CutAccum* accum) const {
  cutSet->write(interLevel, nux, accum);
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
  return cand->isFactor(this) ? runSet->getImplicitTrue(cand) : cutSet->getImplicitTrue(cand);
}


double SplitFrontier::getSum(const StagedCell* obsCell) const {
  return frontier->getSum(obsCell);
}


IndexT SplitFrontier::getSCount(const StagedCell* obsCell) const {
  return frontier->getSCount(obsCell);
}


double SplitFrontier::getSumSucc(const StagedCell* obsCell,
				 bool sense) const {
  return frontier->getSumSucc(obsCell, sense);
}


IndexT SplitFrontier::getSCountSucc(const StagedCell* obsCell,
				    bool sense) const {
  return frontier->getSCountSucc(obsCell, sense);
}


IndexT SplitFrontier::getPTId(const StagedCell* obsCell) const {
  return frontier->getPTId(obsCell);
}


SFReg::SFReg(class Frontier* frontier,
	     bool compoundCriteria,
	     EncodingStyle encodingStyle,
	     SplitStyle splitStyle,
	     void (SplitFrontier::* splitter)(vector<SplitNux>, BranchSense&)):
  SplitFrontier(frontier, compoundCriteria, encodingStyle, splitStyle, splitter),
  ruMono(vector<double>(0)) {
}


void SFReg::immutables(const TrainFrame* frame,
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

  PredictorT numIdx = getNumIdx(cand->getPredIdx());
  double monoProb = mono[numIdx];
  double prob = ruMono[cand->getNodeIdx() * mono.size() + numIdx];
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


double SFReg::getScore(const IndexSet& iSet) const {
  return iSet.getSum() / iSet.getSCount();
}


void SFReg::frontierPreset() {
  if (!mono.empty()) {
    ruMono = PRNG::rUnif(nSplit * mono.size());
  }
}


SFCtg::SFCtg(class Frontier* frontier,
	     bool compoundCriteria,
	     EncodingStyle encodingStyle,
	     SplitStyle splitStyle,
	     void (SplitFrontier::* splitter) (vector<SplitNux>, BranchSense&)) :
  SplitFrontier(frontier, compoundCriteria, encodingStyle, splitStyle, splitter),
  nCtg(frontier->getNCtg()),
  ctgSum(vector<vector<double>>(nSplit)),
  sumSquares(frontier->sumsAndSquares(ctgSum)),
  ctgSumAccum(vector<double>(frame->getNPredNum() * nCtg * nSplit)),
  ctgJitter(PRNG::rUnif(nCtg * nSplit, 0.5)) {
}


double SFCtg::getScore(const IndexSet& iSet) const {
  const double* nodeJitter = &ctgJitter[iSet.getSplitIdx() * nCtg];
  PredictorT argMax = 0;// TODO:  set to nCtg and error if no count.
  IndexT countMax = 0;
  PredictorT ctg = 0;
  for (auto sc : iSet.getCtgSumCount()) {
    IndexT sCount = sc.getSCount();
    if (sCount > countMax) {
      countMax = sCount;
      argMax = ctg;
    }
    else if (sCount > 0 && sCount == countMax) {
      if (nodeJitter[ctg] > nodeJitter[argMax]) {
	argMax = ctg;
      }
    }
    ctg++;
  }

  //  argMax, ties broken by jitters, plus its own jitter.
  return argMax + nodeJitter[argMax];
}


const vector<double>& SFCtg::getSumSlice(const SplitNux* cand) const {
  return ctgSum[cand->getNodeIdx()];
}


void SplitFrontier::maxSimple(const vector<SplitNux>& sc,
			      BranchSense& branchSense) {
  frontier->updateSimple(maxCandidates(groupCand(sc)), branchSense);
}


vector<SplitNux> SplitFrontier::maxCandidates(const vector<vector<SplitNux>>& candVV) {
  vector<SplitNux> argMax(nSplit); // Info initialized to zero.

  OMPBound splitTop = nSplit;
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitIdx = 0; splitIdx < splitTop; splitIdx++) {
      frontier->candMax(splitIdx, argMax[splitIdx], candVV[splitIdx]);
    }
  }

  return argMax;
}


vector<vector<SplitNux>> SplitFrontier::groupCand(const vector<SplitNux>& cand) const {
  vector<vector<SplitNux>> candVV(nSplit);
  for (auto nux : cand) {
    candVV[nux.getNodeIdx()].emplace_back(nux);
  }

  return candVV;
}


CritEncoding SplitFrontier::splitUpdate(const SplitNux& nux,
					BranchSense& branchSense,
					const IndexRange& range,
					bool increment) const {
  accumUpdate(nux);
  CritEncoding enc(this, nux, increment);
  enc.branchUpdate(this, range, branchSense);
  return enc;
}


void SplitFrontier::accumUpdate(const SplitNux& nux) const {
  if (nux.isFactor(this)) { // Only factor accumulators currently require an update.
    runSet->updateAccum(nux);
  }
}


vector<IndexRange> SplitFrontier::getRange(const SplitNux& nux,
					   const CritEncoding& enc) const {
  if (nux.isFactor(this)) {
    if (splitStyle == SplitStyle::topSlot) {
      return runSet->getTopRange(nux, enc);
    }
    else {
      return runSet->getRange(nux, enc);
    }
  }
  else {
    return getCutRange(nux, enc);
  }
}


vector<IndexRange> SplitFrontier::getCutRange(const SplitNux& nux,
					      const CritEncoding& enc) const {
// Returns the left range iff (BOTH left cut AND true encoding or NEITHER left cut NOR true encoding).
  vector<IndexRange> rangeVec;
  rangeVec.push_back(nux.cutRange(cutSet.get(), !(leftCut(&nux) ^ enc.trueEncoding())));
  return rangeVec;
}


double SFCtg::getSumSquares(const SplitNux *cand) const {
  return sumSquares[cand->getNodeIdx()];
}


double* SFCtg::getAccumSlice(const SplitNux* cand) {
  return &ctgSumAccum[getNumIdx(cand->getPredIdx()) * nSplit * nCtg + cand->getNodeIdx() * nCtg];
}

