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
#include "interlevel.h"
#include "predictorframe.h"
#include "ompthread.h"
#include "prng.h"
#include "algsf.h"


vector<double> SFReg::mono; // Numeric monotonicity constraints.


SplitFrontier::SplitFrontier(Frontier* frontier_,
			     bool compoundCriteria_,
			     EncodingStyle encodingStyle_,
			     SplitStyle splitStyle_,
			     void (SplitFrontier::* driver_)(const CandType&,
							     BranchSense&),
			     void (SplitFrontier::* splitter_)(SplitNux&)) :
  frame(frontier_->getFrame()),
  frontier(frontier_),
  interLevel(frontier->getInterLevel()),
  compoundCriteria(compoundCriteria_),
  encodingStyle(encodingStyle_),
  splitStyle(splitStyle_),
  nSplit(frontier->getNSplit()),
  driver(driver_),
  splitter(splitter_),
  runSet(make_unique<RunSet>(this)),
  cutSet(make_unique<CutSet>()) {
}


void SplitFrontier::split(const CandType& cand,
			  BranchSense& branchSense) {
  (this->*driver)(cand, branchSense);
}


void SplitFrontier::splitSimple(const CandType& cnd,
				BranchSense& branchSense) {
  vector<SplitNux> cand = cnd.stagedSimple(interLevel, this);
  for (IndexT blockStart = 0; blockStart < cand.size(); blockStart += splitBlock) {
    OMPBound splitTop = blockStart + splitBlock;
    if (splitTop > cand.size())
      splitTop = cand.size();

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
    {
#pragma omp for schedule(dynamic, 1)
      for (OMPBound splitPos = blockStart; splitPos < splitTop; splitPos++) {
	(this->*splitter)(cand[splitPos]);
      }
    }
  }
  
  maxSimple(cand, branchSense);
}


void SplitFrontier::accumPreset() {
  runSet->accumPreset(this);
  cutSet->accumPreset();
}


PredictorT SplitFrontier::getNCtg() const {
  return frontier->getNCtg();
}


const ObsPart* SplitFrontier::getPartition() const {
  return interLevel->getObsPart();
}


IndexT* SplitFrontier::getIdxBuffer(const SplitNux& nux) const {
  return interLevel->getIdxBuffer(nux);
}


Obs* SplitFrontier::getPredBase(const SplitNux& nux) const {
  return interLevel->getPredBase(nux);
}


bool SplitFrontier::isFactor(const SplitNux& nux) const {
  return frame->isFactor(nux);
}


PredictorT SplitFrontier::getNumIdx(PredictorT predIdx) const {
  return frame->getTypedIdx(predIdx);
}


IndexT SplitFrontier::accumulatorIndex(const SplitNux& cand) const {
  if (isFactor(cand)) {
    return runSet->preIndex(this, cand);
  }
  else {
    return cutSet->preIndex();
  }
}


bool SplitFrontier::leftCut(const SplitNux& cand) const {
  return cutSet->leftCut(cand);
}


void SplitFrontier::writeCut(const SplitNux& nux,
			     const CutAccum& accum) const {
  cutSet->write(interLevel, nux, accum);
}


double SplitFrontier::getQuantRank(const SplitNux& nux) const {
  return cutSet->getQuantRank(nux);
}


IndexT SplitFrontier::getIdxRight(const SplitNux& nux) const {
  return cutSet->getIdxRight(nux);
}


IndexT SplitFrontier::getIdxLeft(const SplitNux& nux) const {
  return cutSet->getIdxLeft(nux);
}


IndexT SplitFrontier::getImplicitTrue(const SplitNux& cand) const {
  return isFactor(cand) ? runSet->getImplicitTrue(cand) : cutSet->getImplicitTrue(cand);
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


// Placeholder bit for proxy lies one beyond the factor's cardinality and
// remains unset for quick test exit.  To support trap-and-bail for factors,
// the number of bits should double, allowing lookup of (in)visibility state.
PredictorT SplitFrontier::critBitCount(const SplitNux& nux) const {
  return 1 + frame->getFactorExtent(nux);
}


SFReg::SFReg(class Frontier* frontier,
	     bool compoundCriteria,
	     EncodingStyle encodingStyle,
	     SplitStyle splitStyle,
	     void (SplitFrontier::* driver)(const CandType& cand,
					    BranchSense&),
	     void (SplitFrontier::* splitter)(SplitNux&)) :
  SplitFrontier(frontier, compoundCriteria, encodingStyle, splitStyle, driver, splitter),
  ruMono(sampleMono(nSplit)) {
}


void SFReg::immutables(const PredictorFrame* frame,
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


int SFReg::getMonoMode(const SplitNux& cand) const {
  if (ruMono.empty())
    return 0;

  PredictorT numIdx = getNumIdx(cand.getPredIdx());
  double monoProb = mono[numIdx];
  double prob = ruMono[cand.getNodeIdx() * mono.size() + numIdx];
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


vector<double> SFReg::sampleMono(IndexT nSplit) {
  if (!mono.empty()) {
    return PRNG::rUnif<double>(nSplit * mono.size());
  }
  else
    return vector<double>(0);
}


SFCtg::SFCtg(class Frontier* frontier,
	     bool compoundCriteria,
	     EncodingStyle encodingStyle,
	     SplitStyle splitStyle,
	     void (SplitFrontier::* driver) (const CandType& cand,
					     BranchSense&),
	     void (SplitFrontier::* splitter)(SplitNux&)) :
  SplitFrontier(frontier, compoundCriteria, encodingStyle, splitStyle, driver, splitter),
  ctgSum(vector<vector<double>>(nSplit)),
  sumSquares(frontier->sumsAndSquares(ctgSum)) {
}


const vector<double>& SFCtg::ctgNodeSums(const SplitNux& cand) const {
  return ctgSum[cand.getNodeIdx()];
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
      argMax[splitIdx] = frontier->candMax(splitIdx, candVV[splitIdx]);
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
  if (isFactor(nux)) { // Only factor accumulators currently require an update.
    runSet->accumUpdate(nux);
  }
}


vector<IndexRange> SplitFrontier::getRange(const SplitNux& nux,
					   const CritEncoding& enc) const {
  return isFactor(nux) ? runSet->getRange(nux, enc) : getCutRange(nux, enc);
}


vector<IndexRange> SplitFrontier::getCutRange(const SplitNux& nux,
					      const CritEncoding& enc) const {
// Returns the left range iff (BOTH left cut AND true encoding or NEITHER left cut NOR true encoding).
  vector<IndexRange> rangeVec;
  rangeVec.push_back(nux.cutRange(cutSet.get(), !(leftCut(nux) ^ enc.trueEncoding())));
  return rangeVec;
}


double SFCtg::getSumSquares(const SplitNux& cand) const {
  return sumSquares[cand.getNodeIdx()];
}
