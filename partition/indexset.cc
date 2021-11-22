// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file indexset.cc

   @brief Maintains frontier tree nodes as blocks within ObsPart.

   @author Mark Seligman
 */

#include "indexset.h"
#include "sample.h"
#include "splitnux.h"
#include "splitfrontier.h"
#include "frontier.h"
#include "path.h"

IndexT IndexSet::minNode = 0;


void IndexSet::immutables(IndexT minNode) {
  IndexSet::minNode = minNode;
}


void IndexSet::deImmutables() {
  minNode = 0;
}


IndexSet::IndexSet() :
  splitIdx(0),
  ptId(0),
  sCount(0),
  sum(0.0),
  minInfo(0.0),
  path(0),
  doesSplit(false),
  unsplitable(false),
  extentTrue(0),
  sCountTrue(0),
  sumTrue(0.0),
  trueEncoding(true),
  trueExtinct(false),
  falseExtinct(false) {
}


void IndexSet::initRoot(const Sample* sample) {
  splitIdx = 0;
  sCount = sample->getNSamp();
  bufRange = IndexRange(0, sample->getBagCount());
  minInfo = 0.0;
  ptId = 0;
  sum = sample->getBagSum();
  path = 0;
  relBase = 0;
  ctgSum = sample->getCtgRoot();
  ctgTrue = vector<SumCount>(ctgSum.size());
  
  initInattainable(sample->getBagCount());
}


void IndexSet::dispatch(Frontier* frontier) {
  if (doesSplit) {
    nonterminal(frontier);
  }
  else {
    terminal(frontier);
  }
}


void IndexSet::terminal(Frontier *frontier) {
  succOnly = frontier->idxSucc(bufRange.getExtent(), offOnly, true);
}


void IndexSet::nonterminal(Frontier* frontier) {
  frontier->getPTIdTF(ptId, ptTrue, ptFalse);
  IndexT succExtent;
  bool extinct;
  extinct = !succSplitable(true, succExtent);
  succTrue = frontier->idxSucc(succExtent, offTrue, extinct);

  extinct = !succSplitable(false, succExtent);
  succFalse = frontier->idxSucc(succExtent, offFalse, extinct);
  IdxPath::pathLR(path, pathTrue, pathFalse);
}


void IndexSet::reindex(const BranchSense* branchSense,
                       Frontier* index,
                       IndexT idxLive,
                       vector<IndexT>& succST) {
  if (!doesSplit) {
    index->relExtinct(relBase, bufRange.getExtent(), ptId);
  }
  else {
    nontermReindex(branchSense, index, idxLive, succST);
  }
}


void IndexSet::nontermReindex(const BranchSense* branchSense,
                              Frontier* index,
                              IndexT idxLive,
                              vector<IndexT>&succST) {
  IndexT baseTrue = offTrue;
  IndexT baseFalse = offFalse;
  for (IndexT relIdx = relBase; relIdx < relBase + bufRange.getExtent(); relIdx++) {
    bool trueBranch = branchSense->senseTrue(relIdx, !trueEncoding);
    IndexT targIdx = getOffSucc(trueBranch);
    if (targIdx < idxLive) {
      succST[targIdx] = index->relLive(relIdx, targIdx, getPathSucc(trueBranch), trueBranch ? baseTrue : baseFalse, getPTSucc(trueBranch));
    }
    else {
      index->relExtinct(relIdx, getPTSucc(trueBranch));
    }
  }
}


void IndexSet::succHands(Frontier* frontier, vector<IndexSet>& indexNext) const {
  if (doesSplit) {
    succHand(frontier, indexNext, true);
    succHand(frontier, indexNext, false);
  }
}


void IndexSet::succHand(Frontier* frontier, vector<IndexSet>& indexNext, bool trueBranch) const {
  IndexT succIdx = getIdxSucc(trueBranch);
  if (succIdx < indexNext.size()) { // Otherwise terminal in next level.
    indexNext[succIdx].succInit(frontier, this, trueBranch);
  }
}


void IndexSet::succInit(Frontier *frontier,
                        const IndexSet* par,
                        bool trueBranch) {
  splitIdx = par->getIdxSucc(trueBranch);
  sCount = par->getSCountSucc(trueBranch);
  bufRange = IndexRange(par->getStartSucc(trueBranch), par->getExtentSucc(trueBranch));
  minInfo = par->getMinInfo();
  ptId = par->getPTIdSucc(frontier, trueBranch);

  unsplitable = (trueBranch && par->trueExtinct) || (!trueBranch && par->falseExtinct);
  
  sum = par->getSumSucc(trueBranch);
  path = par->getPathSucc(trueBranch);
  relBase = frontier->getRelBase(splitIdx);
  frontier->reachingPath(splitIdx, par->getSplitIdx(), bufRange, relBase, path);

  ctgSum = trueBranch ? par->ctgTrue : SumCount::minus(par->ctgSum, par->ctgTrue);
  ctgTrue = vector<SumCount>(ctgSum.size());
  
  // Inattainable value.  Reset only when non-terminal:
  initInattainable(frontier->getBagCount());
}


IndexT IndexSet::getPTIdSucc(const Frontier* frontier, bool trueBranch) const {
  return frontier->getPTIdSucc(ptId, trueBranch);
}


vector<double> IndexSet::sumsAndSquares(double& sumSquares) {
  vector<double> sumOut(ctgSum.size());
  sumSquares =  0.0;
  for (PredictorT ctg = 0; ctg < ctgSum.size(); ctg++) {
    unsplitable |= !ctgSum[ctg].splitable(sCount, sumOut[ctg]);
    sumSquares += sumOut[ctg] * sumOut[ctg];
  }

  return sumOut;
}


void IndexSet::candMax(const vector<SplitNux>& cand,
		       SplitNux& argMaxNux) const {
  IndexT argMax = cand.size();
  double runningMax = 0.0;
  for (IndexT splitOff = 0; splitOff < cand.size(); splitOff++) {
    if (cand[splitOff].maxInfo(runningMax)) {
      argMax = splitOff;
    }
  }

  if (runningMax > 0.0 && isInformative(cand[argMax])) {
    argMaxNux = cand[argMax];
  }
}


void IndexSet::update(const SplitFrontier* sf,
		      const SplitNux& nux,
		      BranchSense* branchSense,
		      const IndexRange& range,
		      bool increment) {
  CritEncoding enc = sf->splitUpdate(nux, branchSense, range, increment);
  doesSplit = true;
  trueEncoding = enc.trueEncoding(); // Final state is most recent update.
  minInfo = nux.getMinInfo(); // REVISE as update
  SumCount::incr(ctgTrue, trueEncoding ? enc.scCtg : SumCount::minus(ctgSum, enc.scCtg));
  enc.getISetVals(sCountTrue, sumTrue, extentTrue);
}


void IndexSet::surveySplit(SplitSurvey& survey) const {
  if (isTerminal()) {
    survey.leafCount++;
  }
  else {
    survey.splitNext += splitCensus(survey);
  }
}


unsigned int IndexSet::splitCensus(SplitSurvey& survey) const {
  return splitAccum(true, survey) + splitAccum(false, survey);
}


unsigned int IndexSet::splitAccum(bool sense,
                                  SplitSurvey& survey) const {
  IndexT succExtent;
  if (succSplitable(sense, succExtent)) {
    survey.idxLive += succExtent;
    survey.idxMax = max(survey.idxMax, succExtent);
    return 1;
  }
  else {
    return 0;
  }
}

  

bool IndexSet::isInformative(const SplitNux& nux) const {
  return nux.getInfo() > minInfo;
}
