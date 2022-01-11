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
  ctgSum = sample->getCtgRoot();
  ctgTrue = vector<SumCount>(ctgSum.size());
  
  initInattainable(sample->getBagCount());
}


void IndexSet::nonterminal(const SampleMap& smNext) {
  ptTrue = smNext.ptIdx[idxNext];
  ptFalse = smNext.ptIdx[idxNext+1];
  IdxPath::pathLR(path, pathTrue, pathFalse);
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

  unsplitable = (bufRange.getExtent() < minNode) || (trueBranch && par->trueExtinct) || (!trueBranch && par->falseExtinct);
  
  sum = par->getSumSucc(trueBranch);
  path = par->getPathSucc(trueBranch);
  frontier->reachingPath(*this, par->getSplitIdx());

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


void IndexSet::candMax(const vector<SplitNux>& candVec,
		       SplitNux& argMaxNux) const {
  const SplitNux* amn = &argMaxNux;
  for (auto & cand : candVec) {
    cand.maxInfo(amn);
  }
  if (isInformative(amn))
    argMaxNux = *amn;
}


void IndexSet::update(const CritEncoding& enc) {
  // trueEncoding:  Final state is most recent update.
  // minInfo:  REVISE as update
  doesSplit = true;
  enc.getISetVals(sCountTrue, sumTrue, extentTrue, trueEncoding, minInfo);
  SumCount::incr(ctgTrue, trueEncoding ? enc.scCtg : SumCount::minus(ctgSum, enc.scCtg));
}


bool IndexSet::isInformative(const SplitNux* nux) const {
  return nux->getInfo() > minInfo;
}
