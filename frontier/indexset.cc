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
#include "sampledobs.h"
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

/**
   @brief Root constructor:  some initialization from SampledObs.
 */
IndexSet::IndexSet(const SampledObs* sample) :
  splitIdx(0),
  bufRange(IndexRange(0, sample->getBagCount())),
  sCount(sample->getNSamp()),
  sum(sample->getBagSum()),
  path(0),
  ptId(0),
  ctgSum(sample->getCtgRoot()),
  minInfo(0.0),
  doesSplit(false),
  unsplitable(bufRange.getExtent() < minNode),
  idxNext(sample->getBagCount()),
  extentTrue(0),
  sCountTrue(0),
  sumTrue(0.0),
  trueEncoding(true),
  ctgTrue(vector<SumCount>(ctgSum.size())),
  trueExtinct(false),
  falseExtinct(false) {
}


IndexSet::IndexSet(const Frontier *frontier,
		   const IndexSet& pred,
		   bool trueBranch) :
  splitIdx(pred.getIdxSucc(trueBranch)),
  bufRange(IndexRange(pred.getStartSucc(trueBranch), pred.getExtentSucc(trueBranch))),
  sCount(pred.getSCountSucc(trueBranch)),
  sum(pred.getSumSucc(trueBranch)),
  path(pred.getPathSucc(trueBranch)),
  ptId(pred.getPTIdSucc(frontier, trueBranch)),
  ctgSum(trueBranch ? pred.ctgTrue : SumCount::minus(pred.ctgSum, pred.ctgTrue)),
  minInfo(pred.getMinInfo()),
  doesSplit(false),
  unsplitable((bufRange.getExtent() < minNode) || (trueBranch && pred.trueExtinct) || (!trueBranch && pred.falseExtinct)),
  idxNext(frontier->getBagCount()), // Inattainable.
  extentTrue(0),
  sCountTrue(0),
  sumTrue(0.0),
  trueEncoding(true),
  ctgTrue(vector<SumCount>(ctgSum.size())),
  trueExtinct(false),
  falseExtinct(false) {
}


PathT IndexSet::getPathSucc(bool trueBranch) const {
  return IdxPath::pathSucc(path, trueBranch);
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


SplitNux IndexSet::candMax(const vector<SplitNux>& candVec) const {
  SplitNux argMaxNux;
  for (auto cand : candVec) {
    if (cand.maxInfo(argMaxNux))
      argMaxNux = cand;
  }
  if (isInformative(argMaxNux))
    return argMaxNux;
  else
    return SplitNux(); // zero-information placeholder.
}


bool IndexSet::isInformative(const SplitNux& nux) const {
  return nux.getInfo() > minInfo;
}


void IndexSet::update(const CritEncoding& enc) {
  // trueEncoding:  Final state is most recent update.
  // minInfo:  REVISE as update
  doesSplit = true;
  enc.getISetVals(sCountTrue, sumTrue, extentTrue, trueEncoding, minInfo);
  SumCount::incr(ctgTrue, trueEncoding ? enc.scCtg : SumCount::minus(ctgSum, enc.scCtg));
}
