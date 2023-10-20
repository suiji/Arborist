// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file quant.cc

   @brief Prediction methods for quantiles.

   @author Mark Seligman
 */

#include "quant.h"
#include "forest.h"
#include "predict.h"
#include "response.h"
#include "sampler.h"
#include <algorithm>


const unsigned int Quant::binSize = 0x1000;
vector<double> Quant::quantile = vector<double>(0);


void Quant::init(vector<double> quantile_) {
  quantile = std::move(quantile_);
}


void Quant::deInit() {
  quantile = vector<double>(0);
}


/**
   @brief Constructor.  Caches parameter values and computes compressed
   leaf indices.
 */
Quant::Quant(const Sampler* sampler,
	     const Predict* predict,
	     bool reportAuxiliary) :
  leaf(predict->forest->getLeaf()),
  empty(!reportAuxiliary || quantile.empty() || leaf.empty() || !sampler->hasSamples()),
  qCount(quantile.size()),
  trapAndBail(DecNode::trapAndBail()),
  leafDom((empty || !trapAndBail) ? vector<vector<IndexRange>>(0) : predict->forest->leafDominators()), 
  valRank(RankedObs<double>(&(reinterpret_cast<const ResponseReg*>(sampler->getResponse())->getYTrain())[0],
			    empty ? 0 : reinterpret_cast<const ResponseReg*>(sampler->getResponse())->getYTrain().size())),
  rankCount(empty ? vector<vector<vector<RankCount>>>(0) : leaf.alignRanks(sampler, valRank.rank())),
  rankScale(empty ? 0 : binScale()),
  binMean(empty ? vector<double>(0) : binMeans(valRank)),
  qPred(vector<double>(empty ? 0 : predict->getNObs() * qCount)),
  qEst(vector<double>(empty ? 0 : predict->getNObs())) {
}


unsigned int Quant::binScale() const {
  unsigned int shiftVal = 0;
  while ((binSize << shiftVal) < valRank.getRankCount())
    shiftVal++;

  return shiftVal;
}


vector<double> Quant::binMeans(const RankedObs<double>& valRank) const {
  vector<double> binMean(std::min(static_cast<IndexT>(binSize), valRank.getRankCount()));
  vector<size_t> binCount(binMean.size());
  for (IndexT idx = 0; idx < valRank.getNRow(); idx++) {
    unsigned int binIdx = binRank(valRank.getRank(idx));
    binMean[binIdx] += valRank.getVal(idx);
    binCount[binIdx]++;
  }
  unsigned int binIdx = 0;
  for (auto bc : binCount) {
    if (bc == 0)
      break;
    binMean[binIdx++] /= bc;
  }

  return binMean;
}


void Quant::predictRow(const Predict* predict,
		       const ForestPredictionReg* prediction,
		       size_t obsIdx) {
  if (isEmpty())
    return;
  vector<IndexT> sCountBin(std::min(static_cast<IndexT>(binSize), valRank.getRankCount()));
  IndexT totSamples = 0;
  if (trapAndBail) {
    for (unsigned int tIdx = 0; tIdx < predict->getNTree(); tIdx++) {
      IndexT nodeIdx;
      if (predict->getFinalIdx(obsIdx, tIdx, nodeIdx)) {
	IndexRange leafRange = leafDom[tIdx][nodeIdx];
	for (IndexT leafIdx = leafRange.getStart(); leafIdx != leafRange.getEnd(); leafIdx++) {
	  totSamples += sampleLeaf(tIdx, leafIdx, sCountBin);
	}
      }
    }
  }
  else {
    for (unsigned int tIdx = 0; tIdx < predict->getNTree(); tIdx++) {
      IndexT leafIdx;
      if (predict->isLeafIdx(obsIdx, tIdx, leafIdx)) {
	totSamples += sampleLeaf(tIdx, leafIdx, sCountBin);
      }
    }
  }
  // Builds sample-count thresholds for each quantile.
  vector<double> countThreshold(qCount);
  unsigned int qSlot = 0;
  for (auto & thresh : countThreshold) {
    thresh = totSamples * quantile[qSlot++];  // Rounding properties?
  }

  // Fills in quantile estimates.
  quantSamples(prediction, sCountBin, countThreshold, totSamples, obsIdx);
}


IndexT Quant::sampleLeaf(unsigned int tIdx,
			 IndexT leafIdx,
			 vector<IndexT>& sCountBin) const {
  IndexT sampleTot = 0;
  // sampleTot can be precomputed and cached, but rank traversal is
  // irregular.
  for (RankCount rc : rankCount[tIdx][leafIdx]) {
    sCountBin[binRank(rc.getRank())] += rc.getSCount();
    sampleTot += rc.getSCount();
  }
  return sampleTot; // Single leaf, so fits in IndexT.
}


void Quant::quantSamples(const ForestPredictionReg* prediction,
			 const vector<IndexT>& sCountBin,
                         const vector<double>& threshold,
			 IndexT totSample,
			 size_t obsIdx) {
  unsigned int qSlot = 0;
  unsigned int binIdx = 0;
  IndexT samplesSeen = 0;
  IndexT leftSamples = 0; // # samples with y-values <= yPred.
  double yPred = prediction->getValue(obsIdx);
  double* qRow = &qPred[qCount * obsIdx];
  for (auto sc : sCountBin) {
    samplesSeen += sc;
    while (qSlot < qCount && samplesSeen >= threshold[qSlot]) {
      qRow[qSlot++] = binMean[binIdx];
    }
    if (yPred > binMean[binIdx]) {
      leftSamples = samplesSeen;
    }
    else if (qSlot >= qCount)
      break;
    binIdx++;
  }

  qEst[obsIdx] = static_cast<double>(leftSamples) / totSample;
}
