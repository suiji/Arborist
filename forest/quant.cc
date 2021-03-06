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
#include "predict.h"
#include "rleframe.h"

#include <algorithm>

const unsigned int Quant::binSize = 0x1000;


/**
   @brief Constructor.  Caches parameter values and computes compressed
   leaf indices.
 */
Quant::Quant(const Predict* predict,
	     const LeafReg* leaf,
	     const RLEFrame* rleFrame,
             const vector<double>& quantile_) :
  quantile(move(quantile_)),
  qCount(quantile.size()),
  sampler(predict->getSampler()),
  empty(sampler->isEmpty() || quantile.empty()),
  valRank(ValRank<double>(&leaf->getYTrain()[0], empty ? 0 : leaf->getYTrain().size())),
  rankCount(empty ? vector<RankCount>(0) : sampler->countLeafRanks(predict, valRank.rank())),
  rankScale(empty ? 0 : binScale()),
  binMean(empty ? vector<double>(0) : binMeans(valRank)),
  qPred(vector<double>(empty ? 0 : rleFrame->getNRow() * qCount)),
  qEst(vector<double>(empty ? 0 : rleFrame->getNRow())) {
}


unsigned int Quant::binScale() const {
  unsigned int shiftVal = 0;
  while ((binSize << shiftVal) < valRank.getRankCount())
    shiftVal++;

  return shiftVal;
}


vector<double> Quant::binMeans(const ValRank<double>& valRank) const {
  vector<double> binMean(std::min(binSize, valRank.getRankCount()));
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


void Quant::predictRow(const PredictReg* predict, size_t row) {
  vector<IndexT> sCountBin(std::min(binSize, valRank.getRankCount()));

  // Scores each rank seen at every predicted leaf.
  //
  IndexT totSamples = 0;
  for (unsigned int tIdx = 0; tIdx < predict->getNTree(); tIdx++) {
    IndexT termIdx;
    if (predict->isLeafIdx(row, tIdx, termIdx)) {
      totSamples += sampleLeaf(predict, tIdx, termIdx, sCountBin);
    }
  }

  // Builds sample-count thresholds for each quantile.
  vector<double> countThreshold(qCount);
  unsigned int qSlot = 0;
  for (auto & thresh : countThreshold) {
    thresh = totSamples * quantile[qSlot++];  // Rounding properties?
  }

  // Fills in quantile estimates.
  quantSamples(predict, sCountBin, countThreshold, totSamples, row);
}

// TODO: Pre-compute for each leaf and cache.  This is the slow step.
IndexT Quant::sampleLeaf(const Predict* predict,
			 unsigned int tIdx,
			 IndexT leafIdx,
			 vector<IndexT>& sCountBin) const {
  IndexT sampleTot = 0;
  size_t leafStart, leafEnd; // Forest-relative leaf indices.
  predict->sampleBounds(tIdx, leafIdx, leafStart, leafEnd);
  for (size_t sIdx = leafStart; sIdx < leafEnd; sIdx++) {
    const RankCount& rc = rankCount[sIdx];
    sCountBin[binRank(rc.rank)] += rc.sCount;
    sampleTot += rc.sCount;
  }
  return sampleTot; // Single leaf, so fits in IndexT.
}


void Quant::quantSamples(const PredictReg* predict,
			 const vector<IndexT>& sCountBin,
                         const vector<double>& threshold,
			 IndexT totSample,
			 size_t row) {
  unsigned int qSlot = 0;
  unsigned int binIdx = 0;
  IndexT samplesSeen = 0;
  IndexT leftSamples = 0; // # samples with y-values <= yPred.
  double yPred = predict->getYPred(row);
  double* qRow = &qPred[qCount * row];
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

  qEst[row] = static_cast<double>(leftSamples) / totSample;
}
