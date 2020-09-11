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
#include "bag.h"
#include "bv.h"
#include "leafpredict.h"
#include "predict.h"
#include "rleframe.h"
#include "ompthread.h"

#include <algorithm>

const unsigned int Quant::binSize = 0x1000;


/**
   @brief Constructor.  Caches parameter values and computes compressed
   leaf indices.
 */
Quant::Quant(const PredictReg* predictReg_,
	     const LeafPredict* leaf_,
             const Bag* bag_,
	     const RLEFrame* rleFrame,
	     vector<double> yTrain,
             vector<double> quantile_) :
  predictReg(predictReg_),
  leaf(leaf_),
  baggedRows(bag_->getBitMatrix()),
  nRow(baggedRows->isEmpty() ? 0 : rleFrame->getNRow()),
  valRank(ValRank<double>(&yTrain[0], yTrain.size())),
  rankCount(leaf->setRankCount(baggedRows, valRank.rank())),
  quantile(move(quantile_)),
  qCount(quantile.size()),
  qPred(vector<double>(nRow * qCount)),
  qEst(vector<double>(nRow)),
  rankScale(binScale()),
  binMean(binMeans(valRank, rankScale)) {
}


void Quant::predictBlock(size_t blockStart, size_t blockEnd) {
  if (baggedRows->isEmpty())
    return; // Insufficient leaf information.

  OMPBound rowStart = static_cast<OMPBound>(blockStart);
  OMPBound rowEnd = static_cast<OMPBound>(blockEnd);
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound row = rowStart; row != rowEnd; row++) {
      double yPred = predictReg->getYPred(row);
      predictRow(row, yPred, &qPred[qCount * row], &qEst[row]);
    }
  }
}


unsigned int Quant::binScale() const {
  unsigned int shiftVal = 0;
  while ((binSize << shiftVal) < valRank.getRankCount())
    shiftVal++;

  return shiftVal;
}


void Quant::predictRow(size_t row,
                       double yPred,
                       double qRow[],
                       double *qEst) {
  vector<IndexT> sCountBin(std::min(binSize, valRank.getRankCount()));
  fill(sCountBin.begin(), sCountBin.end(), 0);

  // Scores each rank seen at every predicted leaf.
  //
  IndexT totSamples = 0;
  for (unsigned int tIdx = 0; tIdx < predictReg->getNTree(); tIdx++) {
    IndexT termIdx;
    if (predictReg->isLeafIdx(row, tIdx, termIdx)) {
      totSamples += leafSample(tIdx, termIdx, sCountBin);
    }
  }

  // Builds sample-count thresholds for each quantile.
  vector<double> countThreshold(qCount);
  unsigned int qSlot = 0;
  for (auto & thresh : countThreshold) {
    thresh = totSamples * quantile[qSlot++];  // Rounding properties?
  }

  // Fills in quantile estimates.
  IndexT samplesLeft = quantSamples(sCountBin, countThreshold, yPred, qRow);
  *qEst = static_cast<double>(samplesLeft) / totSamples;
}


IndexT Quant::quantSamples(const vector<IndexT>& sCountBin,
                           const vector<double> threshold,
                           double yPred,
                           double qRow[]) const {
  unsigned int qSlot = 0;
  unsigned int binIdx = 0;
  IndexT samplesSeen = 0;
  IndexT leftSamples = 0; // Samples with y-values <= yPred.
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

  return leftSamples;
}


vector<double> Quant::binMeans(const ValRank<double>& valRank, unsigned int rankScale) {
  vector<double> binMean(std::min(binSize, valRank.getRankCount()));
  fill(binMean.begin(), binMean.end(), 0.0);
  vector<size_t> binCount(binMean.size());
  fill(binCount.begin(), binCount.end(), 0);
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


IndexT Quant::leafSample(unsigned int tIdx,
                         IndexT leafIdx,
                         vector<IndexT>& sCountBin) const {
  IndexT sampleTot = 0;
  size_t leafStart, leafEnd; // Forest-relative leaf indices.
  leaf->bagBounds(tIdx, leafIdx, leafStart, leafEnd);
  for (size_t bagIdx = leafStart; bagIdx < leafEnd; bagIdx++) {
    IndexT sc = rankCount[bagIdx].sCount;
    IndexT binIdx = binRank(rankCount[bagIdx].rank);
    sCountBin[binIdx] += sc;
    sampleTot += sc;
  }
  return sampleTot;
}
