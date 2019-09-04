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
#include "leaf.h"
#include "predict.h"
#include "ompthread.h"

#include <algorithm>

const unsigned int Quant::binSize = 0x1000;


/**
   @brief Constructor.  Caches parameter values and computes compressed
   leaf indices.
 */
Quant::Quant(const LeafFrameReg* leaf,
             const Bag* bag_,
             const vector<double>& quantile_) :
  leafReg(leaf),
  baggedRows(bag_->getBitMatrix()),
  valRank(ValRank<double>(leafReg->getYTrain(), leafReg->getRowTrain())),
  rankCount(leafReg->setRankCount(baggedRows, valRank.rank())),
  quantile(quantile_),
  qCount(quantile.size()),
  qPred(vector<double>(getNRow() * qCount)),
  qEst(vector<double>(getNRow())),
  rankScale(binScale()),
  binMean(binMeans(valRank, rankScale)) {
}

unsigned int Quant::getNRow() const {
  return baggedRows->isEmpty() ? 0 : leafReg->getRowPredict();
}


void Quant::predictAcross(const PredictFrame* frame,
                          size_t rowStart,
                          size_t extent) {
  if (baggedRows->isEmpty())
    return; // Insufficient leaf information.
 
  OMPBound rowSup = (OMPBound) (rowStart + extent);
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound row = rowStart; row < rowSup; row++) {
      double yPred = leafReg->getYPred(row);
      predictRow(frame, row - rowStart, yPred, &qPred[qCount * row], &qEst[row]);
    }
  }
}


unsigned int Quant::binScale() const {
  unsigned int shiftVal = 0;
  while ((binSize << shiftVal) < valRank.getRankCount())
    shiftVal++;

  return shiftVal;
}


void Quant::predictRow(const PredictFrame *frame,
                       unsigned int blockRow,
                       double yPred,
                       double qRow[],
                       double *qEst) {
  vector<PredictorT> sCount(std::min(binSize, valRank.getRankCount()));
  fill(sCount.begin(), sCount.end(), 0);

  // Scores each rank seen at every predicted leaf.
  //
  IndexT totSamples = 0;
  for (unsigned int tIdx = 0; tIdx < leafReg->getNTree(); tIdx++) {
    IndexT termIdx;
    if (!frame->isBagged(blockRow, tIdx, termIdx)) {
      totSamples += leafSample(tIdx, termIdx, sCount);
    }
  }

  // Builds sample-count thresholds for each quantile.
  vector<double> countThreshold(qCount);
  unsigned int qSlot = 0;
  for (auto & thresh : countThreshold) {
    thresh = totSamples * quantile[qSlot++];  // Rounding properties?
  }

  // Fills in quantile estimates.
  IndexT samplesLeft = quantSamples(sCount, countThreshold, yPred, qRow);
  *qEst = static_cast<double>(samplesLeft) / totSamples;
}


IndexT Quant::quantSamples(const vector<PredictorT>& sCount,
                           const vector<double> threshold,
                           double yPred,
                           double qRow[]) const {
  unsigned int qSlot = 0;
  unsigned int binIdx = 0;
  IndexT samplesSeen = 0;
  IndexT leftSamples = 0; // Samples with y-values <= yPred.
  for (auto sc : sCount) {
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
                         vector<PredictorT> &sCount) const {
  IndexT sampleTot = 0;
  IndexT leafStart, leafEnd;
  leafReg->bagBounds(tIdx, leafIdx, leafStart, leafEnd);
  for (IndexT bagIdx = leafStart; bagIdx < leafEnd; bagIdx++) {
    unsigned int sc = rankCount[bagIdx].sCount;
    unsigned int binIdx = binRank(rankCount[bagIdx].rank);
    sCount[binIdx] += sc;
    sampleTot += sc;
  }
  return sampleTot;
}
