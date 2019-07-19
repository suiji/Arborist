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

const size_t Quant::binSize = 0x1000;


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
 
  OMPBound row;
  OMPBound rowSup = (OMPBound) (rowStart + extent);
#pragma omp parallel default(shared) private(row) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = rowStart; row < rowSup; row++) {
      double yPred = leafReg->getYPred(row);
      predictRow(frame, row - rowStart, yPred, &qPred[qCount * row], &qEst[row]);
    }
  }
}


unsigned int Quant::binScale() const {
  unsigned int shiftVal = 0;
  while ((binSize << shiftVal) < valRank.getNRow())
    shiftVal++;

  return shiftVal;
}


void Quant::predictRow(const PredictFrame *frame,
                       unsigned int blockRow,
                       double yPred,
                       double qRow[],
                       double *qEst) {
  vector<unsigned int> sCount(std::min(binSize, valRank.getNRow()));
  fill(sCount.begin(), sCount.end(), 0);

  // Scores each rank seen at every predicted leaf.
  //
  unsigned int totSamples = 0;
  for (unsigned int tIdx = 0; tIdx < leafReg->getNTree(); tIdx++) {
    unsigned int termIdx;
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
  unsigned int yQuant = quantSamples(sCount, countThreshold, yPred, qRow);
  *qEst = static_cast<double>(yQuant) / totSamples;
}


unsigned int Quant::quantSamples(const vector<unsigned int>& sCount,
                                 const vector<double> threshold,
                                 double yPred,
                                 double qRow[]) const {
  unsigned int qSlot = 0;
  unsigned int binIdx = 0;
  unsigned int samplesSeen = 0;
  unsigned int yQuant = 0;
  for (auto sc : sCount) {
    samplesSeen += sc;
    while (qSlot < qCount && samplesSeen >= threshold[qSlot]) {
      qRow[qSlot++] = binMean[binIdx];
    }
    if (yPred > binMean[binIdx]) {
      yQuant = samplesSeen;
    }
    else if (qSlot >= qCount)
      return yQuant;
    binIdx++;
  }

  return yQuant;
}


vector<double> Quant::binMeans(const ValRank<double>& valRank, unsigned int rankScale) {
  const auto slotWidth = 1 << rankScale;
  size_t binIdx = 0;
  vector<double> binMean(std::min(binSize, valRank.getNRow()));
  for (size_t idxStart = 0; idxStart < valRank.getNRow(); idxStart += slotWidth) {
    size_t idxEnd = min(valRank.getNRow(), idxStart + slotWidth);
    double sum = 0.0;
    unsigned int count = 0;
    for (auto idx = idxStart; idx < idxEnd; idx++) {
      sum += valRank.getVal(idx);
      count++;
    }
    binMean[binIdx++] =  sum / count;
  }

  return binMean;
}


  // TODO:  For binning, rerun, restricting to "hot" bins observed
  // over sample set.  This should improve resolution for hot
  // bins.

unsigned int Quant::leafSample(unsigned int tIdx,
                               unsigned int leafIdx,
                               vector<unsigned int> &sCount) const {
  unsigned int sampleTot = 0;
  unsigned int leafStart, leafEnd;
  leafReg->bagBounds(tIdx, leafIdx, leafStart, leafEnd);
  for (unsigned int bagIdx = leafStart; bagIdx < leafEnd; bagIdx++) {
    unsigned int sc = rankCount[bagIdx].sCount;
    unsigned int bin = binRank(rankCount[bagIdx].rank);
    sCount[bin] += sc;
    sampleTot += sc;
  }
  return sampleTot;
}
