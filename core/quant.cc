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
Quant::Quant(const PredictBox* box,
             const double* quantile_,
             unsigned int qCount_) :
  leafReg(static_cast<LeafFrameReg*>(box->leafFrame)),
  baggedRows(box->bag),
  yTrain(leafReg->getYTrain()),
  yRanked(leafReg->getRowTrain()),
  quantile(quantile_),
  qCount(qCount_),
  nRow(baggedRows->getNRow() == 0 ? 0 : leafReg->rowPredict()),
  qPred(vector<double>(nRow * qCount)),
  rankCount(vector<RankCount>(nRow == 0 ? 0 : leafReg->bagSampleTot())),
  rankScale(binScale()) {
  rankCounts(baggedRows);
}


void Quant::rankCounts(const BitMatrix *baggedRows) {
  if (nRow == 0) // Short circuits if bag information absent.
    return;

  unsigned int row = 0;
  for (auto & yr : yRanked) {
    yr.init(yTrain[row], row);
    row++;
  }
  sort(yRanked.begin(), yRanked.end(), [](const ValRow &a, const ValRow &b) -> bool {
                                         return a.val < b.val;
                                       }
    );

  vector<unsigned int> row2Rank(yRanked.size());
  unsigned int rank = 0;
  for (auto yr : yRanked) {
    row2Rank[yr.row] = rank++;
  }

  vector<unsigned int> leafSeen(leafReg->leafCount());
  fill(leafSeen.begin(), leafSeen.end(), 0);
  for (unsigned int tIdx = 0; tIdx < leafReg->getNTree(); tIdx++) {
    unsigned int bagIdx = 0;
    for (unsigned int row = 0; row < baggedRows->getNRow(); row++) {
      if (baggedRows->testBit(tIdx, row)) {
        unsigned int offset;
        unsigned int leafIdx = leafReg->getLeafIdx(tIdx, bagIdx++, offset);
        unsigned int bagOff = offset + leafSeen[leafIdx]++;
        rankCount[bagOff].init(row2Rank[row], leafReg->getSCount(bagOff));
      }
    }
  }
}

void Quant::predictAcross(const Predict *predict,
                          unsigned int rowStart,
                          unsigned int rowEnd) {
  if (nRow == 0)
    return; // Insufficient leaf information.
 
  OMPBound row;
  OMPBound rowSup = (OMPBound) rowEnd;
#pragma omp parallel default(shared) private(row) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = rowStart; row < rowSup; row++) {
      predictRow(predict, row - rowStart, &qPred[qCount * row]);
    }
  }
}


unsigned int Quant::binScale() {
  unsigned int shiftVal = 0;
  while ((binSize << shiftVal) < yRanked.size())
    shiftVal++;

  return shiftVal;
}



void Quant::predictRow(const Predict *predict,
                   unsigned int blockRow,
                   double qRow[]) {
  vector<unsigned int> sampRanks(std::min(binSize, yRanked.size()));
  fill(sampRanks.begin(), sampRanks.end(), 0);

  // Scores each rank seen at every predicted leaf.
  //
  unsigned int totSamples = 0;
  for (unsigned int tIdx = 0; tIdx < leafReg->getNTree(); tIdx++) {
    unsigned int termIdx;
    if (!predict->isBagged(blockRow, tIdx, termIdx)) {
      totSamples += leafSample(tIdx, termIdx, sampRanks);
    }
  }

  vector<double> countThreshold(qCount);
  unsigned int qSlot = 0;
  for (auto & thresh : countThreshold) {
    thresh = totSamples * quantile[qSlot++];  // Rounding properties?
  }
  
  unsigned int qIdx = 0;
  unsigned int binIdx = 0;
  unsigned int samplesSeen = 0;
  for (auto sCount : sampRanks) {
    samplesSeen += sCount;
    while (qIdx < qCount && samplesSeen >= countThreshold[qIdx]) {
      qRow[qIdx++] = binMean(binIdx);
    }
    binIdx++;
    if (qIdx >= qCount)
      break;
  }
}


double Quant::binMean(unsigned int binIdx) {
  size_t idxStart = binIdx << rankScale;
  size_t idxEnd = min(yRanked.size(), idxStart + (1 << rankScale));
  double sum = 0.0;
  unsigned int count = 0;
  for (auto idx = idxStart; idx < idxEnd; idx++) {
    sum += yRanked[idx].val;
    count++;
  }
  return count == 0 ? 0.0 : sum / count;
}


  // TODO:  For binning, rerun, restricting to "hot" bins observed
  // over sample set.  This should improve resolution for hot
  // bins.


unsigned int Quant::leafSample(unsigned int tIdx,
                               unsigned int leafIdx,
                               vector<unsigned int> &sampRanks) {
  unsigned int sampleTot = 0;
  unsigned int leafStart, leafEnd;
  leafReg->bagBounds(tIdx, leafIdx, leafStart, leafEnd);
  for (unsigned int bagIdx = leafStart; bagIdx < leafEnd; bagIdx++) {
    unsigned int sCount = rankCount[bagIdx].sCount;
    unsigned int bin = binRank(rankCount[bagIdx].rank);
    bin = 
    sampRanks[bin] += sCount;
    sampleTot += sCount;
  }

  return sampleTot;
}
