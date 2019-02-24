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


/**
   @brief Constructor.  Caches parameter values and computes compressed
   leaf indices.
 */
Quant::Quant(const LeafFrameReg* leafReg_,
             const BitMatrix* baggedRows,
             const double* quantile_,
             unsigned int qCount_,
             unsigned int qBin) :
  leafReg(leafReg_),
  yTrain(leafReg->YTrain()),
  yRanked(vector<ValRow>(baggedRows->getNRow())),
  quantile(quantile_),
  qCount(qCount_),
  qPred(vector<double>(leafReg->rowPredict() * qCount)),
  rankCount(vector<RankCount>(leafReg->bagSampleTot())),
  logSmudge(0) {
  if (rankCount.size() == 0) // Short circuits if bag information absent.
    return;

  rankCounts(baggedRows);
  binSize = imputeBinSize(yRanked.size(), qBin, logSmudge);
  if (binSize < yRanked.size()) {
    smudgeLeaves();
  }
}


void Quant::rankCounts(const BitMatrix *baggedRows) {
  for (unsigned int row = 0; row < yRanked.size(); row++) {
    yRanked[row].init(yTrain[row], row);
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
  unsigned int bagIdx = 0;
  for (unsigned int tIdx = 0; tIdx < leafReg->getNTree(); tIdx++) {
    for (unsigned int row = 0; row < baggedRows->getNRow(); row++) {
      if (baggedRows->testBit(tIdx, row)) {
        unsigned int offset;
        unsigned int leafIdx = leafReg->getLeafIdx(tIdx, bagIdx, offset);
        unsigned int bagOff = offset + leafSeen[leafIdx]++;
        rankCount[bagOff].init(row2Rank[row], leafReg->getSCount(bagOff));
        bagIdx++;
      }
    }
  }
}


void Quant::predictAcross(const Predict *predict,
                          unsigned int rowStart,
                          unsigned int rowEnd) {
  if (rankCount.size() == 0)
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


unsigned int Quant::imputeBinSize(unsigned int rowTrain,
                            unsigned int qBin,
                            unsigned int &_logSmudge) {
  logSmudge = 0;
  while ((rowTrain >> logSmudge) > qBin)
    logSmudge++;
  return (rowTrain + (1 << logSmudge) - 1) >> logSmudge;
}


void Quant::smudgeLeaves() {
  sCountSmudge = move(vector<unsigned int>(leafReg->bagSampleTot()));
  for (unsigned int i = 0; i < sCountSmudge.size(); i++)
    sCountSmudge[i] = rankCount[i].sCount;

  binTemp = move(vector<unsigned int>(binSize));
  for (unsigned int leafIdx = 0; leafIdx < leafReg->leafCount(); leafIdx++) {
    unsigned int leafStart, leafEnd;
    leafReg->bagBounds(0, leafIdx, leafStart, leafEnd);
    if (leafEnd - leafStart > binSize) {
      fill(binTemp.begin(), binTemp.end(), 0);
      for (unsigned int bagIdx = leafStart; bagIdx < leafEnd; bagIdx++) {
        unsigned int sCount = rankCount[bagIdx].sCount;
        unsigned int rank = rankCount[bagIdx].rank;
        binTemp[rank >> logSmudge] += sCount;
      }
      for (unsigned int j = 0; j < binSize; j++) {
        sCountSmudge[leafStart + j] = binTemp[j];
      }
    }
  }
}


void Quant::predictRow(const Predict *predict,
                   unsigned int blockRow,
                   double qRow[]) {
  vector<unsigned int> sampRanks(binSize);
  fill(sampRanks.begin(), sampRanks.end(), 0);

  // Scores each rank seen at every predicted leaf.
  //
  unsigned int totRanks = 0;
  for (unsigned int tIdx = 0; tIdx < leafReg->getNTree(); tIdx++) {
    unsigned int termIdx;
    if (!predict->isBagged(blockRow, tIdx, termIdx)) {
      totRanks += (logSmudge == 0) ? ranksExact(tIdx, termIdx, sampRanks) : ranksSmudge(tIdx, termIdx, sampRanks);
    }
  }

  vector<double> countThreshold(qCount);
  for (unsigned int qSlot = 0; qSlot < qCount; qSlot++) {
    countThreshold[qSlot] = totRanks * quantile[qSlot];  // Rounding properties?
  }
  
  unsigned int qIdx = 0;
  unsigned int rkIdx = 0;
  unsigned int rkCount = 0;
  unsigned int smudge = (1 << logSmudge);
  for (unsigned int i = 0; i < binSize && qIdx < qCount; i++) {
    rkCount += sampRanks[i];
    while (qIdx < qCount && rkCount >= countThreshold[qIdx]) {
      qRow[qIdx++] = yRanked[rkIdx].val;
    }
    rkIdx += smudge;
  }

  // TODO:  For binning, rerun, restricting to "hot" bins observed
  // over sample set.  This should improve resolution for hot
  // bins.
}


unsigned int Quant::ranksExact(unsigned int tIdx,
                               unsigned int leafIdx,
                               vector<unsigned int> &sampRanks) {
  int rankTot = 0;
  unsigned int leafStart, leafEnd;
  leafReg->bagBounds(tIdx, leafIdx, leafStart, leafEnd);
  for (unsigned int bagIdx = leafStart; bagIdx < leafEnd; bagIdx++) {
    unsigned int sCount = rankCount[bagIdx].sCount;
    unsigned int rank = rankCount[bagIdx].rank;
    sampRanks[rank] += sCount;
    rankTot += sCount;
  }

  return rankTot;
}


unsigned int Quant::ranksSmudge(unsigned int tIdx,
                                unsigned int leafIdx,
                                vector<unsigned int> &sampRanks) {
  unsigned int rankTot = 0;
  unsigned int leafStart, leafEnd;
  leafReg->bagBounds(tIdx, leafIdx, leafStart, leafEnd);
  if (leafEnd - leafStart <= binSize) {
    for (auto bagIdx = leafStart; bagIdx < leafEnd; bagIdx++) {
      unsigned int rkIdx = (rankCount[bagIdx].rank >> logSmudge);
      auto rkCount = sCountSmudge[bagIdx];
      sampRanks[rkIdx] += rkCount;
      rankTot += rkCount;
    }
  }
  else {
    for (unsigned int rkIdx = 0; rkIdx < binSize; rkIdx++) {
      auto rkCount = sCountSmudge[leafStart + rkIdx];
      sampRanks[rkIdx] += rkCount;
      rankTot += rkCount;
    }
  }

  return rankTot;
}
