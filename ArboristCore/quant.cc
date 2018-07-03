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
#include <algorithm>


/**
   @brief Constructor.  Caches parameter values and computes compressed
   leaf indices.
 */
Quant::Quant(const LeafReg *leafReg_,
             const BitMatrix *baggedRows,
             const vector<double> &quantile_,
             unsigned int qBin) :
  leafReg(leafReg_),
  yTrain(leafReg->YTrain()),
  yRanked(vector<RankedPair>(baggedRows->getNRow())),
  quantile(quantile_),
  qCount(quantile.size()),
  qPred(vector<double>(leafReg->rowPredict() * qCount)),
  rankCount(vector<RankCount>(leafReg->BagLeafTot())),
  logSmudge(0) {
  if (rankCount.size() == 0) // Insufficient leaf information.
    return;

  unsigned int rowTrain = yRanked.size();
  for (unsigned int row = 0; row < rowTrain; row++) {
    yRanked[row] = make_pair(yTrain[row], row);
  }
  sort(yRanked.begin(), yRanked.end());
  rankCounts(baggedRows);
  binSize = BinSize(rowTrain, qBin, logSmudge);
  if (binSize < rowTrain) {
    SmudgeLeaves();
  }
}

/**
   @brief Computes the count and rank of every bagged sample in the forest.
   Quantile prediction is the only client.

   @return void.
 */
void Quant::rankCounts(const BitMatrix *baggedRows) {
  vector<unsigned int> leafSeen(leafReg->getLeafCount());
  fill(leafSeen.begin(), leafSeen.end(), 0);
  vector<unsigned int> row2Rank(yRanked.size());
  for (unsigned int rank = 0; rank < row2Rank.size(); rank++) {
    row2Rank[yRanked[rank].second] = rank;
  }

  unsigned int bagIdx = 0;
  for (unsigned int tIdx = 0; tIdx < leafReg->getNTree(); tIdx++) {
    for (unsigned int row = 0; row < baggedRows->getNRow(); row++) {
      if (baggedRows->testBit(tIdx, row)) {
        unsigned int offset;
        unsigned int leafIdx = leafReg->getLeafIdx(tIdx, bagIdx, offset);
        unsigned int bagOff = offset + leafSeen[leafIdx]++;
        rankCount[bagOff].Init(row2Rank[row], leafReg->getSCount(bagOff));
        bagIdx++;
      }
    }
  }
}


/**
   @brief Fills in the quantile leaves for each row within a contiguous block.

   @param rowStart is the first row at which to predict.

   @param rowEnd is first row at which not to predict.

   @param qPred outputs the quantile values.

   @return void, with output parameter matrix.
 */
void Quant::PredictAcross(const Predict *predict,
                          unsigned int rowStart,
                          unsigned int rowEnd) {
  if (rankCount.size() == 0)
    return; // Insufficient leaf information.
 
  OMPBound row;
  OMPBound rowSup = (OMPBound) rowEnd;
#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = rowStart; row < rowSup; row++) {
      Leaves(predict, row - rowStart, &qPred[qCount * row]);
    }
  }
}


/**
   @brief Computes bin size and smudging factor.

   @param rowTrain is the number of rows used to train.

   @param qBin is the bin size specified by the front end.

   @param logSmudge outputs the log2 of the smudging factor.

   @return bin size, with output reference parameter.
 */
unsigned int Quant::BinSize(unsigned int rowTrain,
                            unsigned int qBin,
                            unsigned int &_logSmudge) {
  logSmudge = 0;
  while ((rowTrain >> logSmudge) > qBin)
    logSmudge++;
  return (rowTrain + (1 << logSmudge) - 1) >> logSmudge;
}


/**
   @brief Builds a vector of binned sample counts for wide leaves.

   @return void.
 */
void Quant::SmudgeLeaves() {
  sCountSmudge = move(vector<unsigned int>(leafReg->BagLeafTot()));
  for (unsigned int i = 0; i < sCountSmudge.size(); i++)
    sCountSmudge[i] = rankCount[i].sCount;

  binTemp = move(vector<unsigned int>(binSize));
  for (unsigned int leafIdx = 0; leafIdx < leafReg->getLeafCount(); leafIdx++) {
    unsigned int leafStart, leafEnd;
    leafReg->BagBounds(0, leafIdx, leafStart, leafEnd);
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


/**
   @brief Writes the quantile values for a given row.

   @param blockRow is the block-relative row index.

   @param qRow[] outputs the 'qCount' quantile values.

   @return void, with output vector parameter.
 */
void Quant::Leaves(const Predict *predict,
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
      totRanks += (logSmudge == 0) ? RanksExact(tIdx, termIdx, sampRanks) : RanksSmudge(tIdx, termIdx, sampRanks);
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
      qRow[qIdx++] = yRanked[rkIdx].first;
    }
    rkIdx += smudge;
  }

  // TODO:  For binning, rerun, restricting to "hot" bins observed
  // over sample set.  This should improve resolution for hot
  // bins.
}


/**
   @brief Accumulates the ranks assocated with predicted leaf.

   @param leafExtent enumerates sample indices associated with a leaf.

   @param sampRanks outputs the count of samples at a given rank.

   @return count of ranks introduced by leaf.
 */
unsigned int Quant::RanksExact(unsigned int tIdx,
                               unsigned int leafIdx,
                               vector<unsigned int> &sampRanks) {
  int rankTot = 0;
  
  unsigned int leafStart, leafEnd;
  leafReg->BagBounds(tIdx, leafIdx, leafStart, leafEnd);
  for (unsigned int bagIdx = leafStart; bagIdx < leafEnd; bagIdx++) {
    unsigned int sCount = rankCount[bagIdx].sCount;
    unsigned int rank = rankCount[bagIdx].rank;
    sampRanks[rank] += sCount;
    rankTot += sCount;
  }

  return rankTot;
}


/**
   @brief Accumulates binned ranks assocated with a predicted leaf.

   @param tIdx is the tree index.

   @param leafIdx is the tree-relative leaf index.

   @param sampRanks[] outputs the binned rank counts.

   @return count of ranks introduced by leaf.
 */
unsigned int Quant::RanksSmudge(unsigned int tIdx,
                                unsigned int leafIdx,
                                vector<unsigned int> &sampRanks) {
  unsigned int rankTot = 0;
  unsigned int leafStart, leafEnd;
  leafReg->BagBounds(tIdx, leafIdx, leafStart, leafEnd);
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
