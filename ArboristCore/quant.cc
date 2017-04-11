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
#include "leaf.h"
#include "predict.h"
#include <algorithm>

//#include <iostream>
using namespace std;


/**
   @brief Constructor.  Caches parameter values and computes compressed
   leaf indices.
 */
Quant::Quant(const PredictReg *_predictReg, const LeafPerfReg *_leafReg, const std::vector<double> &_qVec, unsigned int qBin) : predictReg(_predictReg), leafReg(_leafReg), yTrain(predictReg->YTrain()), yRanked(std::vector<RankedPair>(yTrain.size())), qVec(_qVec), qCount(qVec.size()), rankCount(std::vector<RankCount>(leafReg->BagLeafTot())), logSmudge(0) {
  if (rankCount.size() == 0) // Insufficient leaf information.
    return;
  unsigned int rowTrain = yRanked.size();
  for (unsigned int row = 0; row < rowTrain; row++) {
    yRanked[row] = std::make_pair(yTrain[row], row);
  }
  std::sort(yRanked.begin(), yRanked.end());
  std::vector<unsigned int> row2Rank(rowTrain);
  for (unsigned int rank = 0; rank < rowTrain; rank++) {
    row2Rank[yRanked[rank].second] = rank;
  }
  leafReg->RankCounts(row2Rank, rankCount);

  binSize = BinSize(rowTrain, qBin, logSmudge);
  if (binSize < rowTrain) {
    SmudgeLeaves();
  }
}


/**
   @brief Fills in the quantile leaves for each row within a contiguous block.

   @param rowStart is the first row at which to predict.

   @param rowEnd is first row at which not to predict.

   @param qPred outputs the quantile values.

   @return void, with output parameter matrix.
 */
void Quant::PredictAcross(unsigned int rowStart, unsigned int rowEnd, double qPred[]) {
  if (rankCount.size() == 0)
    return; // Insufficient leaf information.
 
  int row;
#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = rowStart; row < int(rowEnd); row++) {
      Leaves(row - rowStart, &qPred[qCount * row]);
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
unsigned int Quant::BinSize(unsigned int rowTrain, unsigned int qBin, unsigned int &_logSmudge) {
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
  sCountSmudge = std::move(std::vector<unsigned int>(leafReg->BagLeafTot()));
  for (unsigned int i = 0; i < sCountSmudge.size(); i++)
    sCountSmudge[i] = rankCount[i].sCount;

  binTemp = std::move(std::vector<unsigned int>(binSize));
  for (unsigned int leafIdx = 0; leafIdx < leafReg->LeafCount(); leafIdx++) {
    unsigned int leafStart, leafEnd;
    leafReg->BagBounds(0, leafIdx, leafStart, leafEnd);
    if (leafEnd - leafStart > binSize) {
      std::fill(binTemp.begin(), binTemp.end(), 0);
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
void Quant::Leaves(unsigned int blockRow, double qRow[]) {
  std::vector<unsigned int> sampRanks(binSize);
  std::fill(sampRanks.begin(), sampRanks.end(), 0);

  // Scores each rank seen at every predicted leaf.
  //
  unsigned int totRanks = 0;
  for (unsigned int tIdx = 0; tIdx < leafReg->NTree(); tIdx++) {
    if (!predictReg->IsBagged(blockRow, tIdx)) {
      unsigned int leafIdx = predictReg->LeafIdx(blockRow, tIdx);
      totRanks += (logSmudge == 0) ? RanksExact(tIdx, leafIdx, sampRanks) : RanksSmudge(tIdx, leafIdx, sampRanks);
    }
  }

  std::vector<double> countThreshold(qCount);
  for (unsigned int qSlot = 0; qSlot < qCount; qSlot++) {
    countThreshold[qSlot] = totRanks * qVec[qSlot];  // Rounding properties?
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
unsigned int Quant::RanksExact(unsigned int tIdx, unsigned int leafIdx, std::vector<unsigned int> &sampRanks) {
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
unsigned int Quant::RanksSmudge(unsigned int tIdx, unsigned int leafIdx, std::vector<unsigned int> &sampRanks) {
  unsigned int rankTot = 0;

  unsigned int leafStart, leafEnd;
  leafReg->BagBounds(tIdx, leafIdx, leafStart, leafEnd);
  if (leafEnd - leafStart <= binSize) {
    for (unsigned int bagIdx = leafStart; bagIdx < leafEnd; bagIdx++) {
      unsigned int rkIdx = (rankCount[bagIdx].rank >> logSmudge);
      unsigned int rkCount = sCountSmudge[bagIdx];
      sampRanks[rkIdx] += rkCount;
      rankTot += rkCount;
    }
  }
  else {
    for (unsigned int rkIdx = 0; rkIdx < binSize; rkIdx++) {
      unsigned int rkCount = sCountSmudge[leafStart + rkIdx];
      sampRanks[rkIdx] += rkCount;
      rankTot += rkCount;
    }
  }

  return rankTot;
}

