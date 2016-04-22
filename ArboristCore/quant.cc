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

//#include <iostream>
using namespace std;


/**
   @brief Constructor.  Caches parameter values and computes compressed
   leaf indices.
 */
Quant::Quant(const PredictReg *_predictReg, const LeafReg *_leafReg, const std::vector<double> &_yRanked, const std::vector<double> &_qVec, unsigned int qBin) : predictReg(_predictReg), leafReg(_leafReg), yRanked(_yRanked), qVec(_qVec), qCount(qVec.size()), logSmudge(0), sCountSmudge(0) {
  unsigned int nRow = yRanked.size();
  sampleOffset = std::vector<unsigned int>(leafReg->NodeCount());
  leafReg->SampleOffset(sampleOffset, 0, leafReg->NodeCount(), 0);
  binSize = BinSize(nRow, qBin, logSmudge);
  if (binSize < nRow) {
    SmudgeLeaves();
  }
}


/**
 */
Quant::~Quant() {
  if (sCountSmudge != 0)
    delete [] sCountSmudge;
}


/**
   @brief Fills in the quantile leaves for each row within a contiguous block.

   @param predictLeaves contains the predicted leaf indices.

   @param rowStart is the first row at which to predict.

   @param rowEnd is first row at which not to predict.

   @return void, with output parameter matrix.
 */
void Quant::PredictAcross(unsigned int rowStart, unsigned int rowEnd, double qPred[]) {
  unsigned int row;
#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = rowStart; row < rowEnd; row++) {
      Leaves(row - rowStart, &qPred[qCount * row]);
    }
  }
}


/**
   @brief Computes bin size and smudging factor.

   @param qBin is the bin size specified by the front end.

   @param logSmudge outputs the log2 of the smudging factor.

   @return bin size, with output reference parameter.
 */
unsigned int Quant::BinSize(unsigned int nRow, unsigned int qBin, unsigned int &_logSmudge) {
  logSmudge = 0;
  while ((nRow >> logSmudge) > qBin)
    logSmudge++;
  return (nRow + (1 << logSmudge) - 1) >> logSmudge;
}


/**
   @brief Builds a vector of binned sample counts for wide leaves.

   @return void.
 */
void Quant::SmudgeLeaves() {    
  sCountSmudge = new unsigned int[leafReg->BagTot()];
  for (unsigned int i = 0; i < leafReg->BagTot(); i++)
    sCountSmudge[i] = leafReg->SCount(i);
  for (unsigned int i = 0; i < leafReg->NodeCount(); i++) {
    unsigned int infoOff = sampleOffset[i];
    unsigned int extent = leafReg->Extent(i);
    if (extent > binSize) {
      int *binTemp = new int[binSize];
      for (unsigned int j = 0; j < binSize; j++)
	binTemp[j] = 0;
      for (unsigned int j = 0; j < extent; j++) {
	unsigned int sCount = leafReg->SCount(infoOff + j);
	unsigned int rank = leafReg->Rank(infoOff + j);
	binTemp[rank >> logSmudge] += sCount;
      }
      for (unsigned int j = 0; j < binSize; j++) {
	sCountSmudge[infoOff + j] = binTemp[j];
      }
      delete [] binTemp;
    }
  }
}


/**
   @brief Writes the quantile values.

   @param leaves references the per-tree leaf prediction.

   @param qRow[] outputs quantile values.


   @return void, with output vector parameter.
 */
void Quant::Leaves(unsigned int blockRow, double qRow[]) {
  unsigned int *sampRanks = new unsigned int[binSize];
  for (unsigned int i = 0; i < binSize; i++)
    sampRanks[i] = 0;

  // Scores each rank seen at every predicted leaf.
  //
  unsigned int totRanks = 0;
  for (unsigned int tn = 0; tn < leafReg->NTree(); tn++) {
    if (!predictReg->IsBagged(blockRow, tn)) {
      unsigned int leafIdx = predictReg->LeafIdx(blockRow, tn);
      totRanks += (logSmudge == 0) ? RanksExact(tn, leafIdx, sampRanks) : RanksSmudge(tn, leafIdx, sampRanks);
    }
  }

  double *countThreshold = new double[qCount];
  for (unsigned int i = 0; i < qCount; i++) {
    countThreshold[i] = totRanks * qVec[i];  // Rounding properties?
  }
  
  unsigned int qIdx = 0;
  unsigned int rankIdx = 0;
  unsigned int rankCount = 0;
  unsigned int smudge = (1 << logSmudge);
  for (unsigned int i = 0; i < binSize && qIdx < qCount; i++) {
    rankCount += sampRanks[i];
    while (qIdx < qCount && rankCount >= countThreshold[qIdx]) {
      qRow[qIdx++] = yRanked[rankIdx];
    }
    rankIdx += smudge;
  }

  // TODO:  For binning, rerun, restricting to "hot" bins observed
  // over sample set.  This should improve resolution for hot
  // bins.
  delete [] sampRanks;
  delete [] countThreshold;
}


/**
   @brief Accumulates the ranks assocated with predicted leaf.

   @param leafExtent enumerates sample indices associated with a leaf.

   @param leafOff is the forest starting index of leaf's rank values.

   @param sampRanks outputs the count of samples at a given rank.

   @return count of ranks introduced by leaf.
 */
unsigned int Quant::RanksExact(unsigned int tIdx, unsigned int leafIdx, unsigned int sampRanks[]) {
  int rankTot = 0;
  unsigned int infoOff = sampleOffset[leafReg->NodeIdx(tIdx, leafIdx)];
  for (unsigned int i = 0; i < leafReg->Extent(tIdx, leafIdx); i++) {
    unsigned int sCount = leafReg->SCount(infoOff + i);
    unsigned int rank = leafReg->Rank(infoOff + i);
    sampRanks[rank] += sCount;
    rankTot += sCount;
  }

  return rankTot;
}


/**
   @brief Accumulates binned ranks assocated with a predicted leaf.

   @param leafExtent enumerates sample indices associated with a leaf.

   @param leafOff is the forest starting index of leaf's rank values.

   @param sampRanks[] outputs the binned rank counts.

   @return count of ranks introduced by leaf.
 */
unsigned int Quant::RanksSmudge(unsigned int tIdx, unsigned int leafIdx, unsigned int sampRanks[]) {
  unsigned int rankTot = 0;
  unsigned int extent = leafReg->Extent(tIdx, leafIdx);
  unsigned int infoOff = sampleOffset[leafReg->NodeIdx(tIdx, leafIdx)];
  if (extent <= binSize) {
    for (unsigned int i = 0; i < extent; i++) {
      unsigned int rankIdx = (leafReg->Rank(infoOff + i) >> logSmudge);
      unsigned int rankCount = sCountSmudge[infoOff + i];
      sampRanks[rankIdx] += rankCount;
      rankTot += rankCount;
    }
  }
  else {
    for (unsigned int rankIdx = 0; rankIdx < binSize; rankIdx++) {
      unsigned int rankCount = sCountSmudge[infoOff + rankIdx];
      sampRanks[rankIdx] += rankCount;
      rankTot += rankCount;
    }
  }

  return rankTot;
}

