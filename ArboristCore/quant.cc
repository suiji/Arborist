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

//#include <iostream>
using namespace std;


/**
   @brief Constructor.  Caches parameter values and computes compressed
   leaf indices.
 */
Quant::Quant(const Forest *_forest, const std::vector<double> &_yRanked, const std::vector<unsigned int> &_rank, const std::vector<unsigned int> &_sCount, const std::vector<double> &_qVec,  unsigned int qBin) : forest(_forest), height(forest->Height()), nTree(forest->NTree()), yRanked(_yRanked), rank(_rank), sCount(_sCount), qVec(_qVec), qCount(qVec.size()), logSmudge(0), sCountSmudge(0) {
  unsigned int nRow = yRanked.size();
  leafPos = forest->ExtentPosition();
  binSize = BinSize(nRow, qBin, logSmudge);
  if (binSize < nRow) {
    SmudgeLeaves();
  }
}


/**
 */
Quant::~Quant() {
  delete [] leafPos;
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
void Quant::PredictAcross(const int predictLeaves[], unsigned int rowStart, unsigned int rowEnd, double qPred[]) {
  unsigned int row;
#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = rowStart; row < rowEnd; row++) {
      Leaves(predictLeaves + nTree * (row - rowStart), &qPred[qCount * row]);
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
  sCountSmudge = new unsigned int[sCount.size()];
  for (unsigned int i = 0; i < sCount.size(); i++)
    sCountSmudge[i] = sCount[i];
  for (int i = 0; i < height; i++) {
    int leafOff = leafPos[i];
    if (leafOff >= 0) {
      unsigned int leafSize = forest->Extent(i);
      if (leafSize > binSize) {
	int *binTemp = new int[binSize];
	for (unsigned int j = 0; j < binSize; j++)
	  binTemp[j] = 0;
	for (unsigned int j = 0; j < leafSize; j++) {
	  unsigned int rk = rank[leafOff + j];
	  binTemp[rk >> logSmudge] += sCount[leafOff + j];
	}
	for (unsigned int j = 0; j < binSize; j++) {
	  sCountSmudge[leafOff + j] = binTemp[j];
	}
	delete [] binTemp;
      }
    }
  }
}


/**
   @brief Writes the quantile values.

   @param leaves references the per-tree leaf prediction.

   @param qRow[] outputs quantile values.


   @return void, with output vector parameter.
 */
void Quant::Leaves(const int rowLeaves[], double qRow[]) {
  int *sampRanks = new int[binSize];
  for (unsigned int i = 0; i < binSize; i++)
    sampRanks[i] = 0;

  // Scores each rank seen at every predicted leaf.
  //
  int totRanks = 0;
  for (int tn = 0; tn < nTree; tn++) {
    int leafIdx = rowLeaves[tn];
    if (leafIdx >= 0) { // otherwise in-bag:  no prediction for tree at row.
      int leafOff = forest->LeafPos(tn, leafIdx); // Absolute forest offset.
      if (logSmudge == 0) {
        totRanks += RanksExact(forest->Extent(leafOff), leafPos[leafOff], sampRanks);
      }
      else {
        totRanks += RanksSmudge(forest->Extent(leafOff), leafPos[leafOff], sampRanks);
      }
    }
  }

  double *countThreshold = new double[qCount];
  for (int i = 0; i < qCount; i++) {
    countThreshold[i] = totRanks * qVec[i];  // Rounding properties?
  }
  
  int qIdx = 0;
  int rankIdx = 0;
  int rankCount = 0;
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
int Quant::RanksExact(int leafExtent, int leafOff, int sampRanks[]) {
  int rankTot = 0;
  for (int i = 0; i < leafExtent; i++) {
    unsigned int leafRank = rank[leafOff+i];
    int rankCount = sCount[leafOff + i];
    sampRanks[leafRank] += rankCount;
    rankTot += rankCount;
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
int Quant::RanksSmudge(unsigned int leafExtent, int leafOff, int sampRanks[]) {
  int rankTot = 0;
  if (leafExtent <= binSize) {
    for (unsigned int i = 0; i < leafExtent; i++) {
      int rankIdx = (rank[leafOff+i] >> logSmudge);
      int rankCount = sCountSmudge[leafOff + i];
      sampRanks[rankIdx] += rankCount;
      rankTot += rankCount;
    }
  }
  else {
    for (unsigned int rankIdx = 0; rankIdx < binSize; rankIdx++) {
      int rankCount = sCountSmudge[leafOff + rankIdx];
      sampRanks[rankIdx] += rankCount;
      rankTot += rankCount;
    }
  }

  return rankTot;
}

