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
   @brief Static entry for quantile prediction.

   @param qVec is a front-end vector of quantiles to evaluate.

   @param qCount is the number of elements in _qVec.

   @param qBin is the row-count threshold for binning.

   @param qPred outputs the prediction.

   @return error code from Forest static entry.
 */
void Quant::Predict(const ForestReg *forestReg, const double _qVec[], int _qCount, unsigned int _qBin, int *predictLeaves, double qPred[]) {
  int _nTree;
  unsigned int _nRow;
  int *_origin, *_nonTerm, *_extent, *_rank, *_sCount;
  double *_yRanked;
  int _height = forestReg->QuantFields(_nTree, _nRow, _origin, _nonTerm, _extent, _yRanked, _rank, _sCount);
  Quant *quant = new Quant(_height, _nTree, _nRow, _origin, _nonTerm, _extent, _yRanked, _rank, _sCount, _qVec, _qCount, _qBin);
  quant->PredictRows(predictLeaves, qPred);

  delete quant;
}


Quant::Quant(int _height, int _nTree, unsigned int _nRow, int *_origin, int *_nonTerm, int *_extent, double *_yRanked, int *_rank, int *_sCount, const double _qVec[], int _qCount, unsigned int _qBin) : height(_height), nTree(_nTree), nRow(_nRow), origin(_origin), extent(_extent), yRanked(_yRanked), rank(_rank), sCount(_sCount) , qVec(_qVec), qCount(_qCount), qBin(_qBin) {
  LeafPositions(_nonTerm);
}

Quant::~Quant() {
  delete [] leafPos;
}


/**
   @brief Marks absolute starting offset of each leaf in forest, or -1
   to identify nonterminals.

   @return void, with vector side effect.
 */
void Quant::LeafPositions(int nonTerm[]) {
  leafPos = new int[height];
  int idxAccum = 0;
  for (int i = 0; i < height; i++) {
    if (nonTerm[i] == 0) {
      leafPos[i] = idxAccum;
      idxAccum += extent[i];
    }
    else {
      leafPos[i] = -1;
    }
  }
}


/**
   @brief Fills in the quantile leaves for each row.

   @param predictLeavs outputs the matrix of quantile leaves.

   @return void, with output parameter matrix.
 */
void Quant::PredictRows(int predictLeaves[], double qPred[]) {
  unsigned int logSmudge;
  unsigned int binSize = SmudgeLeaves(logSmudge);

  unsigned int row;
#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = 0; row < nRow; row++) {
      Leaves(predictLeaves + nTree * row, qPred + qCount * row, binSize, logSmudge);
    }
  }
}


/**
   @brief Overwrites sample counts for wide leaves with binned values.

   @return bin size.
 */
unsigned int Quant::SmudgeLeaves(unsigned int &logSmudge) {
  logSmudge = 0;
  while ((nRow >> logSmudge) > qBin)
    logSmudge++;
  if (logSmudge == 0)
    return nRow;

  unsigned int binSize = (nRow + (1 << logSmudge) - 1) >> logSmudge;
  for (int i = 0; i < height; i++) {
    int leafOff = LeafPos(i);
    if (leafOff >= 0) {
      unsigned int leafSize = extent[i];
      if (leafSize > binSize) {
	int *binTemp = new int[binSize];
	for (unsigned int j = 0; j < binSize; j++)
	  binTemp[j] = 0;
	for (unsigned int j = 0; j < leafSize; j++) {
	  unsigned int rk = rank[leafOff + j];
	  binTemp[rk >> logSmudge] += sCount[leafOff + j];
	}
	for (unsigned int j = 0; j < binSize; j++) {
	  sCount[leafOff + j] = binTemp[j];
	}
	delete [] binTemp;
      }
    }
  }

  return binSize;
}


/**
   @brief Writes the quantile values.

   @param leaves references the per-tree leaf prediction.

   @param qRow[] outputs quantile values.

   @return void, with output vector parameter.
 */
void Quant::Leaves(const int rowPredict[], double qRow[], unsigned int binSize, unsigned int logSmudge) {
  int *sampRanks = new int[binSize];
  for (unsigned int i = 0; i < binSize; i++)
    sampRanks[i] = 0;

  // Scores each rank seen at every predicted leaf.
  //
  int totRanks = 0;
  for (int tn = 0; tn < nTree; tn++) {
    int leafIdx = rowPredict[tn];
    if (leafIdx >= 0) { // otherwise in-bag:  no prediction for tree at row.
      int leafOff = origin[tn] + leafIdx; // Absolute forest offset of leaf.
      if (logSmudge == 0) {
        totRanks += RanksExact(extent[leafOff], LeafPos(leafOff), sampRanks);
      }
      else {
        totRanks += RanksSmudge(extent[leafOff], LeafPos(leafOff), sampRanks, binSize, logSmudge);
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
int Quant::RanksSmudge(unsigned int leafExtent, int leafOff, int sampRanks[], unsigned int binSize, unsigned int logSmudge) {
  int rankTot = 0;
  if (leafExtent <= binSize) {
    for (unsigned int i = 0; i < leafExtent; i++) {
      int rankIdx = (rank[leafOff+i] >> logSmudge);
      int rankCount = sCount[leafOff + i];
      sampRanks[rankIdx] += rankCount;
      rankTot += rankCount;
    }
  }
  else {
    for (unsigned int rankIdx = 0; rankIdx < binSize; rankIdx++) {
      int rankCount = sCount[leafOff + rankIdx];
      sampRanks[rankIdx] += rankCount;
      rankTot += rankCount;
    }
  }

  return rankTot;
}

