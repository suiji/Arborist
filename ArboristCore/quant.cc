// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file quant.cc

   @brief Methods for predicting and writing quantiles.

   @author Mark Seligman
 */

#include "quant.h"
#include "response.h"
#include "sample.h"

bool Quant::live = false;
int Quant::nTree = -1;
int Quant::nRow = -1;

int Quant::qCount = 0;
double *Quant::qVec = 0;
double *Quant::qPred = 0;


// Nonzero iff quantiles stipulated for training.
int *Quant::treeQRankWidth = 0;
int **Quant::treeQLeafPos = 0;
int **Quant::treeQLeafExtent = 0;
int **Quant::treeQRank = 0;
int **Quant::treeQRankCount = 0;

int Quant::qYLenForest = -1;
int Quant::totBagCount = -1;
int Quant::forestSize = -1;

// Nonzero iff quantiles tabulated.
double *Quant::qYRankedForest = 0;
int *Quant::qRankOriginForest = 0;
int *Quant::qRankForest = 0;
int *Quant::qRankCountForest = 0;
int *Quant::qLeafPosForest = 0;
int *Quant::qLeafExtentForest = 0;

/**
   @brief Entry for training path.

   @param _train indicates whether quantile training has been requested.

   @param nTree is the number of trees requested.

   @return void.
 */
void Quant::FactoryTrain(int _nRow, int _nTree, bool _train) {
  live = _train;
  if (!live)
    return;

  nRow = _nRow;
  nTree = _nTree;
  totBagCount = 0;
  forestSize = 0;

  treeQRankWidth = new int[nTree];
  treeQLeafPos = new int*[nTree];
  treeQLeafExtent = new int*[nTree];
  treeQRank = new int*[nTree];
  treeQRankCount = new int*[nTree];

  for (int i = 0; i < nTree; i++) { // Actually not necessary, as all trees have >= 1 leaf.
    treeQRankWidth[i] = 0;
    treeQLeafPos[i] = 0;
    treeQLeafExtent[i] = 0;
    treeQRank[i] = 0;
  }
}

/**
   @brief Writes quantile data into storage provided by front end and finalizes.

   @return void, with output vector parameters.

 */
void Quant::Write(double rQYRanked[], int rQRankOrigin[], int rQRank[], int rQRankCount[], int rQLeafPos[], int rQLeafExtent[]) {
  // ASSERTION
  //  if (!live)
  //cout << "Attempting to write and deallocate unallocated quantile" << endl;

  for (int i = 0; i < nRow; i++)
    rQYRanked[i] = qYRankedForest[i];

  for (int tn = 0; tn < nTree; tn++)
    rQRankOrigin[tn] = qRankOriginForest[tn];

  for (int i = 0; i < totBagCount; i++) {
    rQRank[i] = qRankForest[i];
    rQRankCount[i] = qRankCountForest[i];
  }
  for (int i = 0; i < forestSize; i++) {
    rQLeafPos[i] = qLeafPosForest[i];
    rQLeafExtent[i] = qLeafExtentForest[i];
  }

  delete [] qYRankedForest;
  delete [] qRankOriginForest;
  delete [] qRankForest;
  delete [] qRankCountForest;
  delete [] qLeafPosForest;
  delete [] qLeafExtentForest;

  nRow = nTree = totBagCount = forestSize = qYLenForest = -1;
  qYRankedForest = 0;
  qRankOriginForest = 0;
  qRankForest = 0;
  qRankCountForest = 0;
  qLeafPosForest = 0;
  qLeafExtentForest = 0;

  // Tree-based quantile data.
  treeQRankWidth = 0;
  treeQLeafPos = 0;
  treeQLeafExtent = 0;
  treeQRank = 0;
  treeQRankCount = 0;;

  live = false;
}

/**
  @brief Loads quantile data stored by the front end.

  @return void.
*/
void Quant::FactoryPredict(int _nTree, double qYRanked[], int qYLen, int qRankOrigin[], int qRank[], int qRankCount[], int qLeafPos[], int qLeafExtent[]) {
  live = true;
  nTree = _nTree;
  qYRankedForest = qYRanked;
  qYLenForest = qYLen;
  qRankOriginForest = qRankOrigin;
  qRankForest = qRank;
  qRankCountForest = qRankCount;
  qLeafPosForest = qLeafPos;
  qLeafExtentForest= qLeafExtent;
}


/**
   @brief Sets the global parameters for quantile prediction using storage provided by the front end.

   @param _qVec is a front-end vector of quantiles to evaluate.

   @param _qCount is the number of elements in _qVec.

   @param _qPred outputs the prediction.

   @param _nRow is an optional row count.  If unset, the current value is maintained.

   @return void.
 */
void Quant::EntryPredict(double _qVec[], int _qCount, double _qPred[], int _nRow) {
  if (_nRow > 0)
    nRow = _nRow;
  qCount = _qCount;
  qVec = _qVec;
  qPred = _qPred;
}


/**
   @basic Finalizer for prediction-only path.

   @return void.
 */
void Quant::DeFactoryPredict() {
  live = false;
  nTree = qYLenForest = -1;
  qCount = 0;
  qVec = qPred = 0;
  qYRankedForest = 0;
  qRankOriginForest = qRankForest = qLeafPosForest = qLeafExtentForest = 0;
}


/**
   @brief Consumes per-tree quantile information into forest-wide vectors.

   @return void.//sum of quantile leaf widths for the forest.
 */
void Quant::ConsumeTrees(const int treeOriginForest[], int _forestSize) {
  if (!live)
    return;

  forestSize = _forestSize;
  totBagCount = 0;

  qYRankedForest = new double[nRow];
  qYLenForest = nRow;
  ResponseReg::GetYRanked(qYRankedForest);

  qRankOriginForest = new int[nTree];
  for (int tn = 0; tn < nTree; tn++) {
    qRankOriginForest[tn] = totBagCount;
    totBagCount += treeQRankWidth[tn]; // bagCount
  }

  qRankForest = new int[totBagCount];
  qRankCountForest = new int[totBagCount];
  qLeafPosForest = new int[forestSize];
  qLeafExtentForest = new int[forestSize];
  for (int tn = 0; tn < nTree; tn++) {
    int *qRank = qRankForest + qRankOriginForest[tn];
    int *qRankCount = qRankCountForest + qRankOriginForest[tn];
    for (int i = 0; i < treeQRankWidth[tn]; i++) {
      qRank[i] = treeQRank[tn][i];
      qRankCount[i] = treeQRankCount[tn][i];
    }
    delete [] treeQRank[tn];
    delete [] treeQRankCount[tn];

    int *qLeafPos = qLeafPosForest + treeOriginForest[tn];
    int *qLeafExtent = qLeafExtentForest + treeOriginForest[tn];
    int end = (tn < (nTree - 1) ? treeOriginForest[tn + 1] : forestSize);
    end -= treeOriginForest[tn];
    for (int i = 0; i < end; i++) {
      qLeafPos[i] = treeQLeafPos[tn][i];
      qLeafExtent[i] = treeQLeafExtent[tn][i];
    }
    delete [] treeQLeafPos[tn];
    delete [] treeQLeafExtent[tn];
  }

  delete [] treeQRankWidth;
  delete [] treeQRank;
  delete [] treeQRankCount;
  delete [] treeQLeafPos;
  delete [] treeQLeafExtent;
}

/**
  @brief Transfers quantile data structures from pretree to decision tree.

  @param tn is the index of the tree under construction.

  @param treeSize is the number of pretree nodes.

  @param bagCount is the size of the in-bag set.

  @return void.
*/
void Quant::TreeRanks(int tn, int treeSize, int bagCount) {
  if (!live)
    return;

  treeQRankWidth[tn] = bagCount;

  int *qLeafPos = new int[treeSize];
  treeQLeafPos[tn] = qLeafPos;
  int *qLeafExtent = new int[treeSize];
  treeQLeafExtent[tn] = qLeafExtent;

  int *qRank  = new int[bagCount];
  treeQRank[tn] = qRank;
  int *qRankCount = new int[bagCount];
  treeQRankCount[tn] = qRankCount;

  SampleReg::TreeQuantiles(treeSize, bagCount, qLeafPos, qLeafExtent, qRank, qRankCount);
}

/**
   @brief Fills in the quantile leaves for each row.

   @param predictLeavs outputs the matrix of quantile leaves.

   @return void, with output parameter matrix.
 */
void Quant::PredictRows(const int treeOriginForest[], int *predictLeaves) {
  if (live) {
    int row;
#pragma omp parallel default(shared) private(row)
  {
    for (row = 0; row < nRow; row++) {
      double *qRow = qPred + row * qCount;
      int *leaves = predictLeaves + row * nTree;
      Leaves(treeOriginForest, leaves, qRow);
    }
  }
  }
}

/**
   @brief Writes the quantile values.

   @param treeOriginForest[] locates tree origins.

   @param leaves[] is a vector of leaves predicted for each out-of-bag row.

   @param qRow[] outputs quantile values.

   @return Void, with output vector parameter.
 */
void Quant::Leaves(const int treeOriginForest[], const int leaves[], double qRow[]) {
  int *sampRanks = new int[qYLenForest];
  for (int i = 0; i < qYLenForest; i++)
    sampRanks[i] = 0;

  // Scores each rank seen at every predicted leaf.
  //
  int totRanks = 0;
  int tn;
  for (tn = 0; tn < nTree; tn++) {
    int predLeaf = leaves[tn];
    if (predLeaf < 0) // in-bag:  no OOB prediction at row for this tree.
      continue;
    int leafPos = qLeafPosForest[treeOriginForest[tn] + predLeaf];
    int leafExtent = qLeafExtentForest[treeOriginForest[tn] + predLeaf];
    int leafOff = qRankOriginForest[tn] + leafPos;
    for (int i = 0; i < leafExtent; i++) {
      int leafRank = qRankForest[leafOff+i];
      int rankCount = qRankCountForest[leafOff + i];
      sampRanks[leafRank] += rankCount;
      totRanks += rankCount;
    }
  }

  double *countThreshold = new double[qCount];
  for (int i = 0; i < qCount; i++) {
    countThreshold[i] = totRanks * qVec[i];  // Rounding properties?
  }

  int qIdx = 0;
  int ranksSeen = 0;
  for (int i = 0; i < qYLenForest && qIdx < qCount; i++) {
     ranksSeen += sampRanks[i];
     while (qIdx < qCount && ranksSeen >= countThreshold[qIdx]) {
       qRow[qIdx++] = qYRankedForest[i];
     }
  }

  delete [] sampRanks;
  delete [] countThreshold;
}
