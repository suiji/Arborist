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
#include "pretree.h"

//#include <iostream>
using namespace std;

bool Quant::live = false;
int Quant::nTree = -1;
unsigned int Quant::nRow = 0;
int Quant::qCount = 0;
unsigned int Quant::binSize = 0;
unsigned int Quant::qBin = 0;
unsigned int Quant::logSmudge = 0;
unsigned int Quant::smudge = 0;
double *Quant::qVec = 0;
double *Quant::qPred = 0;


// Nonzero iff quantiles stipulated for training.
int *Quant::treeBagCount = 0;
unsigned int **Quant::treeQRank = 0;
int **Quant::treeQSCount = 0;

unsigned int Quant::totBagCount = 0;

// Nonzero iff quantiles tabulated.
double *Quant::qYRankedForest = 0;
int *Quant::qRankForest = 0;
int *Quant::qSCountForest = 0;


/**
   @brief Entry for training path.

   @brief _nRow is the row count.

   @param _nTree is the number of trees requested for training.

   @param _train indicates whether quantile training has been requested.

   @return void.
 */
void Quant::FactoryTrain(unsigned int _nRow, int _nTree, bool _train) {
  live = _train;
  if (!live)
    return;

  nRow = _nRow;
  nTree = _nTree;
  totBagCount = 0;

  treeBagCount = new int[nTree];
  treeQRank = new unsigned int*[nTree];
  treeQSCount = new int*[nTree];
}


/**
   @brief Writes quantile data into storage provided by front end and finalizes.

   @param rQYRanked is the sorted respose.

   @param rQRank is the forest-wide collection of sampled ranks, by leaf.

   @param rQSRank is the forest-wide collection of sample counts, by leaf.

   @return void, with output vector parameters.

 */
void Quant::Write(double rQYRanked[], int rQRank[], int rQSCount[]) {
  // ASSERTION
  //  if (!live)
  //cout << "Attempting to write and deallocate unallocated quantile" << endl;

  for (unsigned int i = 0; i < nRow; i++)
    rQYRanked[i] = qYRankedForest[i];

  for (unsigned int i = 0; i < totBagCount; i++) {
    rQRank[i] = qRankForest[i];
    rQSCount[i] = qSCountForest[i];
  }

  delete [] qYRankedForest;
  delete [] qRankForest;
  delete [] qSCountForest;

  nRow = nTree = totBagCount = -1;
  qYRankedForest = 0;
  qRankForest = 0;
  qSCountForest = 0;

  // Tree-based quantile data.
  treeBagCount = 0;
  treeQRank = 0;
  treeQSCount = 0;;

  live = false;
}


/**
  @brief Loads quantile data stored by the front end.

  @param _nTree is the number of trees requested for training.

  @param qYRanked is the sorted response.

  @param qRank is the forest-wide set of sampled ranks, grouped by leaf.

  @param qSCount is the forest-wide set of sample counts, by leaf.

  @return void.
*/
void Quant::FactoryPredict(int _nTree, double qYRanked[], int qRank[], int qSCount[]) {
  live = true;
  nTree = _nTree;
  qYRankedForest = qYRanked;
  qRankForest = qRank;
  qSCountForest = qSCount;
}


/**
   @brief Sets the global parameters for quantile prediction using storage provided by the front end.

   @param _qVec is a front-end vector of quantiles to evaluate.

   @param _qCount is the number of elements in _qVec.

   @param _qBin is the row-count threshold for binning.

   @param _qPred outputs the prediction.

   @param _nRow is an optional row count.  If unset, the current value is maintained.

   @return void.
 */
void Quant::EntryPredict(double _qVec[], int _qCount, unsigned int _qBin, double _qPred[], unsigned int _nRow) {
  if (_nRow > 0)
    nRow = _nRow;
  qCount = _qCount;
  qBin = _qBin;
  qVec = _qVec;
  qPred = _qPred;
}


/**
   @basic Finalizer for prediction-only path.

   @return void.
 */
void Quant::DeFactoryPredict() {
  live = false;
  nTree = -1;
  qBin = qCount = 0;
  qVec = qPred = 0;
  qYRankedForest = 0;
  qRankForest = 0;
}


/**
   @brief Consumes per-tree quantile information into forest-wide vectors.

   @return void.
 */
void Quant::ConsumeTrees() {
  if (!live)
    return;

  qYRankedForest = new double[nRow];
  ResponseReg::GetYRanked(qYRankedForest);

  qRankForest = new int[totBagCount];
  qSCountForest = new int[totBagCount];
  int rankStart = 0;
  for (int tn = 0; tn < nTree; tn++) {
    int bagCount = treeBagCount[tn];
    int *qRank = qRankForest + rankStart;
    int *qSCount = qSCountForest + rankStart;
    for (int i = 0; i < bagCount; i++) {
      qRank[i] = treeQRank[tn][i];
      qSCount[i] = treeQSCount[tn][i];
    }
    delete [] treeQRank[tn];
    delete [] treeQSCount[tn];

    rankStart += bagCount;
  }

  delete [] treeBagCount;
  delete [] treeQRank;
  delete [] treeQSCount;
}


/**
  @brief Transfers quantile data structures from pretree to decision tree.

  @param preTree references the current PreTree.

  @param nonTerm is the decision tree's bump field, which is zero iff leaf.

  @param leafExtent is the decision tree's pred field, which enumerates leaf widths.

  @param tn is the current tree number.

  @return void.
*/
void Quant::TreeRanks(const PreTree *preTree, const int nonTerm[], const int leafExtent[], int tn) {
  if (!live)
    return;

  int bagCount = preTree->BagCount();
  treeBagCount[tn] = bagCount;
  totBagCount += bagCount;

  treeQRank[tn] = new unsigned int[bagCount];
  treeQSCount[tn] = new int[bagCount];
  Quantiles(preTree, nonTerm, leafExtent, treeQRank[tn], treeQSCount[tn]);
}


/**
   @brief Derives and copies quantile leaf information.

   @param pt references the PreTree object.

   @param nonTerm is zero iff forest index is at leaf.

   @param leafExtent gives leaf width at forest index.

   @param qRank outputs quantile leaf ranks; vector length bagCount.

   @param qrankCount outputs rank multiplicities; vector length bagCount.

   @return void, with output parameter vectors.
 */
void Quant::Quantiles(const PreTree *pt, const int nonTerm[], const int leafExtent[], unsigned int qRank[], int qSCount[]) {
  int treeHeight = pt->TreeHeight();
  int *leafPos = LeafPos(treeHeight, nonTerm, leafExtent);
  int *seen = new int[treeHeight];
  for (int i = 0; i < treeHeight; i++) {
    seen[i] = 0;
  }

  for (int sIdx = 0; sIdx < pt->BagCount(); sIdx++) {
    int sCount;
    unsigned int rank;
    int leafIdx = pt->QuantileFields(sIdx, sCount, rank);
    int rkOff = leafPos[leafIdx] + seen[leafIdx]++;
    qSCount[rkOff] = sCount;
    qRank[rkOff] = rank;
  }
  
  delete [] seen;
  delete [] leafPos;
}


/**
   @brief Defines starting positions for ranks associated with a given leaf.

   @param treeHeight is the height of the current tree.

   @param nonTerm is zero iff leaf reference.

   @param leafExtent enumerates leaf widths.

   @return vector of leaf sample offsets, by tree index.
 */
int *Quant::LeafPos(int treeHeight, const int nonTerm[], const int leafExtent[]) {
  int totCt = 0;
  int *leafPos = new int[treeHeight];
  for (int i = 0; i < treeHeight; i++) {
    if (nonTerm[i] == 0) {
      leafPos[i] = totCt;
      totCt += leafExtent[i];
    }
    else
      leafPos[i] = -1;
  }
  // ASSERTION:  totCt == pt->BagCount()
  // By this point leafPos[i] >= 0 iff this 'i' references is a leaf.

  return leafPos;
}  


/**
   @brief Fills in the quantile leaves for each row.

   @param predictLeavs outputs the matrix of quantile leaves.

   @return void, with output parameter matrix.
 */
void Quant::PredictRows(const int treeOriginForest[], const int nonTermForest[], const int extentForest[], int forestLength, int predictLeaves[]) {
  if (!live)
    return;

  int *posForest = new int[forestLength];
  AbsOffset(nonTermForest, extentForest, forestLength, posForest);
  
  logSmudge = 0;
  while ((nRow >> logSmudge) > qBin)
    logSmudge++;
  binSize = (nRow + (1 << logSmudge) - 1) >> logSmudge;
  if (logSmudge > 0)
    SmudgeLeaves(treeOriginForest, nonTermForest, extentForest, posForest, forestLength);
  smudge = (1 << logSmudge);

  unsigned int row;
#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = 0; row < nRow; row++) {
      double *qRow = qPred + row * qCount;
      int *leaves = predictLeaves + row * nTree;
      Leaves(treeOriginForest, extentForest, posForest, leaves, qRow);
    }
  }

  delete [] posForest;
}


/**
   @brief For each leaf in tree, stamps absolute offset for start of rank set.

   @param nonTerm is the tree's bump field, which is zero iff node is terminal.
   
   @param leafExtent is the tree's pred field which, for leaves, enumerates
   samples subsumed.

   @param forestLen is the sum of tree heights.

   @param posForest outputs rank/rankSCount offsets for leaves.

   @return void, with output vector.
 */
void Quant::AbsOffset(const int nonTerm[], const int leafExtent[], int forestLength, int posForest[]) {
  int idxAccum = 0;
  for (int i = 0; i < forestLength; i++) {
    if (nonTerm[i] == 0) {
      posForest[i] = idxAccum;
      idxAccum += leafExtent[i];
    }
  }
}


/**
   @brief Overwrites rank counts for wide leaves with binned values.

   @return void.
 */
void Quant::SmudgeLeaves(const int treeOriginForest[], const int nonTermForest[], const int extentForest[], const int posForest[], int forestLength) {
  for (int i = 0; i < forestLength; i++) {
    if (nonTermForest[i] == 0) {
      int rankOff = posForest[i];
      unsigned int leafSize = extentForest[i];
      if (leafSize > binSize) {
	int *binTemp = new int[binSize];
	for (unsigned int j = 0; j < binSize; j++)
	  binTemp[j] = 0;
	for (unsigned int j = 0; j < leafSize; j++) {
	  unsigned int rank = qRankForest[rankOff + j];
	  binTemp[rank >> logSmudge] += qSCountForest[rankOff + j];
	}
	for (unsigned int j = 0; j < binSize; j++) {
	  qSCountForest[rankOff + j] = binTemp[j];
	}
	delete [] binTemp;
      }
    }
  }
}


/**
   @brief Writes the quantile values.

   @param treeOriginForest[] locates tree origins.

   @param extentForest gives the number of distinct samples referenced by leaf.

   @param posForest is the forest-wide vector of starting rank indices.

   @param leaves references the per-tree leaf prediction.

   @param qRow[] outputs quantile values.

   @return void, with output vector parameter.
 */
void Quant::Leaves(const int treeOriginForest[], const int extentForest[], const int posForest[], const int leaves[], double qRow[]) {
  int qTrainRanks = logSmudge > 0 ? binSize : nRow;
  int *sampRanks = new int[qTrainRanks];
  for (int i = 0; i < qTrainRanks; i++)
    sampRanks[i] = 0;

  // Scores each rank seen at every predicted leaf.
  //
  int totRanks = 0;
  for (int tn = 0; tn < nTree; tn++) {
    int leafIdx = leaves[tn];
    if (leafIdx >= 0) { // otherwise in-bag:  no prediction for tree at row.
      int tOrig = treeOriginForest[tn];
      int leafOff = tOrig + leafIdx; // Absolute forest offset of leaf.
      int rankOff = posForest[leafOff];
      if (logSmudge == 0) {
        totRanks += RanksExact(extentForest[leafOff], rankOff, sampRanks);
      }
      else {
        totRanks += RanksSmudge(extentForest[leafOff], rankOff, sampRanks);
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
  for (int i = 0; i < qTrainRanks && qIdx < qCount; i++) {
    rankCount += sampRanks[i];
    while (qIdx < qCount && rankCount >= countThreshold[qIdx]) {
      qRow[qIdx++] = qYRankedForest[rankIdx];
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

   @param rankOff is the forest starting index of leaf's rank values.

   @return count of ranks introduced by leaf.
 */
int Quant::RanksExact(int leafExtent, int rankOff, int sampRanks[]) {
  int rankTot = 0;
  for (int i = 0; i < leafExtent; i++) {
    unsigned int leafRank = qRankForest[rankOff+i];
    int rankCount = qSCountForest[rankOff + i];
    sampRanks[leafRank] += rankCount;
    rankTot += rankCount;
  }

  return rankTot;
}


/**
   @brief Accumulates binned ranks assocated with a predicted leaf.

   @param leafExtent enumerates sample indices associated with a leaf.

   @param rankOff is the forest starting index of leaf's rank values.

   @param sampRanks[] outputs the binned rank counts.

   @return count of ranks introduced by leaf.
 */
int Quant::RanksSmudge(unsigned int leafExtent, int rankOff, int sampRanks[]) {
  int rankTot = 0;
  if (leafExtent <= binSize) {
    for (unsigned int i = 0; i < leafExtent; i++) {
      int rankIdx = (qRankForest[rankOff+i] >> logSmudge);
      int rankCount = qSCountForest[rankOff + i];
      sampRanks[rankIdx] += rankCount;
      rankTot += rankCount;
    }
  }
  else {
    for (unsigned int rankIdx = 0; rankIdx < binSize; rankIdx++) {
      int rankCount = qSCountForest[rankOff + rankIdx];
      sampRanks[rankIdx] += rankCount;
      rankTot += rankCount;
    }
  }

  return rankTot;
}

