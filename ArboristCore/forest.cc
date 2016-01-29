// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.cc

   @brief Methods for building and walking the decision tree.

   @author Mark Seligman

   These methods are mostly mechanical.  Several methods are tasked
   with populating or depopulating tree-related data structures.  The
   tree-walking methods are clones of one another, with slight variations
   based on response or predictor type.
 */


#include "bv.h"
#include "forest.h"
#include "predblock.h"

//#include <iostream>
using namespace std;


/**
   @brief Reload constructor uses front end's storage.
*/
Forest::Forest(int _nTree, int _height, int _pred[], double _split[], int _bump[], int _origins[], int _facOrigin[], unsigned int _facSplit[]) : nTree(_nTree), treeOrigin(_origins), facOrigin(_facOrigin), facSplit(_facSplit), fePred(_pred), feNum(_split), feBump(_bump), height(_height) {
  forestNode = new ForestNode[height];
  for (int i = 0; i < height; i++) { // Caches copy as packed structures.
    forestNode[i].Set(fePred[i], feBump[i], feNum[i]);
  }
}

 
/**
   @brief Dispatches prediction method based on available predictor types.

   @param bag is the packed in-bag representation, if validating.

   @return void.
 */
void Forest::PredictAcross(int *predictLeaves, const unsigned int *bag) {
  if (PredBlock::NPredFac() == 0)
    PredictAcrossNum(predictLeaves, PredBlock::NRow(), bag);
  else if (PredBlock::NPredNum() == 0)
    PredictAcrossFac(predictLeaves, PredBlock::NRow(), bag);
  else
    PredictAcrossMixed(predictLeaves, PredBlock::NRow(), bag);
}


Forest::~Forest() {
  delete [] forestNode;
}

/**
   @brief Multi-row prediction for regression tree, with predictors of only numeric.

   @param leaves outputs the predicted leaf offsets.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossNum(int *leaves, unsigned int nRow, const unsigned int bag[]) {
  unsigned int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    PredictRowNum(row, PBPredict::RowNum(row), &leaves[nTree * row], bag);
  }
  }
}


/**
   @brief Multi-row prediction for regression tree, with predictors of both numeric and factor type.

   @param leaves outputs the predicted leaf offsets.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossFac(int *leaves, unsigned int nRow, const unsigned int bag[]) {
  unsigned int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    PredictRowFac(row, PBPredict::RowFac(row), &leaves[row * nTree], bag);
  }
  }

}


/**
   @brief Multi-row prediction for regression tree, with predictors of both numeric and factor type.

   @param prediction contains the mean score across trees.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossMixed(int *leaves, unsigned int nRow, const unsigned int bag[]) {
  unsigned int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < nRow; row++) {
    PredictRowMixed(row, PBPredict::RowNum(row), PBPredict::RowFac(row), &leaves[row * nTree], bag);
  }
  }

}


/**
   @brief Prediction for regression tree, with predictors of only numeric type.

   @param row is the row of data over which a prediction is made.

   @param rowT is a numeric data array section corresponding to the row.

   @param leaves[] are the tree terminals predicted for each row.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */

void Forest::PredictRowNum(unsigned int row, const double rowT[], int leaves[], const unsigned int bag[]) {
  for (int tc = 0; tc < nTree; tc++) {
    if (InBag(bag, tc, row)) {
      leaves[tc] = -1;
      continue;
    }

    ForestNode *treeBase;
    unsigned int *bitBase;
    TreeBases(tc, treeBase, bitBase);
    int idx = 0;
    unsigned int bump;
    int pred; // N.B.:  Use BlockIdx() if numericals not numbered from 0.
    double num;
    treeBase[0].Ref(pred, bump, num);
    while (bump != 0) {
      idx += (rowT[pred] <= num ? bump : bump + 1);
      treeBase[idx].Ref(pred, bump, num);
    }
    leaves[tc] = idx;
  }
}


/**
   @brief Prediction for regression tree, with factor-valued predictors only.

   @param row is the row of data over which a prediction is made.

   @param rowT is a factor data array section corresponding to the row.

   @param leaves[] are the tree terminals predicted for each row.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void Forest::PredictRowFac(unsigned int row, const int rowT[], int leaves[],  const unsigned int bag[]) {
  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (InBag(bag, tc, row)) {
      leaves[tc] = -1;
      continue;
    }

    ForestNode *treeBase;
    unsigned int *bitBase;
    TreeBases(tc, treeBase, bitBase);

    int idx = 0;
    unsigned int bump;
    int pred; // N.B.: Use BlockIdx() if not factor-only (zero based).
    double num;
    treeBase[0].Ref(pred, bump, num);
    while (bump != 0) {
      unsigned int splitOff = (unsigned int) num;
      idx += (BV::IsSet(bitBase, splitOff + rowT[pred]) ? bump : bump + 1);
      treeBase[idx].Ref(pred, bump, num);
    }
    leaves[tc] = idx;
  }
}


/**
   @brief Prediction for regression tree, with predictors of both numeric and factor type.

   @param row is the row of data over which a prediction is made.

   @param rowNT is a numeric data array section corresponding to the row.

   @param rowFT is a factor data array section corresponding to the row.

   @param leaves[] are the tree terminals predicted for each row.

   @param useBag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void Forest::PredictRowMixed(unsigned int row, const double rowNT[], const int rowFT[], int leaves[], const unsigned int bag[]) {
  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (InBag(bag, tc, row)) {
      leaves[tc] = -1;
      continue;
    }

    ForestNode *treeBase;
    unsigned int *bitBase;
    TreeBases(tc, treeBase, bitBase);

    int idx = 0;
    unsigned int bump;
    int pred;
    double num;
    treeBase[0].Ref(pred, bump, num);
    while (bump != 0) {
      bool isFactor;
      unsigned int blockIdx = PredBlock::BlockIdx(pred, isFactor);
      unsigned int splitOff = (unsigned int) num;
      idx += isFactor ? (BV::IsSet(bitBase, splitOff + rowFT[blockIdx]) ? bump : bump + 1) : (rowNT[blockIdx] <= num ?  bump : bump + 1);
      treeBase[idx].Ref(pred, bump, num);
    }
    leaves[tc] = idx;
  }
}


/**
   @brief Determines whether a given row index is in-bag in a given tree.

   @param treeNum is the index of a given tree.

   @param row is the row index to be tested.

   @return True iff the row is in-bag.
 */
bool Forest::InBag(const unsigned int bag[], int treeNum, unsigned int row) {
  return bag != 0 && BV::IsSet(bag, row * nTree + treeNum);
}


/**
   @brief Sets the in-bag bit for a given <tree, row> pair.

   @param bag is the forest-wide bit matrix.

   @param _nTree is the number of trees being grown.

   @param treeNum is the current tree index.

   @param row is the row index.

   @return void.
*/
void Forest::BagSet(unsigned int bag[], int _nTree, unsigned int treeNum, unsigned int row) {
  BV::SetBit(bag, row * _nTree + treeNum);
}

