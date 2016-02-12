// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.cc

   @brief Methods for building and walking the decision tree.

   @author Mark Seligman
 */


#include "bv.h"
#include "forest.h"
#include "predblock.h"
#include "rowrank.h"

//#include <iostream>
using namespace std;


/**
   @brief Crescent constructor for training.
*/
Forest::Forest(std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<unsigned int> &_facVec) : nTree(_origin.size()), forestNode(_forestNode), treeOrigin(_origin), facOrigin(_facOrigin), facVec(_facVec) {
  facSplit = new BVJagged(facVec, _facOrigin);
}


/**
 */ 
Forest::~Forest() {
  delete facSplit;
}


/**
   @brief Dispatches prediction method based on available predictor types.

   @param predictLeaves outputs the predicted leaf indices.

   @param bag is the packed in-bag representation, if validating.

   @return void.
 */
void Forest::PredictAcross(int *predictLeaves, const class BitMatrix *bag) {
  if (PredBlock::NPredFac() == 0)
    PredictAcrossNum(predictLeaves, PredBlock::NRow(), bag);
  else if (PredBlock::NPredNum() == 0)
    PredictAcrossFac(predictLeaves, PredBlock::NRow(), bag);
  else
    PredictAcrossMixed(predictLeaves, PredBlock::NRow(), bag);
}


/**
   @brief Multi-row prediction for regression tree, with predictors of only numeric.

   @param leaves outputs the predicted leaf offsets.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossNum(int *leaves, unsigned int nRow, const class BitMatrix *bag) {
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
void Forest::PredictAcrossFac(int *leaves, unsigned int nRow, const class BitMatrix *bag) {
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

   @param bag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossMixed(int *leaves, unsigned int nRow, const class BitMatrix *bag) {
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

void Forest::PredictRowNum(unsigned int row, const double rowT[], int leaves[], const class BitMatrix *bag) {
  for (int tc = 0; tc < nTree; tc++) {
    if (bag->IsSet(row, tc)) {
      leaves[tc] = -1;
      continue;
    }

    unsigned int treeBase = treeOrigin[tc];
    unsigned int idx = 0;
    unsigned int bump;
    unsigned int pred; // N.B.:  Use BlockIdx() if numericals not numbered from 0.
    double num;
    forestNode[treeBase].Ref(pred, bump, num);
    while (bump != 0) {
      idx += (rowT[pred] <= num ? bump : bump + 1);
      forestNode[treeBase +idx].Ref(pred, bump, num);
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
void Forest::PredictRowFac(unsigned int row, const int rowT[], int leaves[],  const class BitMatrix *bag) {
  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (bag->IsSet(row, tc)) {
      leaves[tc] = -1;
      continue;
    }

    unsigned int treeBase = treeOrigin[tc];
    unsigned int idx = 0;
    unsigned int bump;
    unsigned int pred; // N.B.: Use BlockIdx() if not factor-only (zero based).
    double num;
    forestNode[treeBase].Ref(pred, bump, num);
    while (bump != 0) {
      unsigned int bitOff = (unsigned int) num + rowT[pred];
      idx += facSplit->IsSet(tc, bitOff) ? bump : bump + 1;
      forestNode[treeBase + idx].Ref(pred, bump, num);
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
void Forest::PredictRowMixed(unsigned int row, const double rowNT[], const int rowFT[], int leaves[], const class BitMatrix *bag) {
  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (bag->IsSet(row, tc)) {
      leaves[tc] = -1;
      continue;
    }

    unsigned int treeBase = treeOrigin[tc];
    unsigned int idx = 0;
    unsigned int bump;
    unsigned int pred;
    double num;
    forestNode[treeBase].Ref(pred, bump, num);
    while (bump != 0) {
      bool isFactor;
      unsigned int blockIdx = PredBlock::BlockIdx(pred, isFactor);
      idx += isFactor ? (facSplit->IsSet(tc, (unsigned int) num + rowFT[blockIdx]) ? bump : bump + 1) : (rowNT[blockIdx] <= num ? bump : bump + 1);
      forestNode[treeBase + idx].Ref(pred, bump, num);
    }
    leaves[tc] = idx;
  }
}


/**
   @brief Produces a new forest node and initializes to values passed.
 */
void Forest::NodeProduce(unsigned int _predIdx, unsigned int _bump, double _split) {
  ForestNode fn;
  fn.Set(_predIdx, _bump, _split);
  forestNode.push_back(fn);
}


/**
   @brief Produces new splits for an entire tree.
 */
void Forest::BitProduce(BV *splitBits, unsigned int bitEnd) {
  splitBits->Consume(facVec, bitEnd);
}


/**
   @brief Updates numerical splitting values from ranks.

   @param rowRank

   @return void
 */
void Forest::ScoreUpdate(const RowRank *rowRank) {
  for (unsigned int i = 0; i < forestNode.size(); i++) {
    unsigned int predIdx = forestNode[i].Pred();
    if (forestNode[i].Nonterminal()  && !PredBlock::IsFactor(predIdx)) {
      forestNode[i].Num() = rowRank->MeanRank(predIdx, forestNode[i].Num());
    }
  }
}


void Forest::Reserve(unsigned int blockHeight, unsigned int blockFac, double slop) {
  forestNode.reserve(slop * blockHeight);
  if (blockFac > 0) {
    facVec.reserve(slop * blockFac);
  }
}


/**
   @param tIdx is current tree index.

   @brief Registers current vector sizes of crescent forest as origin values.
 */
void Forest::Origins(unsigned int tIdx) {
  treeOrigin[tIdx] = Height();
  facOrigin[tIdx] = SplitHeight();
}


/**
     @brief Classification score.  Uses forest-width jitter.
     
*/
void Forest::ScoreCtg(int tIdx, unsigned int off, unsigned int ctg, double weight) const {
  unsigned int idx = treeOrigin[tIdx] + off;
  forestNode[idx].Num() = ctg + weight / (PredBlock::NRow() * nTree);
}


/**
   @brief Defines extent-based starting positions ranks associated with leaf.

   @param treeHeight is the height of the current tree.

   @param tIdx is the tree index or, if negative, the entire forest.

   @return vector of leaf sample offsets, by tree index.
 */
int *Forest::ExtentPosition(int tIdx) const {
  unsigned int treeHeight = tIdx >= 0 ? TreeHeight(tIdx) : Height();
  int totCt = 0;
  int *extentPos = new int[treeHeight];
  for (unsigned int i = 0; i < treeHeight; i++) {
    if (Nonterminal(tIdx, i)) {
      extentPos[i] = -1;
    }
    else {
      extentPos[i] = totCt;
      totCt += Extent(tIdx, i);
    }
  }
  // ASSERTION:  totCt == bagCount
  // By this point extentPos[i] >= 0 iff this 'i' references is a leaf.

  return extentPos;
}
