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
void Forest::PredictAcross(int *predictLeaves, unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const {
  if (PredBlock::NPredFac() == 0)
    PredictAcrossNum(predictLeaves, rowStart, rowEnd, bag);
  else if (PredBlock::NPredNum() == 0)
    PredictAcrossFac(predictLeaves, rowStart, rowEnd, bag);
  else
    PredictAcrossMixed(predictLeaves, rowStart, rowEnd, bag);
}


/**
   @brief Multi-row prediction for regression tree, with predictors of only numeric.

   @param leaves outputs the predicted leaf offsets.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossNum(int *leaves, unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const {
  unsigned int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = rowStart; row < rowEnd; row++) {
    PredictRowNum(row, PBPredict::RowNum(row), &leaves[nTree * (row - rowStart)], bag);
  }
  }
}


/**
   @brief Multi-row prediction for regression tree, with predictors of both numeric and factor type.

   @param leaves outputs the predicted leaf offsets.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossFac(int *leaves, unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const {
  unsigned int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = rowStart; row < rowEnd; row++) {
      PredictRowFac(row, PBPredict::RowFac(row), &leaves[nTree * (row - rowStart)], bag);
  }
  }

}


/**
   @brief Multi-row prediction for regression tree, with predictors of both numeric and factor type.

   @param prediction contains the mean score across trees.

   @param bag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossMixed(int *leaves, unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const {
  unsigned int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = rowStart; row < rowEnd; row++) {
    PredictRowMixed(row, PBPredict::RowNum(row), PBPredict::RowFac(row), &leaves[nTree * (row - rowStart)], bag);
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

void Forest::PredictRowNum(unsigned int row, const double rowT[], int leaves[], const class BitMatrix *bag) const {
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
      forestNode[treeBase + idx].Ref(pred, bump, num);
    }
    leaves[tc] = pred;
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
void Forest::PredictRowFac(unsigned int row, const int rowT[], int leaves[],  const class BitMatrix *bag) const {
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
    leaves[tc] = pred;
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
void Forest::PredictRowMixed(unsigned int row, const double rowNT[], const int rowFT[], int leaves[], const class BitMatrix *bag) const {
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
    leaves[tc] = pred;
  }
}


/**
 */
void Forest::NodeInit(unsigned int treeHeight) {
  ForestNode fn;
  fn.Init();
  forestNode.insert(forestNode.end(), treeHeight, fn);
}


/**
   @brief Produces new splits for an entire tree.
 */
void Forest::BitProduce(BV *splitBits, unsigned int bitEnd) {
  splitBits->Consume(facVec, bitEnd);
}


/**
  @brief Reserves space in the relevant vectors for new trees.
 */
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
   @brief Post-pass to update numerical splitting values from ranks.

   @param rowRank holds the presorted predictor values.

   @return void
 */
void Forest::SplitUpdate(const RowRank *rowRank) const {
  for (unsigned int i = 0; i < forestNode.size(); i++) {
    forestNode[i].SplitUpdate(rowRank);
  }
}


/**
   @brief Assigns value at mean rank to numerical split.

   @param rowRank holds the presorted predictor values.

   @return void.
 */
void ForestNode::SplitUpdate(const RowRank *rowRank) {
  if (Nonterminal() && !PredBlock::IsFactor(pred)) {
    num = rowRank->MeanRank(pred, num);    
  }
}
