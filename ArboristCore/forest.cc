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
#include "predict.h"

//#include <iostream>
using namespace std;


/**
   @brief Crescent constructor for training.
*/
Forest::Forest(std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<unsigned int> &_facVec) : nTree(_origin.size()), forestNode(_forestNode), treeOrigin(_origin), facOrigin(_facOrigin), facVec(_facVec), predict(0) {
  facSplit = new BVJagged(facVec, _facOrigin);
}


/**
   @brief Constructor for prediction.
*/
Forest::Forest(std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<unsigned int> &_facVec, Predict *_predict) : nTree(_origin.size()), forestNode(_forestNode), treeOrigin(_origin), facOrigin(_facOrigin), facVec(_facVec), predict(_predict) {
  facSplit = new BVJagged(facVec, _facOrigin);
}


/**
 */ 
Forest::~Forest() {
  delete facSplit;
}


/**
   @brief Dispatches prediction method based on available predictor types.

   @param bag is the packed in-bag representation, if validating.

   @return void.
 */
void Forest::PredictAcross(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const {
  if (PredBlock::NPredFac() == 0)
    PredictAcrossNum(rowStart, rowEnd, bag);
  else if (PredBlock::NPredNum() == 0)
    PredictAcrossFac(rowStart, rowEnd, bag);
  else
    PredictAcrossMixed(rowStart, rowEnd, bag);
}


/**
   @brief Multi-row prediction for regression tree, with predictors of only numeric.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossNum(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const {
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = int(rowStart); row < int(rowEnd); row++) {
    PredictRowNum(row, PBPredict::RowNum(row), row - rowStart, bag);
  }
  }
}


/**
   @brief Multi-row prediction for regression tree, with predictors of both numeric and factor type.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossFac(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const {
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = int(rowStart); row < int(rowEnd); row++) {
      PredictRowFac(row, PBPredict::RowFac(row), row - rowStart, bag);
  }
  }

}


/**
   @brief Multi-row prediction with predictors of both numeric and factor type.

   @param rowStart is the first row in the block.

   @param rowEnd is the first row beyond the block.

   @param bag indicates whether prediction is restricted to out-of-bag data.

   @return Void with output vector parameter.
 */
void Forest::PredictAcrossMixed(unsigned int rowStart, unsigned int rowEnd, const class BitMatrix *bag) const {
  int row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = int(rowStart); row < int(rowEnd); row++) {
    PredictRowMixed(row, PBPredict::RowNum(row), PBPredict::RowFac(row), row - rowStart, bag);
  }
  }

}


/**
   @brief Prediction with predictors of only numeric type.

   @param row is the row of data over which a prediction is made.

   @param rowT is a numeric data array section corresponding to the row.

   @param bag indexes out-of-bag rows, and may be null.

   @return Void with output vector parameter.
 */

void Forest::PredictRowNum(unsigned int row, const double rowT[], unsigned int blockRow, const class BitMatrix *bag) const {
  for (int tc = 0; tc < nTree; tc++) {
    if (bag->TestBit(row, tc)) {
      predict->BagIdx(blockRow, tc);
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
    predict->LeafIdx(blockRow, tc, pred);
  }
}


/**
   @brief Prediction with factor-valued predictors only.

   @param row is the row of data over which a prediction is made.

   @param rowT is a factor data array section corresponding to the row.

   @param bag indexes out-of-bag rows, and may be null.

   @return Void with output vector parameter.
 */
void Forest::PredictRowFac(unsigned int row, const int rowT[], unsigned int blockRow, const class BitMatrix *bag) const {
  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (bag->TestBit(row, tc)) {
      predict->BagIdx(blockRow, tc);
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
      idx += facSplit->TestBit(tc, bitOff) ? bump : bump + 1;
      forestNode[treeBase + idx].Ref(pred, bump, num);
    }
    predict->LeafIdx(blockRow, tc, pred);
  }
}


/**
   @brief Prediction with predictors of both numeric and factor type.

   @param row is the row of data over which a prediction is made.

   @param rowNT is a numeric data array section corresponding to the row.

   @param rowFT is a factor data array section corresponding to the row.

   @param bag indexes out-of-bag rows, and may be null.

   @return Void with output vector parameter.
 */
void Forest::PredictRowMixed(unsigned int row, const double rowNT[], const int rowFT[], unsigned int blockRow, const class BitMatrix *bag) const {
  int tc;
  for (tc = 0; tc < nTree; tc++) {
    if (bag->TestBit(row, tc)) {
      predict->BagIdx(blockRow, tc);
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
      idx += isFactor ? (facSplit->TestBit(tc, (unsigned int) num + rowFT[blockIdx]) ? bump : bump + 1) : (rowNT[blockIdx] <= num ? bump : bump + 1);
      forestNode[treeBase + idx].Ref(pred, bump, num);
    }
    predict->LeafIdx(blockRow, tc, pred);
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
void Forest::BitProduce(const BV *splitBits, unsigned int bitEnd) {
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
   @brief Registers current vector sizes of crescent forest as origin values.

   @param tIdx is current tree index.
   
   @return void.
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


/**
   @brief Unpacks node fields into vector of per-tree vectors.

   @return void, with output reference vectors.
 */
void ForestNode::Export(const std::vector<unsigned int> &_nodeOrigin, const std::vector<ForestNode> &_forestNode, std::vector<std::vector<unsigned int> > &_pred, std::vector<std::vector<unsigned int> > &_bump, std::vector<std::vector<double> > &_split) {
  for (unsigned int tIdx = 0; tIdx < _nodeOrigin.size(); tIdx++) {
    unsigned int treeHeight = TreeHeight(_nodeOrigin, _forestNode.size(), tIdx);
    _pred[tIdx] = std::vector<unsigned int>(treeHeight);
    _bump[tIdx] = std::vector<unsigned int>(treeHeight);
    _split[tIdx] = std::vector<double>(treeHeight);
    TreeExport(_forestNode, _pred[tIdx], _bump[tIdx], _split[tIdx], _nodeOrigin[tIdx], TreeHeight(_nodeOrigin, _forestNode.size(), tIdx));
  }
}


/**
   @brief Exports node field values for a single tree.

   @return void, with output reference vectors.
 */
void ForestNode::TreeExport(const std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_pred, std::vector<unsigned int> &_bump, std::vector<double> &_split, unsigned int treeOff, unsigned int treeHeight) {
  for (unsigned int i = 0; i < treeHeight; i++) {
    _forestNode[treeOff + i].Ref(_pred[i], _bump[i], _split[i]);
  }
}
