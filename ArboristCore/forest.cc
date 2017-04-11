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
//using namespace std;


const double *ForestNode::splitQuant = 0;


/**
   @brief Crescent constructor for training.
*/
ForestTrain::ForestTrain(std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<unsigned int> &_facVec) : forestNode(_forestNode), treeOrigin(_origin), facOrigin(_facOrigin), facVec(_facVec) {
}

ForestTrain::~ForestTrain() {
}


/**
   @brief Constructor for prediction.
*/
Forest::Forest(const ForestNode _forestNode[], const unsigned int _origin[], unsigned int _nTree, unsigned int _facVec[], size_t _facLen, const unsigned int _facOrigin[], unsigned int _nFac, Predict *_predict) : forestNode(_forestNode), treeOrigin(_origin), nTree(_nTree), facSplit(new BVJagged(_facVec, _facLen, _facOrigin, _nFac)), predict(_predict), predMap(predict->PredMap())  {
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
  if (predMap->NPredFac() == 0)
    PredictAcrossNum(rowStart, rowEnd, bag);
  else if (predMap->NPredNum() == 0)
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
      PredictRowNum(row, predict->RowNum(row - rowStart), row - rowStart, bag);
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
      PredictRowFac(row, predict->RowFac(row - rowStart), row - rowStart, bag);
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
      PredictRowMixed(row, predict->RowNum(row - rowStart), predict->RowFac(row - rowStart), row - rowStart, bag);
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
  for (unsigned int tIdx = 0; tIdx < NTree(); tIdx++) {
    if (bag->TestBit(row, tIdx)) {
      predict->BagIdx(blockRow, tIdx);
      continue;
    }

    unsigned int idx = treeOrigin[tIdx];
    unsigned int bump;
    unsigned int pred; // N.B.:  Use BlockIdx() if numericals not numbered from 0.
    double num;
    Ref(idx, pred, bump, num);
    while (bump != 0) {
      idx += (rowT[pred] <= num ? bump : bump + 1);
      Ref(idx, pred, bump, num);
    }
    predict->LeafIdx(blockRow, tIdx, pred);
  }
}


/**
   @brief Prediction with factor-valued predictors only.

   @param row is the row of data over which a prediction is made.

   @param rowT is a factor data array section corresponding to the row.

   @param bag indexes out-of-bag rows, and may be null.

   @return Void with output vector parameter.
 */
void Forest::PredictRowFac(unsigned int row, const unsigned int rowT[], unsigned int blockRow, const class BitMatrix *bag) const {
  for (unsigned int tIdx = 0; tIdx < NTree(); tIdx++) {
    if (bag->TestBit(row, tIdx)) {
      predict->BagIdx(blockRow, tIdx);
      continue;
    }

    unsigned int idx = treeOrigin[tIdx];
    unsigned int bump;
    unsigned int pred; // N.B.: Use BlockIdx() if not factor-only (zero based).
    double num;
    Ref(idx, pred, bump, num);
    while (bump != 0) {
      unsigned int bitOff = (unsigned int) num + rowT[pred];
      idx += facSplit->TestBit(tIdx, bitOff) ? bump : bump + 1;
      Ref(idx, pred, bump, num);
    }
    predict->LeafIdx(blockRow, tIdx, pred);
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
void Forest::PredictRowMixed(unsigned int row, const double rowNT[], const unsigned int rowFT[], unsigned int blockRow, const class BitMatrix *bag) const {
  for (unsigned int tIdx = 0; tIdx < NTree(); tIdx++) {
    if (bag->TestBit(row, tIdx)) {
      predict->BagIdx(blockRow, tIdx);
      continue;
    }

    unsigned int idx = treeOrigin[tIdx];
    unsigned int bump;
    unsigned int pred;
    double num;
    Ref(idx, pred, bump, num);
    while (bump != 0) {
      bool isFactor;
      unsigned int blockIdx = predMap->BlockIdx(pred, isFactor);
      idx += isFactor ? (facSplit->TestBit(tIdx, (unsigned int) num + rowFT[blockIdx]) ? bump : bump + 1) : (rowNT[blockIdx] <= num ? bump : bump + 1);
      Ref(idx, pred, bump, num);
    }
    predict->LeafIdx(blockRow, tIdx, pred);
  }
}


/**
 */
void ForestTrain::NodeInit(unsigned int treeHeight) {
  ForestNode fn;
  fn.Init();
  forestNode.insert(forestNode.end(), treeHeight, fn);
}


/**
   @brief Produces new splits for an entire tree.
 */
void ForestTrain::BitProduce(const BV *splitBits, unsigned int bitEnd) {
  splitBits->Consume(facVec, bitEnd);
}


/**
  @brief Reserves space in the relevant vectors for new trees.
 */
void ForestTrain::Reserve(unsigned int blockHeight, unsigned int blockFac, double slop) {
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
void ForestTrain::Origins(unsigned int tIdx) {
  treeOrigin[tIdx] = Height();
  facOrigin[tIdx] = SplitHeight();
}


/**
   @brief Post-pass to update numerical splitting values from ranks.

   @param rowRank holds the presorted predictor values.

   @return void
 */
void ForestTrain::SplitUpdate(const PMTrain *pmTrain, const RowRank *rowRank) const {
  for (unsigned int i = 0; i < forestNode.size(); i++) {
    forestNode[i].SplitUpdate(pmTrain, rowRank);
  }
}


/**
   @brief Assigns value at quantile rank to numerical split.

   @param rowRank holds the presorted predictor values.

   @return void.
 */
void ForestNode::SplitUpdate(const PMTrain *pmTrain, const RowRank *rowRank) {
  if (Nonterminal() && !pmTrain->IsFactor(pred)) {
    splitVal.num = rowRank->QuantRank(pred, splitVal.rankRange, splitQuant);
  }
}


/**
   @brief Unpacks node fields into vector of per-tree vectors.

   @return void, with output reference vectors.
 */
void ForestNode::Export(const unsigned int _nodeOrigin[], unsigned int _nTree, const ForestNode *_forestNode, unsigned int nodeEnd, std::vector<std::vector<unsigned int> > &_pred, std::vector<std::vector<unsigned int> > &_bump, std::vector<std::vector<double> > &_split) {
  for (unsigned int tIdx = 0; tIdx < _nTree; tIdx++) {
    unsigned int treeHeight = TreeHeight(_nodeOrigin, _nTree, nodeEnd, tIdx);
    _pred[tIdx] = std::vector<unsigned int>(treeHeight);
    _bump[tIdx] = std::vector<unsigned int>(treeHeight);
    _split[tIdx] = std::vector<double>(treeHeight);
    TreeExport(_forestNode, _pred[tIdx], _bump[tIdx], _split[tIdx], _nodeOrigin[tIdx], treeHeight);
  }
}


/**
   @brief Exports node field values for a single tree.

   @return void, with output reference vectors.
 */
void ForestNode::TreeExport(const ForestNode *_forestNode, std::vector<unsigned int> &_pred, std::vector<unsigned int> &_bump, std::vector<double> &_split, unsigned int treeOff, unsigned int treeHeight) {
  for (unsigned int i = 0; i < treeHeight; i++) {
    _forestNode[treeOff + i].Ref(_pred[i], _bump[i], _split[i]);
  }
}
