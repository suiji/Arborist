// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predict.cc

   @brief Methods for validation and prediction.

   @author Mark Seligman
 */

#include "framemap.h"
#include "forest.h"
#include "leaf.h"
#include "predict.h"
#include "bv.h"
#include "quant.h"

#include <cfloat>
#include <algorithm>


Predict::Predict(const FramePredict *_framePredict,
                 const Forest *_forest,
                 bool validate) :
  useBag(validate),
  framePredict(_framePredict),
  forest(_forest),
  nTree(forest->NTree()),
  nRow(framePredict->NRow()) {
  predictLeaves = make_unique<unsigned int[]>(rowBlock * nTree);
}


void Predict::reg(LeafReg* leaf, const Forest* forest, const BitMatrix* bag, const FramePredict* framePredict, bool validate, Quant* quant) {
  auto predict = make_unique<Predict>(framePredict, forest, validate);
  predict->PredictAcross(leaf, bag, quant);
}


void Predict::ctg(LeafCtg *leaf, const Forest* forest, const BitMatrix* bag, const FramePredict* framePredict, bool validate) {
  auto predict = make_unique<Predict>(framePredict, forest, validate);
  predict->PredictAcross(leaf, bag, nullptr);
}


/**
 */
void Predict::PredictAcross(Leaf *leaf, const BitMatrix *bag, Quant *quant) {
  noLeaf = leaf->NoLeaf();
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += rowBlock) {
    unsigned int rowEnd = min(rowStart + rowBlock, nRow);
    framePredict->BlockTranspose(rowStart, rowEnd);
    PredictBlock(rowStart, rowEnd, bag);
    leaf->ScoreBlock(this, rowStart, rowEnd);
    if (quant != nullptr) {
      quant->PredictAcross(this, rowStart, rowEnd);
    }
  }
}


/**
   @brief Dispatches prediction method based on available predictor types.

   @param bag is the packed in-bag representation, if validating.

   @return void.
 */
void Predict::PredictBlock(unsigned int rowStart,
                           unsigned int rowEnd,
                           const BitMatrix *bag) {
  if (framePredict->NPredFac() == 0)
    PredictBlockNum(rowStart, rowEnd, bag);
  else if (framePredict->NPredNum() == 0)
    PredictBlockFac(rowStart, rowEnd, bag);
  else
    PredictBlockMixed(rowStart, rowEnd, bag);
}


/**
   @brief Multi-row prediction for regression tree, with predictors of only numeric.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Predict::PredictBlockNum(unsigned int rowStart,
                              unsigned int rowEnd,
                              const BitMatrix *bag) {
  OMPBound row;
  OMPBound rowSup = (OMPBound) rowEnd;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = (OMPBound) rowStart; row < rowSup; row++) {
      RowNum(row, row - rowStart, forest->Node(), forest->Origin(), bag);
    }
  }
}


/**
   @brief Multi-row prediction for regression tree, with predictors of both numeric and factor type.

   @param bag enumerates the in-bag rows, if validating.

   @return Void with output vector parameter.
 */
void Predict::PredictBlockFac(unsigned int rowStart,
                              unsigned int rowEnd,
                              const BitMatrix *bag) {
  OMPBound row;
  OMPBound rowSup = (OMPBound) rowEnd;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = (OMPBound) rowStart; row < rowSup; row++) {
      RowFac(row, row - rowStart, forest->Node(), forest->Origin(), forest->FacSplit(), bag);
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
void Predict::PredictBlockMixed(unsigned int rowStart,
                                unsigned int rowEnd,
                                const BitMatrix *bag) {
  OMPBound row;
  OMPBound rowSup = (OMPBound) rowEnd;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = (OMPBound) rowStart; row < rowSup; row++) {
      RowMixed(row, row - rowStart, forest->Node(), forest->Origin(), forest->FacSplit(), bag);
    }
  }
}



void Predict::RowNum(unsigned int row,
                     unsigned int blockRow,
                     const ForestNode *forestNode,
                     const unsigned int *origin,
                     const class BitMatrix *bag) {
  auto rowT = framePredict->RowNum(blockRow);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    auto leafIdx = noLeaf;
    if (!(useBag && bag->testBit(tIdx, row))) {
      auto idx = origin[tIdx];
      do {
        idx += forestNode[idx].advance(rowT, leafIdx);
      } while (leafIdx == noLeaf);
    }
    predictLeaf(blockRow, tIdx, leafIdx);
  }
}


/**
   @brief Prediction with factor-valued predictors only.

   @param row is the row of data over which a prediction is made.

   @param rowT is a factor data array section corresponding to the row.

   @param bag indexes out-of-bag rows, and may be null.

   @return Void with output vector parameter.
 */
void Predict::RowFac(unsigned int row,
                     unsigned int blockRow,
                     const ForestNode *forestNode,
                     const unsigned int *origin,
                     const BVJagged *facSplit,
                     const BitMatrix *bag) {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    auto leafIdx = noLeaf;
    if (!(useBag && bag->testBit(tIdx, row))) {
      auto rowT = framePredict->RowFac(blockRow);
      auto idx = origin[tIdx];
      do {
        idx += forestNode[idx].advance(facSplit, rowT, tIdx, leafIdx);
      } while (leafIdx == noLeaf);
    }
    predictLeaf(blockRow, tIdx, leafIdx);
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
void Predict::RowMixed(unsigned int row,
                       unsigned int blockRow,
                       const ForestNode *forestNode,
                       const unsigned int *origin,
                       const BVJagged *facSplit,
                       const BitMatrix *bag) {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    auto leafIdx = noLeaf;
    if (!(useBag && bag->testBit(tIdx, row))) {
      auto rowNT = framePredict->RowNum(blockRow);
      auto rowFT = framePredict->RowFac(blockRow);
      auto idx = origin[tIdx];
      do {
        idx += forestNode[idx].advance(framePredict, facSplit, rowFT, rowNT, tIdx, leafIdx);
      } while (leafIdx == noLeaf);
    }
    predictLeaf(blockRow, tIdx, leafIdx);
  }
}
