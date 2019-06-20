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

#include "bag.h"
#include "block.h"
#include "forest.h"
#include "leaf.h"
#include "predict.h"
#include "bv.h"
#include "quant.h"
#include "ompthread.h"


Predict::Predict(const Bag* bag_,
                 const Forest* forest,
                 LeafFrame* leaf_,
                 Quant* quant_,
                 bool oob_) :
  bag(bag_),
  nTree(forest->getNTree()),
  treeOrigin(forest->cacheOrigin()),
  treeNode(forest->getNode()),
  facSplit(forest->getFacSplit()),
  leaf(leaf_),
  noLeaf(leaf->getNoLeaf()),
  quant(quant_), 
  oob(oob_),
  predictLeaves(make_unique<unsigned int[]>(rowBlock * nTree)) {
}


PredictFrame::PredictFrame(Predict* predict_,
                           const BlockDense<double>* blockNum_,
                           const BlockDense<unsigned int>* blockFac_) :
  predict(predict_),
  blockNum(blockNum_),
  blockFac(blockFac_),
  predictRow(getNPredFac() == 0 ? &PredictFrame::predictNum : (getNPredNum() == 0 ? &PredictFrame::predictFac : &PredictFrame::predictMixed)) {
}


void PredictFrame::predictAcross(size_t rowStart) const {
  predictBlock(rowStart);
  predict->scoreBlock(rowStart, getExtent());
  predict->quantBlock(rowStart, getExtent());
}


void Predict::scoreBlock(size_t rowStart, size_t extent) const {
  leaf->scoreBlock(predictLeaves.get(), rowStart, extent);
}


void Predict::quantBlock(size_t rowStart, size_t extent) const {
  if (quant != nullptr) {
    quant->predictAcross(this, rowStart, extent);
  }
}


void PredictFrame::predictBlock(size_t rowStart) const {
  OMPBound row;
  OMPBound rowSup = (OMPBound) (rowStart + getExtent());

#pragma omp parallel default(shared) private(row) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = (OMPBound) rowStart; row < rowSup; row++) {
      (this->*PredictFrame::predictRow)(row, row - rowStart);
    }
  }
}


void PredictFrame::predictNum(size_t row, size_t rowOff) const {
  predict->rowNum(this, row, rowOff);
}


void PredictFrame::predictFac(size_t row, size_t rowOff) const {
  predict->rowFac(this, row, rowOff);
}


void PredictFrame::predictMixed(size_t row, size_t rowOff) const {
  predict->rowMixed(this, row, rowOff);
}


void Predict::rowNum(const PredictFrame* frame,
                     size_t row,
                     size_t blockRow) {
  auto rowT = frame->baseNum(blockRow);

  unsigned int tIdx = 0;
  for (auto orig : treeOrigin) {
    auto leafIdx = noLeaf;
    if (!bag->isBagged(oob, tIdx, row)) {
      auto idx = orig;
      do {
        idx += treeNode[idx].advance(rowT, leafIdx);
      } while (leafIdx == noLeaf);
    }
    predictLeaf(blockRow, tIdx++, leafIdx);
  }
}


void Predict::rowFac(const PredictFrame* frame,
                     size_t row,
                     size_t blockRow) {
  auto rowT = frame->baseFac(blockRow);

  unsigned int tIdx = 0;
  for (auto orig : treeOrigin) {
    auto leafIdx = noLeaf;
    if (!bag->isBagged(oob, tIdx, row)) {
      auto idx = orig;
      do {
        idx += treeNode[idx].advance(facSplit, rowT, tIdx, leafIdx);
      } while (leafIdx == noLeaf);
    }
    predictLeaf(blockRow, tIdx++, leafIdx);
  }
}


void Predict::rowMixed(const PredictFrame* frame,
                       size_t row,
                       size_t blockRow) {
  auto rowNT = frame->baseNum(blockRow);
  auto rowFT = frame->baseFac(blockRow);

  unsigned int tIdx = 0;
  for (auto orig : treeOrigin) {
    auto leafIdx = noLeaf;
    if (!bag->isBagged(oob, tIdx, row)) {
      auto idx = orig;
      do {
        idx += treeNode[idx].advance(frame, facSplit, rowFT, rowNT, tIdx, leafIdx);
      } while (leafIdx == noLeaf);
    }
    predictLeaf(blockRow, tIdx++, leafIdx);
  }
}


/**
   @brief Computes pointer to base of row of numeric values.

   @param rowOff is a block-relative row offset.

   @return base address for numeric values at row.
*/
const double* PredictFrame::baseNum(size_t rowOff) const {
  return blockNum->rowBase(rowOff);
}


/**
   @brief Computes pointer to base of row of factor values.

   @param rowOff is a block-relative row offset.

   @return base address for factor values at row.
*/
const unsigned int* PredictFrame::baseFac(size_t rowOff) const {
  return blockFac->rowBase(rowOff);
}
