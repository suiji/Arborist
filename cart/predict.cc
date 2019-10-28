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
#include "cartnode.h"
#include "quant.h"
#include "ompthread.h"


const size_t PredictFrame::rowBlock = 0x2000;

Predict::Predict(const Bag* bag_,
                 const Forest* forest,
                 LeafFrame* leaf_,
                 Quant* quant_,
                 bool oob_) :
  bag(bag_),
  treeOrigin(forest->cacheOrigin()),
  treeNode(forest->getNode()),
  facSplit(forest->getFacSplit()),
  leaf(leaf_),
  quant(quant_), 
  oob(oob_),
  nTree(forest->getNTree()),
  noLeaf(leaf->getNoLeaf()) {
}


PredictFrame::PredictFrame(Predict* predict_,
                           const BlockDense<double>* blockNum_,
                           const BlockDense<unsigned int>* blockFac_) :
  predict(predict_),
  nTree(predict->nTree),
  noLeaf(predict->noLeaf),
  blockNum(blockNum_),
  blockFac(blockFac_),
  predictRow(getNPredFac() == 0 ? &PredictFrame::predictNum : (getNPredNum() == 0 ? &PredictFrame::predictFac : &PredictFrame::predictMixed)),
  predictLeaves(make_unique<unsigned int[]>(getExtent() * nTree)) {
}


size_t PredictFrame::getBlockRows(size_t nRow) {
  return min(nRow, rowBlock);
}


void PredictFrame::predictAcross(size_t rowStart) {
  predictBlock(rowStart);
  predict->scoreBlock(predictLeaves.get(), rowStart, getExtent());
  predict->quantBlock(this, rowStart, getExtent());
}


void Predict::scoreBlock(const unsigned int predictLeaves[], size_t rowStart, size_t extent) const {
  leaf->scoreBlock(predictLeaves, rowStart, extent);
}


void Predict::quantBlock(const PredictFrame* frame, size_t rowStart, size_t extent) const {
  if (quant != nullptr) {
    quant->predictAcross(frame, rowStart, extent);
  }
}


void PredictFrame::predictBlock(size_t rowStart) {
  OMPBound rowSup = (OMPBound) (rowStart + getExtent());

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound row = (OMPBound) rowStart; row < rowSup; row++) {
      (this->*PredictFrame::predictRow)(row, row - rowStart);
    }
  }
}


void PredictFrame::predictNum(size_t row, size_t rowOff) {
  auto rowT = baseNum(rowOff);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    unsigned int leafIdx = predict->rowNum(tIdx, rowT, row);
    predictLeaf(rowOff, tIdx, leafIdx);
  }
}


void PredictFrame::predictFac(size_t row, size_t rowOff)  {
  auto rowT = baseFac(rowOff);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    unsigned int leafIdx = predict->rowFac(tIdx, rowT, row);
    predictLeaf(rowOff, tIdx, leafIdx);
  }
}


void PredictFrame::predictMixed(size_t row, size_t rowOff) {
  const double* rowNT = baseNum(rowOff);
  const unsigned int* rowFT = baseFac(rowOff);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    unsigned int leafIdx = predict->rowMixed(tIdx, this, rowNT, rowFT, row);
    predictLeaf(rowOff, tIdx, leafIdx);
  }
}


unsigned int Predict::rowNum(unsigned int tIdx,
                             const double* rowT,
                             size_t row) {
  unsigned int leafIdx = noLeaf;
  if (!bag->isBagged(oob, tIdx, row)) {
    auto idx = treeOrigin[tIdx];//orig;
    do {
      idx += treeNode[idx].advance(rowT, leafIdx);
    } while (leafIdx == noLeaf);
  }
  return leafIdx;
}


unsigned int Predict::rowFac(const unsigned int tIdx,
                             const unsigned int* rowT,
                             size_t row) {
  unsigned int leafIdx = noLeaf;
  if (!bag->isBagged(oob, tIdx, row)) {
    auto idx = treeOrigin[tIdx];//orig;
    do {
      idx += treeNode[idx].advance(facSplit, rowT, tIdx, leafIdx);
    } while (leafIdx == noLeaf);
  }
  return leafIdx;
}


unsigned int Predict::rowMixed(unsigned int tIdx,
                               const PredictFrame* frame,
                               const double* rowNT,
                               const unsigned int* rowFT,
                               size_t row) {
  unsigned int leafIdx = noLeaf;
  if (!bag->isBagged(oob, tIdx, row)) {
    auto idx = treeOrigin[tIdx];
    do {
      idx += treeNode[idx].advance(frame, facSplit, rowFT, rowNT, tIdx, leafIdx);
    } while (leafIdx == noLeaf);
  }

  return leafIdx;
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
