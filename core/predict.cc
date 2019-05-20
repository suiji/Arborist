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

#include "blockframe.h"
#include "forest.h"
#include "leaf.h"
#include "predict.h"
#include "bv.h"
#include "quant.h"
#include "ompthread.h"


PredictBox::PredictBox(bool oob_,
                       const BlockFrame* blockFrame_,
                       const Forest* forest_,
                       const BitMatrix* bag_,
                       LeafFrame* leafFrame_,
                       unsigned int nThread) :
  oob(oob_),
  blockFrame(blockFrame_),
  forest(forest_),
  bag(bag_),
  leafFrame(leafFrame_) {
  OmpThread::init(nThread);
}

PredictBox::~PredictBox() {
  OmpThread::deInit();
}


Predict::Predict(const PredictBox* box) :
  blockFrame(box->blockFrame),
  forest(box->forest),
  nTree(forest->getNTree()),
  nRow(blockFrame->getNRow()),
  treeOrigin(forest->cacheOrigin()),
  leaf(box->leafFrame),
  noLeaf(leaf->getNoLeaf()),
  bag(box->bag),
  oob(box->oob) {
  predictLeaves = make_unique<unsigned int[]>(rowBlock * nTree);
}


void Predict::predict(const PredictBox* box, Quant* quant) {
  auto predict = make_unique<Predict>(box);
  predict->predictAcross(quant);
}


void Predict::predictAcross(Quant *quant) {
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += rowBlock) {
    unsigned int rowEnd = min(rowStart + rowBlock, nRow);
    blockFrame->reWindow(rowStart, rowEnd, rowBlock);
    predictBlock(rowStart, rowEnd);
    leaf->scoreBlock(predictLeaves.get(), rowStart, rowEnd);
    if (quant != nullptr) {
      quant->predictAcross(this, rowStart, rowEnd);
    }
  }
}


void Predict::predictBlock(unsigned int rowStart,
                           unsigned int rowEnd) {
  if (blockFrame->getNPredFac() == 0)
    predictBlockNum(rowStart, rowEnd);
  else if (blockFrame->getNPredNum() == 0)
    predictBlockFac(rowStart, rowEnd);
  else
    predictBlockMixed(rowStart, rowEnd);
}


void Predict::predictBlockNum(unsigned int rowStart,
                              unsigned int rowEnd) {
  OMPBound row;
  OMPBound rowSup = (OMPBound) rowEnd;

#pragma omp parallel default(shared) private(row) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = (OMPBound) rowStart; row < rowSup; row++) {
      rowNum(row, row - rowStart, forest->getNode());
    }
  }
}


void Predict::predictBlockFac(unsigned int rowStart,
                              unsigned int rowEnd) {
  OMPBound row;
  OMPBound rowSup = (OMPBound) rowEnd;

#pragma omp parallel default(shared) private(row) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = (OMPBound) rowStart; row < rowSup; row++) {
      rowFac(row, row - rowStart, forest->getNode(), forest->getFacSplit());
  }
  }

}


void Predict::predictBlockMixed(unsigned int rowStart,
                                unsigned int rowEnd) {
  OMPBound row;
  OMPBound rowSup = (OMPBound) rowEnd;

#pragma omp parallel default(shared) private(row) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = (OMPBound) rowStart; row < rowSup; row++) {
      rowMixed(row, row - rowStart, forest->getNode(), forest->getFacSplit());
    }
  }
}


void Predict::rowNum(unsigned int row,
                     unsigned int blockRow,
                     const TreeNode *treeNode) {
  auto rowT = blockFrame->baseNum(blockRow);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    auto leafIdx = noLeaf;
    if (!(oob && bag->testBit(tIdx, row))) {
      auto idx = treeOrigin[tIdx];
      do {
        idx += treeNode[idx].advance(rowT, leafIdx);
      } while (leafIdx == noLeaf);
    }
    predictLeaf(blockRow, tIdx, leafIdx);
  }
}


void Predict::rowFac(unsigned int row,
                     unsigned int blockRow,
                     const TreeNode *treeNode,
                     const BVJagged *facSplit) {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    auto leafIdx = noLeaf;
    if (!(oob && bag->testBit(tIdx, row))) {
      auto rowT = blockFrame->baseFac(blockRow);
      auto idx = treeOrigin[tIdx];
      do {
        idx += treeNode[idx].advance(facSplit, rowT, tIdx, leafIdx);
      } while (leafIdx == noLeaf);
    }
    predictLeaf(blockRow, tIdx, leafIdx);
  }
}


void Predict::rowMixed(unsigned int row,
                       unsigned int blockRow,
                       const TreeNode *treeNode,
                       const BVJagged *facSplit) {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    auto leafIdx = noLeaf;
    if (!(oob && bag->testBit(tIdx, row))) {
      auto rowNT = blockFrame->baseNum(blockRow);
      auto rowFT = blockFrame->baseFac(blockRow);
      auto idx = treeOrigin[tIdx];
      do {
        idx += treeNode[idx].advance(blockFrame, facSplit, rowFT, rowNT, tIdx, leafIdx);
      } while (leafIdx == noLeaf);
    }
    predictLeaf(blockRow, tIdx, leafIdx);
  }
}
