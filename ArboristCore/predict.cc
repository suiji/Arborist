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
#include "ompthread.h"


PredictBox::PredictBox(const FramePredict* framePredict_,
                       const Forest* forest_,
                       const BitMatrix* bag_,
                       LeafFrame* leafFrame_,
                       unsigned int nThread) :
  framePredict(framePredict_),
  forest(forest_),
  bag(bag_),
  leafFrame(leafFrame_) {
  OmpThread::init(nThread);
}

PredictBox::~PredictBox() {
  OmpThread::deInit();
}


Predict::Predict(const PredictBox* box) :
  framePredict(box->framePredict),
  forest(box->forest),
  nTree(forest->getNTree()),
  nRow(framePredict->getNRow()),
  treeOrigin(forest->cacheOrigin()) {
  predictLeaves = make_unique<unsigned int[]>(rowBlock * nTree);
}

void Predict::predict(const PredictBox* box) {
  auto predict = make_unique<Predict>(box);
  predict->predictAcross(box->leafFrame, box->bag);
}


unique_ptr<Quant> Predict::predictQuant(const PredictBox* box, const double* quantile, unsigned int nQuant, unsigned int qBin) {
  auto quant = make_unique<Quant>(static_cast<LeafFrameReg*>(box->leafFrame), box->bag, quantile, nQuant, qBin);
  auto predict = make_unique<Predict>(box);
  predict->predictAcross(box->leafFrame, box->bag, quant.get());

  return move(quant);
}


void Predict::predictAcross(LeafFrame* leaf, const BitMatrix *bag, Quant *quant) {
  noLeaf = leaf->getNoLeaf();
  for (unsigned int rowStart = 0; rowStart < nRow; rowStart += rowBlock) {
    unsigned int rowEnd = min(rowStart + rowBlock, nRow);
    framePredict->transpose(rowStart, rowEnd);
    predictBlock(rowStart, rowEnd, bag);
    leaf->scoreBlock(predictLeaves.get(), rowStart, rowEnd);
    if (quant != nullptr) {
      quant->predictAcross(this, rowStart, rowEnd);
    }
  }
}


void Predict::predictBlock(unsigned int rowStart,
                           unsigned int rowEnd,
                           const BitMatrix *bag) {
  if (framePredict->getNPredFac() == 0)
    predictBlockNum(rowStart, rowEnd, bag);
  else if (framePredict->getNPredNum() == 0)
    predictBlockFac(rowStart, rowEnd, bag);
  else
    predictBlockMixed(rowStart, rowEnd, bag);
}


void Predict::predictBlockNum(unsigned int rowStart,
                              unsigned int rowEnd,
                              const BitMatrix *bag) {
  OMPBound row;
  OMPBound rowSup = (OMPBound) rowEnd;

#pragma omp parallel default(shared) private(row) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = (OMPBound) rowStart; row < rowSup; row++) {
      rowNum(row, row - rowStart, forest->getNode(), bag);
    }
  }
}


void Predict::predictBlockFac(unsigned int rowStart,
                              unsigned int rowEnd,
                              const BitMatrix *bag) {
  OMPBound row;
  OMPBound rowSup = (OMPBound) rowEnd;

#pragma omp parallel default(shared) private(row) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = (OMPBound) rowStart; row < rowSup; row++) {
      rowFac(row, row - rowStart, forest->getNode(), forest->getFacSplit(), bag);
  }
  }

}


void Predict::predictBlockMixed(unsigned int rowStart,
                                unsigned int rowEnd,
                                const BitMatrix *bag) {
  OMPBound row;
  OMPBound rowSup = (OMPBound) rowEnd;

#pragma omp parallel default(shared) private(row) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (row = (OMPBound) rowStart; row < rowSup; row++) {
      rowMixed(row, row - rowStart, forest->getNode(), forest->getFacSplit(), bag);
    }
  }
}


void Predict::rowNum(unsigned int row,
                     unsigned int blockRow,
                     const TreeNode *treeNode,
                     const class BitMatrix *bag) {
  auto rowT = framePredict->baseNum(blockRow);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    auto leafIdx = noLeaf;
    if (!(bag != nullptr && bag->testBit(tIdx, row))) {
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
                     const BVJagged *facSplit,
                     const BitMatrix *bag) {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    auto leafIdx = noLeaf;
    if (!(bag != nullptr && bag->testBit(tIdx, row))) {
      auto rowT = framePredict->baseFac(blockRow);
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
                       const BVJagged *facSplit,
                       const BitMatrix *bag) {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    auto leafIdx = noLeaf;
    if (!(bag != nullptr && bag->testBit(tIdx, row))) {
      auto rowNT = framePredict->baseNum(blockRow);
      auto rowFT = framePredict->baseFac(blockRow);
      auto idx = treeOrigin[tIdx];
      do {
        idx += treeNode[idx].advance(framePredict, facSplit, rowFT, rowNT, tIdx, leafIdx);
      } while (leafIdx == noLeaf);
    }
    predictLeaf(blockRow, tIdx, leafIdx);
  }
}
