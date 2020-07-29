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
#include "forest.h"
#include "leaf.h"
#include "predict.h"
#include "bv.h"
#include "treenode.h"
#include "quant.h"
#include "ompthread.h"
#include "rleframe.h"

const size_t Predict::rowBlock = 0x2000;

Predict::Predict(const Bag* bag_,
                 const Forest* forest,
                 LeafFrame* leaf_,
		 RLEFrame* rleFrame_,
                 Quant* quant_,
                 bool oob_) :
  bag(bag_),
  treeOrigin(forest->cacheOrigin()),
  treeNode(forest->getNode()),
  facSplit(forest->getFacSplit()),
  leaf(leaf_),
  rleFrame(rleFrame_),
  quant(quant_),
  oob(oob_),
  nPredNum(rleFrame->getNPredNum()),
  nPredFac(rleFrame->getNPredFac()),
  nTree(forest->getNTree()),
  noLeaf(leaf->getNoLeaf()),
  trFac(vector<unsigned int>(rowBlock * nPredFac)),
  trNum(vector<double>(rowBlock * nPredNum)),
  trIdx(vector<size_t>(nPredNum + nPredFac)) {
  rleFrame->reorderRow(); // For now, all frames pre-ranked.
}


PredictFrame::PredictFrame(Predict* predict_,
			   IndexT extent_) :
  predict(predict_),
  nTree(predict->nTree),
  noLeaf(predict->noLeaf),
  extent(extent_),
  predictRow(predict->nPredFac == 0 ? &PredictFrame::predictNum : (predict->nPredNum == 0 ? &PredictFrame::predictFac : &PredictFrame::predictMixed)),
  predictLeaves(make_unique<IndexT[]>(extent * nTree)) {
}


size_t Predict::getBlockRows(size_t nRow) {
  return min(nRow, rowBlock);
}


void PredictFrame::predictAcross(size_t rowStart) {
  predict->transpose(rowStart);
  predictBlock(rowStart);
  predict->scoreBlock(predictLeaves.get(), rowStart, extent);
  predict->quantBlock(this, rowStart, extent);
}


void Predict::transpose(size_t rowStart) {
  rleFrame->transpose(trIdx, rowStart, rowBlock, trFac, trNum);
}


void PredictFrame::predictBlock(size_t rowStart) {
  OMPBound rowSup = static_cast<OMPBound>(rowStart + extent);

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound row = (OMPBound) rowStart; row < rowSup; row++) {
      (this->*PredictFrame::predictRow)(row, row - rowStart);
    }
  }
}


void Predict::scoreBlock(const IndexT predictLeaves[], size_t rowStart, size_t extent) const {
  leaf->scoreBlock(predictLeaves, rowStart, extent);
}


void Predict::quantBlock(const PredictFrame* frame, size_t rowStart, size_t extent) const {
  if (quant != nullptr) {
    quant->predictAcross(frame, rowStart, extent);
  }
}


void PredictFrame::predictNum(size_t row, size_t rowOff) {
  auto rowT = predict->baseNum(rowOff);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT leafIdx = predict->rowNum(tIdx, rowT, row);
    predictLeaf(rowOff, tIdx, leafIdx);
  }
}


void PredictFrame::predictFac(size_t row, size_t rowOff)  {
  auto rowT = predict->baseFac(rowOff);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT leafIdx = predict->rowFac(tIdx, rowT, row);
    predictLeaf(rowOff, tIdx, leafIdx);
  }
}


void PredictFrame::predictMixed(size_t row, size_t rowOff) {
  const double* rowNT = predict->baseNum(rowOff);
  const PredictorT* rowFT = predict->baseFac(rowOff);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT leafIdx = predict->rowMixed(tIdx, rowNT, rowFT, row);
    predictLeaf(rowOff, tIdx, leafIdx);
  }
}


IndexT Predict::rowNum(unsigned int tIdx,
		       const double* rowT,
		       size_t row) {
  IndexT leafIdx = noLeaf;
  if (!bag->isBagged(oob, tIdx, row)) {
    auto idx = treeOrigin[tIdx];
    do {
      idx += treeNode[idx].advance(rowT, leafIdx);
    } while (leafIdx == noLeaf);
  }
  return leafIdx;
}


IndexT Predict::rowFac(const unsigned int tIdx,
		       const unsigned int* rowT,
		       size_t row) {
  IndexT leafIdx = noLeaf;
  if (!bag->isBagged(oob, tIdx, row)) {
    auto idx = treeOrigin[tIdx];
    do {
      idx += treeNode[idx].advance(facSplit, rowT, tIdx, leafIdx);
    } while (leafIdx == noLeaf);
  }
  return leafIdx;
}


IndexT Predict::rowMixed(unsigned int tIdx,
			 const double* rowNT,
			 const unsigned int* rowFT,
			 size_t row) {
  IndexT leafIdx = noLeaf;
  if (!bag->isBagged(oob, tIdx, row)) {
    auto idx = treeOrigin[tIdx];
    do {
      idx += treeNode[idx].advance(this, facSplit, rowFT, rowNT, tIdx, leafIdx);
    } while (leafIdx == noLeaf);
  }

  return leafIdx;
}


const double* Predict::baseNum(size_t rowOff) const {
  return &trNum[rowOff * rleFrame->getNPredNum()];
}
  

const PredictorT* Predict::baseFac(size_t rowOff) const {
  return &trFac[rowOff * rleFrame->getNPredFac()];
}
