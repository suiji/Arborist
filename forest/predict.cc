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

#include "sampler.h"
#include "forest.h"
#include "predict.h"
#include "bv.h"
#include "treenode.h"
#include "quant.h"
#include "ompthread.h"
#include "rleframe.h"
#include "bheap.h"
#include "leaf.h"

#include <cmath>
const size_t Predict::scoreChunk = 0x2000;
const unsigned int Predict::seqChunk = 0x20;


Predict::Predict(const Forest* forest,
		 const Sampler* sampler_,
		 RLEFrame* rleFrame_,
		 bool testing_,
		 unsigned int nPermute_) :
  sampler(sampler_),
  treeOrigin(forest->treeOrigins()),
  treeNode(forest->getNode()),
  facSplit(forest->getFacSplit()),
  rleFrame(rleFrame_),
  testing(testing_),
  nPermute(nPermute_),
  predictLeaves(vector<IndexT>(scoreChunk * forest->getNTree())),
  accumNEst(vector<IndexT>(scoreChunk)),
  scoreBlock(forest->getScores()),
  scoreHeight(scoreHeights(scoreBlock)),
  nPredNum(rleFrame->getNPredNum()),
  nPredFac(rleFrame->getNPredFac()),
  nRow(rleFrame->getNRow()),
  nTree(forest->getNTree()),
  noLeaf(scoreHeight.back()),
  walkTree(nPredFac == 0 ? &Predict::walkNum : (nPredNum == 0 ? &Predict::walkFac : &Predict::walkMixed)),
  trFac(vector<unsigned int>(scoreChunk * nPredFac)),
  trNum(vector<double>(scoreChunk * nPredNum)) {
  rleFrame->reorderRow(); // For now, all frames pre-ranked.
}


vector<size_t> Predict::scoreHeights(const vector<vector<double>>& scoreBlock) {
  vector<size_t> scoreHeight;
  size_t height = 0; // Accumulated height.
  for (auto treeScores : scoreBlock) {
    height += treeScores.size();
    scoreHeight.push_back(height);
  }

  return scoreHeight;
}


void Predict::sampleBounds(unsigned int tIdx,
			   IndexT leafIdx,
			   size_t& leafStart,
			   size_t& leafEnd) const {
  sampler->getSampleBounds(getScoreIdx(tIdx, leafIdx), leafStart, leafEnd);
}


PredictReg::PredictReg(const Forest* forest,
		       const Sampler* sampler_,
		       RLEFrame* rleFrame,
		       const vector<double>& yTest_,
		       unsigned int nPermute_,
		       const vector<double>& quantile) :
  Predict(forest, sampler_, rleFrame, !yTest_.empty(), nPermute_),
  leaf(reinterpret_cast<const LeafReg*>(sampler->getLeaf())),
  yTest(move(yTest_)),
  yPred(vector<double>(nRow)),
  yPermute(vector<double>(nPermute > 0 ? nRow : 0)),
  accumAbsErr(vector<double>(scoreChunk)),
  accumSSE(vector<double>(scoreChunk)),
  saePermute(nPermute > 0 ? rleFrame->getNPred() : 0),
  ssePermute(nPermute > 0 ? rleFrame->getNPred() : 0),
  quant(make_unique<Quant>(this, leaf, rleFrame, move(quantile))),
  yTarg(&yPred),
  saeTarg(&saePredict),
  sseTarg(&ssePredict) {
}


PredictCtg::PredictCtg(const Forest* forest,
		       const Sampler* sampler_,
		       RLEFrame* rleFrame,
		       const vector<PredictorT>& yTest_,
		       unsigned int nPermute_,
		       bool doProb) :
  Predict(forest, sampler_, rleFrame, !yTest_.empty(), nPermute_),
  leaf(reinterpret_cast<const LeafCtg*>(sampler->getLeaf())),
  yTest(move(yTest_)),
  yPred(vector<PredictorT>(nRow)),
  nCtgTrain(leaf->getNCtg()),
  nCtgMerged(testing ? 1 + *max_element(yTest.begin(), yTest.end()) : 0),
  ctgProb(make_unique<CtgProb>(this, leaf, sampler, doProb)),
  // Can only predict trained categories, so census and
  // probability matrices have 'nCtgTrain' columns.
  yPermute(vector<PredictorT>(nPermute > 0 ? nRow : 0)),
  census(vector<PredictorT>(nRow * nCtgTrain)),
  confusion(vector<size_t>(nCtgTrain * nCtgMerged)),
  misprediction(vector<double>(nCtgMerged)),
  oobPredict(0.0),
  censusPermute(vector<PredictorT>(nPermute > 0 ? census.size() : 0)),
  confusionPermute(vector<size_t>(nPermute > 0 ? confusion.size() : 0)),
  mispredPermute(vector<vector<double>>(nPermute > 0 ? rleFrame->getNPred(): 0)),
  oobPermute(vector<double>(nPermute > 0 ? rleFrame->getNPred() : 0)),
  yTarg(&yPred),
  confusionTarg(&confusion),
  censusTarg(&census),
  mispredTarg(&misprediction),
  oobTarg(&oobPredict) {
}


void Predict::predict() {
  blocks();
  predictPermute();
}


void Predict::predictPermute() {
  if (nPermute == 0) {
    return;
  }
  
  for (PredictorT predIdx = 0; predIdx < rleFrame->getNPred(); predIdx++) {
    setPermuteTarget(predIdx);
    vector<RLEVal<unsigned int>> rleTemp = move(rleFrame->rlePred[predIdx]);
    rleFrame->rlePred[predIdx] = rleFrame->permute(predIdx, BHeap::permute(nRow));
    blocks();
    rleFrame->rlePred[predIdx] = move(rleTemp);
  }
}


void PredictReg::setPermuteTarget(PredictorT predIdx) {
  yTarg = &yPermute;
  sseTarg = &ssePermute[predIdx];
  saeTarg = &saePermute[predIdx];
  fill(accumSSE.begin(), accumSSE.end(), 0.0);
  fill(accumAbsErr.begin(), accumAbsErr.end(), 0.0);
}


void PredictCtg::setPermuteTarget(PredictorT predIdx) {
  mispredPermute[predIdx] = move(vector<double>(nCtgMerged));
  yTarg = &yPermute;
  confusionTarg = &confusionPermute;
  censusTarg = &censusPermute;
  mispredTarg = &mispredPermute[predIdx];
  oobTarg = &oobPermute[predIdx];
  fill(confusionPermute.begin(), confusionPermute.end(), 0);
  fill(censusPermute.begin(), censusPermute.end(), 0);
}


void Predict::blocks() {
  vector<size_t> trIdx(nPredNum + nPredFac);
  size_t row = predictBlock(0, nRow, trIdx);
  // Remainder rows handled in custom-fitted block.
  if (nRow > row) {
    (void) predictBlock(row, nRow, trIdx);
  }

  estAccum();
}


size_t Predict::predictBlock(size_t rowStart,
			     size_t rowEnd,
			     vector<size_t>& trIdx) {
  size_t blockRows = min(scoreChunk, rowEnd - rowStart);
  size_t row = rowStart;
  for (; row + blockRows <= rowEnd; row += blockRows) {
    rleFrame->transpose(trIdx, row, scoreChunk, trFac, trNum);
    fill(predictLeaves.begin(), predictLeaves.end(), noLeaf);
    blockStart = row;
    blockEnd = row + blockRows;
    predictBlock();
  }
  return row;
}


void Predict::predictBlock() {
  OMPBound rowEnd = static_cast<OMPBound>(blockEnd);
  OMPBound rowStart = static_cast<OMPBound>(blockStart);

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound row = rowStart; row < rowEnd; row += seqChunk) {
    scoreSeq(row, min(rowEnd, row + seqChunk));
  }
  }
}


// Sequential inner loop to avoid false sharing.
void PredictReg::scoreSeq(size_t rowStart, size_t rowEnd) {
  for (size_t row = rowStart; row != rowEnd; row++) {
    (this->*Predict::walkTree)(row);
    testing ? testRow(row) : (void) scoreRow(row);
  }
}


void PredictCtg::scoreSeq(size_t rowStart, size_t rowEnd) {
  for (size_t row = rowStart; row != rowEnd; row++) {
    (this->*Predict::walkTree)(row);
    testing ? testRow(row) : scoreRow(row);
  }
}



unsigned int PredictReg::scoreRow(size_t row) {
  (*yTarg)[row] = leaf->predictObs(this, row);
  if (!quant->isEmpty()) {
    quant->predictRow(this, row);
  }
  return nEst;
}


void PredictCtg::scoreRow(size_t row) {
  (*yTarg)[row] = leaf->predictObs(this, row, &census[ctgIdx(row)]);
  if (!ctgProb->isEmpty())
    ctgProb->predictRow(this, row, &census[ctgIdx(row)]);
}


void PredictReg::testRow(size_t row) {
  IndexT rowIdx = row - blockStart;
  accumNEst[rowIdx] += scoreRow(row);
  double testError = fabs(yTest[row] - (*yTarg)[row]);
  accumAbsErr[rowIdx] += testError;
  accumSSE[rowIdx] += testError * testError;
}


void PredictCtg::testRow(size_t row) {
  scoreRow(row);
}


const vector<double>  PredictReg::getQPred() const {
  return quant->getQPred();
}


const vector<double> PredictReg::getQEst() const {
  return quant->getQEst();
}


void Predict::walkNum(size_t row) {
  auto rowT = baseNum(row);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (!sampler->isBagged(tIdx, row)) {
      rowNum(tIdx, rowT, row);
    }
  }
}


void Predict::walkFac(size_t row) {
  auto rowT = baseFac(row);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (!sampler->isBagged(tIdx, row)) {
      rowFac(tIdx, rowT, row);
    }
  }
}


void Predict::walkMixed(size_t row) {
  const double* rowNT = baseNum(row);
  const PredictorT* rowFT = baseFac(row);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (!sampler->isBagged(tIdx, row)) {
      rowMixed(tIdx, rowNT, rowFT, row);
    }
  }
}


void Predict::rowNum(unsigned int tIdx,
		       const double* rowT,
		       size_t row) {
  IndexT leafIdx = noLeaf;
  auto idx = treeOrigin[tIdx];
  do {
    idx += treeNode[idx].advance(rowT, leafIdx);
  } while (leafIdx == noLeaf);

  predictLeaf(row, tIdx, leafIdx);
}


void Predict::rowFac(const unsigned int tIdx,
		     const unsigned int* rowT,
		     size_t row) {
  IndexT leafIdx = noLeaf;
  auto idx = treeOrigin[tIdx];
  do {
    idx += treeNode[idx].advance(facSplit, rowT, tIdx, leafIdx);
  } while (leafIdx == noLeaf);

  predictLeaf(row, tIdx, leafIdx);
}


void Predict::rowMixed(unsigned int tIdx,
			 const double* rowNT,
			 const unsigned int* rowFT,
			 size_t row) {
  IndexT leafIdx = noLeaf;
  auto idx = treeOrigin[tIdx];
  do {
    idx += treeNode[idx].advance(this, facSplit, rowFT, rowNT, tIdx, leafIdx);
  } while (leafIdx == noLeaf);

  predictLeaf(row, tIdx, leafIdx);
}


const double* Predict::baseNum(size_t row) const {
  return &trNum[(row - blockStart) * nPredNum];
}
  

const PredictorT* Predict::baseFac(size_t row) const {
  return &trFac[(row - blockStart) * nPredFac];
}


void Predict::estAccum() {
  nEst = accumulate(accumNEst.begin(), accumNEst.end(), 0);
}


void PredictReg::estAccum() {
  Predict::estAccum();
  *saeTarg = accumulate(accumAbsErr.begin(), accumAbsErr.end(), 0.0);
  *sseTarg = accumulate(accumSSE.begin(), accumSSE.end(), 0.0);
}


void PredictCtg::estAccum() {
  Predict::estAccum();
  if (!(*confusionTarg).empty()) {
    for (size_t row = 0; row < nRow; row++) {
      (*confusionTarg)[ctgIdx(yTest[row], (*yTarg)[row])]++;
    }
    setMisprediction();
  }
}


void PredictCtg::setMisprediction() {
  size_t totRight = 0;
  for (PredictorT ctgRec = 0; ctgRec < nCtgMerged; ctgRec++) {
    size_t numWrong = 0;
    size_t numRight = 0;
    for (PredictorT ctgPred = 0; ctgPred < nCtgTrain; ctgPred++) {
      size_t numConf = (*confusionTarg)[ctgIdx(ctgRec, ctgPred)];
      if (ctgPred != ctgRec) {  // Misprediction iff off-diagonal.
        numWrong += numConf;
      }
      else {
        numRight = numConf;
      }
    }
    
    (*mispredTarg)[ctgRec] = numWrong + numRight == 0 ? 0.0 : double(numWrong) / double(numWrong + numRight);
    totRight += numRight;
  }
  *oobTarg = double(totRight) / nRow;
}


  /**
     @brief Getter for probability matrix.
   */

const vector<double>& PredictCtg::getProb() const {
  return ctgProb->getProb();
}

  
void PredictCtg::dump() const {
  // TODO
  //  ctgProb->dump(probTree);
}
