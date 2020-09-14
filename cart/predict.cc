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
#include "leafpredict.h"
#include "predict.h"
#include "bv.h"
#include "treenode.h"
#include "quant.h"
#include "ompthread.h"
#include "rleframe.h"
#include "bheap.h"

#include <numeric>

const size_t Predict::rowChunk = 0x2000;

Predict::Predict(const Bag* bag_,
                 const Forest* forest,
                 const LeafPredict* leaf,
		 RLEFrame* rleFrame_,
		 bool oob_,
		 unsigned int nPermute_) :
  bag(bag_),
  treeOrigin(forest->cacheOrigin()),
  treeNode(forest->getNode()),
  facSplit(forest->getFacSplit()),
  rleFrame(rleFrame_),
  oob(oob_),
  nPermute(nPermute_),
  predictLeaves(vector<IndexT>(rowChunk * forest->getNTree())),
  accumNEst(vector<IndexT>(rowChunk)),
  leafBlock(leaf->getLeafBlock()),
  nPredNum(rleFrame->getNPredNum()),
  nPredFac(rleFrame->getNPredFac()),
  nRow(rleFrame->getNRow()),
  nTree(forest->getNTree()),
  noLeaf(leaf->getNoLeaf()),
  walkTree(nPredFac == 0 ? &Predict::walkNum : (nPredNum == 0 ? &Predict::walkFac : &Predict::walkMixed)),
  trFac(vector<unsigned int>(rowChunk * nPredFac)),
  trNum(vector<double>(rowChunk * nPredNum)),
  trIdx(vector<size_t>(nPredNum + nPredFac)) {
  rleFrame->reorderRow(); // For now, all frames pre-ranked.
}


PredictReg::PredictReg(const Bag* bag_,
			 const Forest* forest,
			 const LeafPredict* leaf,
			 RLEFrame* rleFrame,
			 vector<double> yTrain,
			 double defaultScore_,
			 vector<double> yTest_,
			 bool oob_,
			 unsigned int nPermute_,
			 vector<double> quantile) :
  Predict(bag_, forest, leaf, rleFrame, oob_, nPermute_),
  defaultScore(defaultScore_),
  yTest(move(yTest_)),
  yPred(vector<double>(nRow)),
  yPermute(vector<double>(nPermute > 0 ? nRow : 0)),
  accumAbsErr(vector<double>(rowChunk)),
  accumSSE(vector<double>(rowChunk)),
  saePermute(nPermute > 0 ? rleFrame->getNPred() : 0),
  ssePermute(nPermute > 0 ? rleFrame->getNPred() : 0),
  quant(quantile.empty() ? nullptr : make_unique<Quant>(this, leaf, bag, rleFrame, move(yTrain), move(quantile))),
  yTarg(&yPred),
  saeTarg(&saePredict),
  sseTarg(&ssePredict) {
}


PredictReg::~PredictReg() {
}

PredictCtg::PredictCtg(const Bag* bag_,
			 const Forest* forest,
			 const LeafPredict* leaf,
			 RLEFrame* rleFrame,
			 const unsigned int* leafHeight,
			 const double* leafProbs,
			 unsigned int nCtgTrain_,
		       vector<PredictorT> yTest_,
			 bool oob_,
			 unsigned int nPermute_,
			 bool doProb) :
  Predict(bag_, forest, leaf, rleFrame, oob_, nPermute_),
  yTest(move(yTest_)),
  yPred(vector<PredictorT>(nRow)),
  nCtgTrain(nCtgTrain_),
  nCtgMerged(yTest.empty() ? 0 : 1 + *max_element(yTest.begin(), yTest.end())),
  ctgProb(make_unique<CtgProb>(nCtgTrain, nTree, leafHeight, leafProbs)),
  ctgDefault(ctgProb->ctgDefault()),
  // Can only predict trained categories, so census and
  // probability matrices have 'nCtgTrain' columns.
  yPermute(vector<PredictorT>(nPermute > 0 ? nRow : 0)),
  votes(vector<double>(nRow * nCtgTrain)),
  census(vector<PredictorT>(nRow * nCtgTrain)),
  confusion(vector<size_t>(nCtgTrain * nCtgMerged)),
  misprediction(vector<double>(nCtgMerged)),
  oobPredict(0.0),
  prob(vector<double>(doProb ? votes.size() : 0)),
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
    fill(trIdx.begin(), trIdx.end(), 0); // Resets trace counters.
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
  mispredPermute[predIdx] = vector<double>(nCtgMerged);
  yTarg = &yPermute;
  confusionTarg = &confusionPermute;
  censusTarg = &censusPermute;
  mispredTarg = &mispredPermute[predIdx];
  oobTarg = &oobPermute[predIdx];
  fill(confusionPermute.begin(), confusionPermute.end(), 0);
  fill(censusPermute.begin(), censusPermute.end(), 0);
}


void Predict::blocks() {
  size_t row = predictBlock(0, nRow);
  // Remainder rows handled in custom-fitted block.
  if (nRow > row) {
    (void) predictBlock(row, nRow);
  }

  estAccum();
}


size_t Predict::predictBlock(size_t rowStart,
			     size_t rowEnd) {
  size_t blockRows = min(rowChunk, rowEnd - rowStart);
  size_t row = rowStart;
  for (; row + blockRows <= rowEnd; row += blockRows) {
    rleFrame->transpose(trIdx, row, rowChunk, trFac, trNum);
    fill(predictLeaves.begin(), predictLeaves.end(), noLeaf);
    blockStart = row;
    blockEnd = row + blockRows;
    predictBlock();
  }
  return row;
}


void PredictReg::predictBlock() {
  OMPBound rowEnd = static_cast<OMPBound>(blockEnd);
  OMPBound rowStart = static_cast<OMPBound>(blockStart);
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound row = rowStart; row < rowEnd; row++) {
    (this->*Predict::walkTree)(row);
    yTest.empty() ? scoreRow(row) : testRow(row);
  }
  }
  if (quant != nullptr) {
    quant->predictBlock(blockStart, blockEnd);
  }
}


// Scores each row independently, in parallel.
void PredictCtg::predictBlock() {
  OMPBound rowEnd = static_cast<OMPBound>(blockEnd);
  OMPBound rowStart = static_cast<OMPBound>(blockStart);

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound row = rowStart; row < rowEnd; row++) {
    (this->*Predict::walkTree)(row);
    /* yTest.empty() ? */ scoreRow(row);
    if (!prob.empty()) {
      ctgProb->probAcross(this, row, &prob[ctgIdx(row)]);
    }
  }
  }
}


void PredictReg::scoreRow(size_t row) {
  double sumScore = 0.0;
  IndexT nEst = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    double score;
    if (isLeafIdx(row, tIdx, score)) {
      nEst++;
      sumScore += score;
    }
  }

  (*yTarg)[row] = nEst > 0 ? sumScore / nEst : defaultScore;
}


void PredictReg::testRow(size_t row) {
  IndexT rowIdx = row - blockStart;
  unsigned int& nEst = accumNEst[rowIdx];
  unsigned int nEstStart = nEst;
  double& absError = accumAbsErr[rowIdx];
  double& sse = accumSSE[rowIdx];
  double sumScore = 0.0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    double score;
    if (isLeafIdx(row, tIdx, score)) {
      nEst++;
      sumScore += score;
    }
  }

  (*yTarg)[row] = nEst > nEstStart ? sumScore / (nEst - nEstStart) : defaultScore;
  double testError = fabs(yTest[row] - (*yTarg)[row]);
  absError += testError;
  sse += testError * testError;
}


void PredictCtg::scoreRow(size_t row) {
  unsigned int treesSeen = 0;
  double* blockVotes = &votes[ctgIdx(row)];
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    double score;
    if (isLeafIdx(row, tIdx, score)) {
      treesSeen++;
      PredictorT ctg = floor(score); // Truncates jittered score for indexing.
      blockVotes[ctg] += (1.0 + score) - ctg; // 1 plus small jitter.
    }
  }
  if (treesSeen == 0) { // Default category unity, all others zero.
    blockVotes[ctgDefault] = 1.0;
  }
  (*yTarg)[row] = argMax(row);
}


PredictorT PredictCtg::argMax(size_t row) {
  const double* blockVotes = &votes[ctgIdx(row)];
  PredictorT* blockCensus = &census[ctgIdx(row)];
  PredictorT argMax = nCtgTrain; // Unrealizeable.
  double scoreMax = 0.0; // Unrealizeable.
  for (PredictorT ctg = 0; ctg < nCtgTrain; ctg++) {
    double ctgScore = blockVotes[ctg]; // Jittered vote count.
    blockCensus[ctg] = ctgScore; // De-jittered.
    if (ctgScore > scoreMax) {
      scoreMax = ctgScore;
      argMax = ctg;
    }
  }
  return argMax;
}


const vector<double>  PredictReg::getQPred() const {
  return (quant == nullptr || quant->getNRow() == 0) ? vector<double>(0) : quant->getQPred();
}


const vector<double> PredictReg::getQEst() const {
  return (quant == nullptr || quant->getNRow() == 0) ? vector<double>(0) : quant->getQEst();
}


void Predict::walkNum(size_t row) {
  auto rowT = baseNum(row);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (!oob || !bag->isBagged(tIdx, row)) {
      rowNum(tIdx, rowT, row);
    }
  }
}


void Predict::walkFac(size_t row) {
  auto rowT = baseFac(row);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (!oob || !bag->isBagged(tIdx, row)) {
      rowFac(tIdx, rowT, row);
    }
  }
}


void Predict::walkMixed(size_t row) {
  const double* rowNT = baseNum(row);
  const PredictorT* rowFT = baseFac(row);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (!oob || !bag->isBagged(tIdx, row)) {
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
      (*confusionTarg)[ctgIdx(yTest[row], yPred[row])]++;
    }
    setMisprediction();
  }
}


void PredictCtg::setMisprediction() {
  size_t totRight = 0;
  for (unsigned int ctgRec = 0; ctgRec < nCtgMerged; ctgRec++) {
    size_t numWrong = 0;
    size_t numRight = 0;
    for (unsigned int ctgPred = 0; ctgPred < nCtgTrain; ctgPred++) {
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


CtgProb::CtgProb(PredictorT ctgTrain,
                 unsigned int nTree,
                 const unsigned int* leafHeight,
                 const double* prob) :
  nCtg(ctgTrain),
  probDefault(vector<double>(nCtg)),
  ctgHeight(scaleHeight(leafHeight, nTree)),
  raw(make_unique<Jagged3<const double*, const unsigned int*> >(nCtg, nTree, &ctgHeight[0], prob)) {
  setDefault();
}


vector<unsigned int> CtgProb::scaleHeight(const unsigned int* leafHeight,
                                          unsigned int nTree) const {
  vector<unsigned int> height(nTree);
  unsigned int i = 0;
  for (auto & ht : height) {
    ht = nCtg * leafHeight[i++];
  }

  return height;
}


void CtgProb::addLeaf(double* probRow,
                      unsigned int tIdx,
                      unsigned int leafIdx) const {
  size_t idxBase = raw->minorOffset(tIdx, leafIdx);
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    probRow[ctg] += raw->getItem(idxBase + ctg);
  }
}


void CtgProb::probAcross(const PredictCtg* predict,
			 size_t row,
                         double* probRow) const {
  unsigned int treesSeen = 0;
  for (auto tc = 0ul; tc < raw->getNMajor(); tc++) {
    IndexT termIdx;
    if (predict->isLeafIdx(row, tc, termIdx)) {
      treesSeen++;
      addLeaf(probRow, tc, termIdx);
    }
  }
  if (treesSeen == 0) {
    applyDefault(probRow);
  }
  else {
    double scale = 1.0 / treesSeen;
    for (auto ctg = 0ul; ctg < nCtg; ctg++)
      probRow[ctg] *= scale;
  }
}


void CtgProb::setDefault() {
  // Fastest-changing dimension is category.
  for (size_t idx = 0; idx < raw->size(); idx++) {
    probDefault[idx % nCtg] += raw->getItem(idx);
  }

  // Scales by recip leaf count.
  double scale = 1.0 / (raw->size() / nCtg);
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    probDefault[ctg] *= scale;
  }
}


unsigned int CtgProb::ctgDefault() const {
  unsigned int argMax = 0;
  double probMax = 0.0;
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    if (probDefault[ctg] > probMax) {
      probMax = probDefault[ctg];
      argMax = ctg;
    }
  }

  return argMax;  
}


void CtgProb::applyDefault(double *probPredict) const {
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    probPredict[ctg] = probDefault[ctg];
  }
}

