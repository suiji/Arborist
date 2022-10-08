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

#include "leaf.h"
#include "sampler.h"
#include "forest.h"
#include "predict.h"
#include "bv.h"
#include "treenode.h"
#include "quant.h"
#include "ompthread.h"
#include "rleframe.h"
#include "sample.h"
#include "response.h"

#include <cmath>
const size_t Predict::scoreChunk = 0x2000;
const unsigned int Predict::seqChunk = 0x20;


Predict::Predict(const Forest* forest,
		 const Sampler* sampler_,
		 RLEFrame* rleFrame,
		 bool testing_,
		 unsigned int nPermute_,
		 bool trapUnobserved_) :
  trapUnobserved(trapUnobserved_),
  sampler(sampler_),
  decNode(forest->getNode()),
  factorBits(forest->getFactorBits()),
  bitsObserved(forest->getBitsObserved()),
  testing(testing_),
  nPermute(nPermute_),
  predictLeaves(vector<IndexT>(scoreChunk * forest->getNTree())),
  accumNEst(vector<IndexT>(scoreChunk)),
  scoreBlock(forest->getTreeScores()),
  nPredNum(rleFrame->getNPredNum()),
  nPredFac(rleFrame->getNPredFac()),
  nRow(rleFrame->getNRow()),
  nTree(forest->getNTree()),
  noNode(forest->maxTreeHeight()),
  walkTree(getWalker()),
  trFac(vector<CtgT>(scoreChunk * nPredFac)),
  trNum(vector<double>(scoreChunk * nPredNum)) {
  rleFrame->reorderRow(); // For now, all frames pre-ranked.
}

void (Predict::* Predict::getWalker())(size_t) {
  if (nPredFac == 0)
    return &Predict::walkNum;
  else if (nPredNum == 0)
    return &Predict::walkFac;
  else
    return &Predict::walkMixed;
}


PredictReg::PredictReg(const Forest* forest,
		       const Sampler* sampler_,
		       const Leaf* leaf,
		       RLEFrame* rleFrame,
		       const vector<double>& yTest_,
		       unsigned int nPermute_,
		       const vector<double>& quantile,
		       bool trapUnobserved_) :
  Predict(forest, sampler_, rleFrame, !yTest_.empty(), nPermute_, trapUnobserved_),
  response(reinterpret_cast<const ResponseReg*>(sampler->getResponse())),
  yTest(std::move(yTest_)),
  yPred(vector<double>(nRow)),
  yPermute(vector<double>(nPermute > 0 ? nRow : 0)),
  accumAbsErr(vector<double>(scoreChunk)),
  accumSSE(vector<double>(scoreChunk)),
  saePermute(nPermute > 0 ? rleFrame->getNPred() : 0),
  ssePermute(nPermute > 0 ? rleFrame->getNPred() : 0),
  quant(make_unique<Quant>(forest, leaf, this, response, std::move(quantile))),
  yTarg(&yPred),
  saeTarg(&saePredict),
  sseTarg(&ssePredict) {
}


PredictCtg::PredictCtg(const Forest* forest,
		       const Sampler* sampler_,
		       RLEFrame* rleFrame,
		       const vector<PredictorT>& yTest_,
		       unsigned int nPermute_,
		       bool doProb,
		       bool trapUnobserved_) :
  Predict(forest, sampler_, rleFrame, !yTest_.empty(), nPermute_, trapUnobserved_),
  response(reinterpret_cast<const ResponseCtg*>(sampler->getResponse())),
  yTest(std::move(yTest_)),
  yPred(vector<PredictorT>(nRow)),
  nCtgTrain(response->getNCtg()),
  nCtgMerged(testing ? 1 + *max_element(yTest.begin(), yTest.end()) : 0),
  ctgProb(make_unique<CtgProb>(this, response, doProb)),
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


void Predict::predict(RLEFrame* rleFrame) {
  blocks(rleFrame);
  predictPermute(rleFrame);
}


void Predict::predictPermute(RLEFrame* rleFrame) {
  if (nPermute == 0) {
    return;
  }
  
  for (PredictorT predIdx = 0; predIdx < rleFrame->getNPred(); predIdx++) {
    setPermuteTarget(predIdx);
    vector<RLEVal<szType>> rleTemp = std::move(rleFrame->rlePred[predIdx]);
    rleFrame->rlePred[predIdx] = rleFrame->permute(predIdx, Sample::permute(nRow));
    blocks(rleFrame);
    rleFrame->rlePred[predIdx] = std::move(rleTemp);
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


void Predict::blocks(const RLEFrame* rleFrame) {
  vector<size_t> trIdx(nPredNum + nPredFac);
  size_t row = predictBlock(rleFrame, 0, nRow, trIdx);
  // Remainder rows handled in custom-fitted block.
  if (nRow > row) {
    (void) predictBlock(rleFrame, row, nRow, trIdx);
  }

  estAccum();
}


size_t Predict::predictBlock(const RLEFrame* rleFrame,
			     size_t rowStart,
			     size_t rowEnd,
			     vector<size_t>& trIdx) {
  size_t blockRows = min(scoreChunk, rowEnd - rowStart);
  size_t row = rowStart;
  for (; row + blockRows <= rowEnd; row += blockRows) {
    transpose(rleFrame, trIdx, row, scoreChunk);
    blockStart = row; // Not local.
    predictBlock(blockRows);
  }
  return row;
}


void Predict::transpose(const RLEFrame* rleFrame,
			vector<size_t>& idxTr,
			size_t rowStart,
			size_t rowExtent) {
  CtgT* facOut = trFac.empty() ? nullptr : &trFac[0];
  double* numOut = trNum.empty() ? nullptr : &trNum[0];
  for (size_t row = rowStart; row != min(nRow, rowStart + rowExtent); row++) {
    unsigned int numIdx = 0;
    unsigned int facIdx = 0;
    vector<szType> rankVec = rleFrame->idxRank(idxTr, row);
    for (unsigned int predIdx = 0; predIdx < rankVec.size(); predIdx++) {
      unsigned int rank = rankVec[predIdx];
      if (rleFrame->factorTop[predIdx] == 0) {
	*numOut++ = rleFrame->numRanked[numIdx++][rank];
      }
      else {// TODO:  Replace subtraction with (front end)::fac2Rank()
	*facOut++ = rleFrame->facRanked[facIdx++][rank] - 1;
      }
    }
  }
}


void Predict::predictBlock(size_t span) {
  fill(predictLeaves.begin(), predictLeaves.end(), noNode);

  OMPBound rowEnd = static_cast<OMPBound>(blockStart + span);
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
  (*yTarg)[row] = response->predictObs(this, row);
  if (!quant->isEmpty()) {
    quant->predictRow(this, row);
  }
  return nEst;
}


void PredictCtg::scoreRow(size_t row) {
  (*yTarg)[row] = response->predictObs(this, row, &census[ctgIdx(row)]);
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
  const CtgT* rowT = baseFac(row);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (!sampler->isBagged(tIdx, row)) {
      rowFac(tIdx, rowT, row);
    }
  }
}


void Predict::walkMixed(size_t row) {
  const double* rowNT = baseNum(row);
  const CtgT* rowFT = baseFac(row);
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (!sampler->isBagged(tIdx, row)) {
      rowMixed(tIdx, rowNT, rowFT, row);
    }
  }
}


void Predict::rowNum(unsigned int tIdx,
		       const double* rowT,
		       size_t row) {
  const vector<DecNode>& cTree = decNode[tIdx];
  auto idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = cTree[idx].advance(rowT, trapUnobserved);
    idx += delIdx;
  } while (delIdx != 0);

  predictLeaf(row, tIdx, idx);
}


void Predict::rowFac(const unsigned int tIdx,
		     const CtgT* rowT,
		     size_t row) {
  const vector<DecNode>& cTree = decNode[tIdx];
  IndexT idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = cTree[idx].advance(factorBits, bitsObserved, rowT, tIdx, trapUnobserved);
    idx += delIdx;
  } while (delIdx != 0);

  predictLeaf(row, tIdx, idx);
}


void Predict::rowMixed(unsigned int tIdx,
		       const double* rowNT,
		       const CtgT* rowFT,
		       size_t row) {
  const vector<DecNode>& cTree = decNode[tIdx];
  auto idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = cTree[idx].advance(this, factorBits, bitsObserved, rowFT, rowNT, tIdx, trapUnobserved);
    idx += delIdx;
  } while (delIdx != 0);

  predictLeaf(row, tIdx, idx);
}


bool Predict::isLeafIdx(size_t row,
			unsigned int tIdx,
			IndexT& leafIdx) const {
    IndexT termIdx = predictLeaves[nTree * (row - blockStart) + tIdx];
    return termIdx == noNode ? false : decNode[tIdx][termIdx].getLeafIdx(leafIdx);
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


void PredictCtg::dump() const {
  // TODO
  //  ctgProb->dump(probTree);
}


CtgProb::CtgProb(const Predict* predict,
		 const ResponseCtg* response,
		 bool doProb) :
  nCtg(response->getNCtg()),
  probDefault(response->defaultProb()),
  probs(vector<double>(doProb ? predict->getNRow() * nCtg : 0)) {
}


void CtgProb::predictRow(const Predict* predict, size_t row, PredictorT* ctgRow) {
  unsigned int nEst = accumulate(ctgRow, ctgRow + nCtg, 0ul);
  double* probRow = &probs[row * nCtg];
  if (nEst == 0) {
    applyDefault(probRow);
  }
  else {
    double scale = 1.0 / nEst;
    for (PredictorT ctg = 0; ctg < nCtg; ctg++)
      probRow[ctg] = ctgRow[ctg] * scale;
  }
}


void CtgProb::applyDefault(double probPredict[]) const {
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    probPredict[ctg] = probDefault[ctg];
  }
}


void CtgProb::dump() const {
  // TODO
}
