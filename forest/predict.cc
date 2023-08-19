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
#include "forestscorer.h"
#include "response.h"

#include <cmath>
const size_t Predict::scoreChunk = 0x2000;
const unsigned int Predict::seqChunk = 0x20;


Predict::Predict(const Forest* forest,
		 const Sampler* sampler_,
		 RLEFrame* rleFrame,
		 bool testing_,
		 const PredictOption& option) :
  trapUnobserved(option.trapUnobserved),
  sampler(sampler_),
  decNode(forest->getNode()),
  factorBits(forest->getFactorBits()),
  bitsObserved(forest->getBitsObserved()),
  testing(testing_),
  nPermute(option.nPermute),
  idxFinal(vector<IndexT>(scoreChunk * forest->getNTree())),
  accumNEst(vector<IndexT>(scoreChunk)),
  scoreBlock(forest->getTreeScores()),
  nPredNum(rleFrame->getNPredNum()),
  nPredFac(rleFrame->getNPredFac()),
  nObs(rleFrame->getNRow()),
  nTree(forest->getNTree()),
  noNode(forest->noNode()),
  walkObs(getObsWalker()),
  trFac(vector<CtgT>(scoreChunk * nPredFac)),
  trNum(vector<double>(scoreChunk * nPredNum)),
  indices(vector<size_t>(option.indexing ? nTree * nObs : 0)) {
  rleFrame->reorderRow(); // For now, all frames pre-ranked.
}


IndexT (Predict::* Predict::getObsWalker())(unsigned int, size_t) {
  if (nPredFac == 0)
    return &Predict::obsNum;
  else if (nPredNum == 0)
    return &Predict::obsFac;
  else
    return &Predict::obsMixed;
}


PredictReg::PredictReg(const Forest* forest,
		       const Sampler* sampler_,
		       const Leaf* leaf,
		       RLEFrame* rleFrame,
		       const vector<double>& yTest_,
		       const PredictOption& option,
		       vector<double> quantile) :
  Predict(forest, sampler_, rleFrame, !yTest_.empty(), option),
  response(reinterpret_cast<const ResponseReg*>(sampler->getResponse())),
  yTest(std::move(yTest_)),
  yPred(vector<double>(nObs)),
  yPermute(vector<double>(nPermute > 0 ? nObs : 0)),
  accumAbsErr(vector<double>(scoreChunk)),
  accumSSE(vector<double>(scoreChunk)),
  saePermute(nPermute > 0 ? rleFrame->getNPred() : 0),
  ssePermute(nPermute > 0 ? rleFrame->getNPred() : 0),
  yTarg(&yPred),
  saeTarg(&saePredict),
  sseTarg(&ssePredict) {
  scorer = forest->makeScorer(response, forest, leaf, this, std::move(quantile));
}


PredictCtg::PredictCtg(const Forest* forest,
		       const Sampler* sampler_,
		       RLEFrame* rleFrame,
		       const vector<PredictorT>& yTest_,
		       const PredictOption& option,
		       bool doProb) :
  Predict(forest, sampler_, rleFrame, !yTest_.empty(), option),
  response(reinterpret_cast<const ResponseCtg*>(sampler->getResponse())),
  yTest(std::move(yTest_)),
  yPred(vector<PredictorT>(nObs)),
  nCtgTrain(response->getNCtg()),
  nCtgMerged(testing ? 1 + *max_element(yTest.begin(), yTest.end()) : 0),
  // Can only predict trained categories, so census and
  // probability matrices have 'nCtgTrain' columns.
  yPermute(vector<PredictorT>(nPermute > 0 ? nObs : 0)),
  confusion(vector<size_t>(nCtgTrain * nCtgMerged)),
  misprediction(vector<double>(nCtgMerged)),
  oobPredict(0.0),
  censusPermute(vector<unsigned int>(nPermute > 0 ? nObs * nCtgTrain : 0)),
  confusionPermute(vector<size_t>(nPermute > 0 ? confusion.size() : 0)),
  mispredPermute(vector<vector<double>>(nPermute > 0 ? rleFrame->getNPred(): 0)),
  oobPermute(vector<double>(nPermute > 0 ? rleFrame->getNPred() : 0)),
  yTarg(&yPred),
  confusionTarg(&confusion),
  censusTarg(scorer->getCensusBase()),
  mispredTarg(&misprediction),
  oobTarg(&oobPredict) {
  scorer = forest->makeScorer(response, nObs, doProb);
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
    rleFrame->rlePred[predIdx] = rleFrame->permute(predIdx, Sample::permute(nObs));
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
  size_t row = predictBlock(rleFrame, 0, nObs, trIdx);
  // Remainder rows handled in custom-fitted block.
  if (nObs > row) {
    (void) predictBlock(rleFrame, row, nObs, trIdx);
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
    predictObs(blockRows);
  }
  
  return row;
}


void Predict::transpose(const RLEFrame* rleFrame,
			vector<size_t>& idxTr,
			size_t rowStart,
			size_t rowExtent) {
  CtgT* facOut = trFac.empty() ? nullptr : &trFac[0];
  double* numOut = trNum.empty() ? nullptr : &trNum[0];
  for (size_t row = rowStart; row != min(nObs, rowStart + rowExtent); row++) {
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


void Predict::predictObs(size_t span) {
  fill(idxFinal.begin(), idxFinal.end(), noNode);

  OMPBound rowEnd = static_cast<OMPBound>(blockStart + span);
  OMPBound rowStart = static_cast<OMPBound>(blockStart);

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound row = rowStart; row < rowEnd; row += seqChunk) {
    scoreSeq(row, min(rowEnd, row + seqChunk));
  }
  }
  if (!indices.empty()) { // Copies written portion of index block.
    copy(&idxFinal[0], &idxFinal[span * nTree], &indices[rowStart * nTree]);
  }
}


// Sequential inner loop to avoid false sharing.
void PredictReg::scoreSeq(size_t rowStart, size_t rowEnd) {
  for (size_t row = rowStart; row != rowEnd; row++) {
    walkTree(row);
    testing ? testObs(row) : (void) scoreObs(row);
  }
}


void PredictCtg::scoreSeq(size_t rowStart, size_t rowEnd) {
  for (size_t row = rowStart; row != rowEnd; row++) {
    walkTree(row);
    testing ? testObs(row) : (void) scoreObs(row);
  }
}


void PredictReg::testObs(size_t row) {
  IndexT rowIdx = row - blockStart;
  accumNEst[rowIdx] += scoreObs(row);
  double testError = fabs(yTest[row] - (*yTarg)[row]);
  accumAbsErr[rowIdx] += testError;
  accumSSE[rowIdx] += testError * testError;
}


void PredictCtg::testObs(size_t row) {
  (void) scoreObs(row);
}


const vector<double>&  PredictReg::getQPred() const {
  return scorer->getQPred();
}


const vector<double>& PredictReg::getQEst() const {
  return scorer->getQEst();
}


void Predict::walkTree(size_t obsIdx) {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    if (!sampler->isBagged(tIdx, obsIdx)) {
      predictLeaf(tIdx, obsIdx);
    }
  }
}


IndexT Predict::obsNum(unsigned int tIdx,
		       size_t obsIdx) {
  const vector<DecNode>& cTree = decNode[tIdx];
  IndexT idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = cTree[idx].advance(baseNum(obsIdx), trapUnobserved);
    idx += delIdx;
  } while (delIdx != 0);

  return idx;
}


IndexT Predict::obsFac(const unsigned int tIdx,
		       size_t obsIdx) {
  const vector<DecNode>& cTree = decNode[tIdx];
  IndexT idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = cTree[idx].advance(factorBits, bitsObserved, baseFac(obsIdx), tIdx, trapUnobserved);
    idx += delIdx;
  } while (delIdx != 0);

  return idx;
}


IndexT Predict::obsMixed(unsigned int tIdx,
			 size_t obsIdx) {
  const vector<DecNode>& cTree = decNode[tIdx];
  IndexT idx = 0;
  IndexT delIdx = 0;
  do {
    delIdx = cTree[idx].advance(this, factorBits, bitsObserved, baseFac(obsIdx), baseNum(obsIdx), tIdx, trapUnobserved);
    idx += delIdx;
  } while (delIdx != 0);

  return idx;
}


bool Predict::isLeafIdx(size_t row,
			unsigned int tIdx,
			IndexT& leafIdx) const {
    IndexT termIdx = idxFinal[nTree * (row - blockStart) + tIdx];
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
    for (size_t row = 0; row < nObs; row++) {
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
  *oobTarg = double(totRight) / nObs;
}


const vector<unsigned int>& PredictCtg::getCensus() const {
  return scorer->getCensus();
}


const vector<double>& PredictCtg::getProb() const {
  return scorer->getProb();
}


vector<double> Predict::forestWeight(const Forest* forest,
				     const Sampler* sampler,
				     const Leaf* leaf,
				     size_t nPredict,
				     const double finalIdx[],
				     unsigned int nThread) {
  vector<vector<double>> obsWeight(nPredict);
  for (size_t idxPredict = 0; idxPredict != nPredict; idxPredict++) {
    obsWeight[idxPredict] = vector<double>(sampler->getNObs());
  }

  for (unsigned int tIdx = 0; tIdx < forest->getNTree(); tIdx++) {
    vector<vector<IdCount>> node2Idc = obsCounts(forest, sampler, leaf, tIdx);
    weighNode(forest, &finalIdx[tIdx], node2Idc, obsWeight);
  }

  return normalizeWeight(sampler, obsWeight);
}


vector<vector<IdCount>> Predict::obsCounts(const Forest* forest,
					   const Sampler* sampler,
					   const Leaf* leaf,
					   unsigned int tIdx) {
  const vector<DecNode>& decNode = forest->getNode(tIdx);
  const vector<IdCount> idCount = sampler->unpack(tIdx);
  const vector<vector<size_t>>& indices = leaf->getIndices(tIdx);

  // Dominators need not be computed if it is known in advance
  // that all final indices are terminal.  This will be the case
  // if prediction does not employ trap-and-bail.
  vector<IndexRange> leafDom = forest->leafDominators(decNode);
  vector<vector<IdCount>> node2Idc(decNode.size());
  for (IndexT nodeIdx = 0; nodeIdx != decNode.size(); nodeIdx++) {
    IndexRange leafRange = leafDom[nodeIdx];
    for (IndexT leafIdx = leafRange.getStart(); leafIdx != leafRange.getEnd(); leafIdx++) {
      for (size_t sIdx : indices[leafIdx]) {
	node2Idc[nodeIdx].emplace_back(idCount[sIdx]);
      }
    }
  }

  return node2Idc;
}


void Predict::weighNode(const Forest* forest,
			const double treeIdx[],
			const vector<vector<IdCount>>& nodeCount,
			vector<vector<double>>& obsWeight) {
  IndexT noNode = forest->noNode(); // Excludes bagged observations.
  size_t finalPosition = 0; // Position of final indices for tree.
  for (vector<double>& nodeWeight : obsWeight) {
    IndexT nodeIdx = treeIdx[finalPosition];
    if (nodeIdx != noNode) {
      IndexT sampleCount = 0;
      for (const IdCount &idc : nodeCount[nodeIdx]) {
	sampleCount += idc.sCount;
      }

      double recipSCount = 1.0 / sampleCount;
      for (const IdCount& idc : nodeCount[nodeIdx]) {
	nodeWeight[idc.id] += idc.sCount * recipSCount;
      }
    }
    finalPosition += forest->getNTree();
  }
}


vector<double> Predict::normalizeWeight(const Sampler* sampler,
					const vector<vector<double>>& obsWeight) {
  size_t nObs = sampler->getNObs();
  //  vector<vector<double>> weight(obsWeight.size());
  vector<double> weight(obsWeight.size() * nObs);
  size_t idxPredict = 0;
  for (const vector<double>& obsW : obsWeight) {
    double weightRecip = 1.0 / accumulate(obsW.begin(), obsW.end(), 0.0);
    //weight[idxPredict] = vector<double>(obsW.size());
    // write to weight + idxPredict * nObs
    transform(obsW.begin(), obsW.end(), &weight[idxPredict * nObs],//weight[idxPredict].begin(),
                   [&weightRecip](double element) { return element * weightRecip; });
    idxPredict++;
  }
  return weight;
}


unsigned int PredictReg::scoreObs(size_t obsIdx) {
  return scorer->scoreObs(this, obsIdx, *yTarg);
}


unsigned int PredictCtg::scoreObs(size_t obsIdx) {
  return scorer->scoreObs(this, obsIdx, *yTarg);
}
