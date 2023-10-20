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
#include "decnode.h"
#include "quant.h"
#include "ompthread.h"
#include "rleframe.h"
#include "sample.h"

#include <cmath>

const size_t Predict::obsChunk = 0x2000;
const unsigned int Predict::seqChunk = 0x20;


bool Predict::bagging = false;
unsigned int Predict::nPermute = 0;


void Predict::init(bool bagging_,
		   unsigned int nPermute_) {
  bagging = bagging_;
  nPermute = nPermute_;
}


void Predict::deInit() {
  bagging = false;
  nPermute = 0;
}


Predict::Predict(const Sampler* sampler,
		 unique_ptr<RLEFrame> rleFrame_) :
  bag(sampler->bagRows(bagging)),
  rleFrame(std::move(rleFrame_)),
  nObs(rleFrame == nullptr ? 0 : rleFrame->getNRow()),
  trFrame(PredictFrame(rleFrame.get())) {
  if (rleFrame != nullptr) { // TEMPORARY
    rleFrame->reorderRow(); // For now, all frames pre-ranked.
  }
}


PredictReg::PredictReg(const Sampler* sampler,
		       unique_ptr<RLEFrame> rleFrame_) :
  Predict(sampler, std::move(rleFrame_)) {
}


unique_ptr<PredictReg> Predict::makeReg(const Sampler* sampler,
					unique_ptr<RLEFrame> rleFrame) {
  return make_unique<PredictReg>(sampler, std::move(rleFrame));
}


unique_ptr<PredictCtg> Predict::makeCtg(const Sampler* sampler,
					unique_ptr<RLEFrame> rleFrame) {
  return make_unique<PredictCtg>(sampler, std::move(rleFrame));
}


PredictCtg::PredictCtg(const Sampler* sampler, unique_ptr<RLEFrame> rleFrame_) :
  Predict(sampler, std::move(rleFrame_)) {
}


unique_ptr<SummaryReg> PredictReg::predictReg(const Sampler* sampler,
					      Forest* forest,
					      const vector<double>& yTest) {
  this->forest = forest;
  nTree = forest->getNTree();
  unique_ptr<SummaryReg> summary = make_unique<SummaryReg>(sampler, this, forest);
  summary->build(this, sampler, yTest);
  return summary;
}


unique_ptr<SummaryCtg> PredictCtg::predictCtg(const Sampler* sampler,
					      Forest* forest,
					      const vector<unsigned int>& yTest) {
  this->forest = forest;
  nTree = forest->getNTree();
  unique_ptr<SummaryCtg> summary = make_unique<SummaryCtg>(sampler, this, forest);
  summary->build(this, sampler, yTest);
  return summary;
}



SummaryReg::SummaryReg(const Sampler* sampler,
		       const Predict* predict,
		       Forest* forest) :
  prediction(forest->makePredictionReg(sampler, predict)) {
}


void SummaryReg::build(Predict* predictObj,
		       const Sampler* sampler,
		       const vector<double>& yTest) {
  predictObj->predict(prediction.get());
  test = prediction->test(yTest);
  permutationTest = permute(predictObj, sampler, yTest);
}


SummaryCtg::SummaryCtg(const Sampler* sampler,
		       const Predict* predict,
		       Forest* forest) :
  nCtgTrain(sampler->getNCtg()),
  prediction(forest->makePredictionCtg(sampler, predict)) {
}


void SummaryCtg::build(Predict* predictObj,
		       const Sampler* sampler,
		       const vector<unsigned int>& yTest) {
  predictObj->predict(prediction.get());
  test = prediction->test(yTest);
  permutationTest = permute(predictObj, sampler, yTest);
}


void Predict::predict(ForestPrediction* prediction) {
  blockStart = 0;
  forest->initWalkers(trFrame);
  idxFinal = vector<IndexT>(nTree * obsChunk);
  noNode = forest->getNoNode();
  
  predictBlock(prediction);
  // Remainder rows handled in custom-fitted block.
  if (nObs > blockStart) {
    predictBlock(prediction);
  }
}


void Predict::predictBlock(ForestPrediction* prediction) {
  size_t blockSpan = min(obsChunk, nObs - blockStart);
  for (; blockStart + blockSpan <= nObs; blockStart += blockSpan) {
    predictObs(prediction, blockSpan);
  }
}


void Predict::predictObs(ForestPrediction* prediction,
			 size_t span) {
  resetIndices();
  trFrame.transpose(rleFrame.get(), blockStart, span);

  OMPBound rowEnd = static_cast<OMPBound>(blockStart + span);
  OMPBound rowStart = static_cast<OMPBound>(blockStart);

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound row = rowStart; row < rowEnd; row += seqChunk) {
    size_t chunkEnd = min(rowEnd, row + seqChunk);
    walkTree(trFrame, row, chunkEnd);
    prediction->callScorer(this, row, chunkEnd);
  }
  }
  prediction->cacheIndices(idxFinal, span * nTree, blockStart * nTree);
}


void Predict::resetIndices() {
  fill(idxFinal.begin(), idxFinal.end(), noNode);
}


void Predict::walkTree(const PredictFrame& frame,
		       size_t obsStart,
		       size_t obsEnd) {
  for (size_t obsIdx = obsStart; obsIdx != obsEnd; obsIdx++) {
    for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
      if (!isBagged(tIdx, obsIdx)) {
	setFinalIdx(obsIdx, tIdx, forest->walkObs(frame, obsIdx, tIdx));
      }
    }
  }
}


bool Predict::isLeafIdx(size_t obsIdx,
			unsigned int tIdx,
			IndexT& leafIdx) const {
  IndexT nodeIdx;
  if (getFinalIdx(obsIdx, tIdx, nodeIdx))
    return forest->getLeafIdx(tIdx, nodeIdx, leafIdx);
  else
    return false;
}


bool Predict::isNodeIdx(size_t obsIdx,
			unsigned int tIdx,
			double& score) const {
  IndexT nodeIdx;
  if (getFinalIdx(obsIdx, tIdx, nodeIdx)) {
    score = forest->getScore(tIdx, nodeIdx);
    return true;
  }
  else {
    return false;
  }
    // Non-bagging scenarios should always see a leaf.
    //    if (!bagging) assert(termIdx != noNode);
}


vector<vector<unique_ptr<TestReg>>> SummaryReg::permute(const Predict* predict,
							const Sampler* sampler,
							const vector<double>& yTest) {
  if (yTest.empty() || Predict::nPermute == 0)
    return vector<vector<unique_ptr<TestReg>>>(0);

  RLEFrame* rleFrame = predict->getFrame();
  vector<vector<unique_ptr<TestReg>>> testPermute(rleFrame->getNPred());
  for (PredictorT predIdx = 0; predIdx < rleFrame->getNPred(); predIdx++) {
    vector<RLEVal<szType>> rleTemp = std::move(rleFrame->rlePred[predIdx]);
    for (unsigned int rep = 0; rep != Predict::nPermute; rep++) {
      rleFrame->rlePred[predIdx] = rleFrame->permute(predIdx, Sample::permute(rleFrame->getNRow()));
      unique_ptr<ForestPredictionReg> repReg = predict->forest->makePredictionReg(sampler, predict, false);
      testPermute[predIdx].emplace_back(repReg->test(yTest));
    }
    rleFrame->rlePred[predIdx] = std::move(rleTemp);
  }

  return testPermute;
}


vector<vector<unique_ptr<TestCtg>>> SummaryCtg::permute(const Predict* predict,
							const Sampler* sampler,
							const vector<unsigned int>& yTest) {
  if (yTest.empty() || Predict::nPermute == 0)
    return vector<vector<unique_ptr<TestCtg>>>(0);

  RLEFrame* rleFrame = predict->getFrame();
  vector<vector<unique_ptr<TestCtg>>> testPermute(rleFrame->getNPred());
  for (PredictorT predIdx = 0; predIdx < rleFrame->getNPred(); predIdx++) {
    vector<RLEVal<szType>> rleTemp = std::move(rleFrame->rlePred[predIdx]);
    for (unsigned int rep = 0; rep != Predict::nPermute; rep++) {
      rleFrame->rlePred[predIdx] = rleFrame->permute(predIdx, Sample::permute(rleFrame->getNRow()));
      unique_ptr<ForestPredictionCtg> repCtg = predict->forest->makePredictionCtg(sampler, predict, false);
      testPermute[predIdx].emplace_back(repCtg->test(yTest));
    }
    rleFrame->rlePred[predIdx] = std::move(rleTemp);
  }

  return testPermute;
}


double SummaryReg::getSSE() const {
  return test->SSE;
}


double SummaryReg::getSAE() const {
  return test->absError;
}


const vector<double>& SummaryReg::getYPred() const {
  return prediction->prediction.value;
}


vector<vector<double>> SummaryReg::getSSEPermuted() const {
  return test->getSSEPermuted(permutationTest);
}


vector<vector<double>> SummaryReg::getSAEPermuted() const {
  return test->getSAEPermuted(permutationTest);
}


const vector<CtgT>& SummaryCtg::getYPred() const {
  return prediction->prediction.value;
}


const vector<size_t>& SummaryCtg::getConfusion() const {
  return test->confusion;
}


const vector<double>& SummaryCtg::getMisprediction() const {
  return test->misprediction;
}


double SummaryCtg::getOOBError() const {
  return test->oobErr;
}    


const vector<unsigned int>& SummaryCtg::getCensus() const {
  return prediction->census;
}


const vector<double>& SummaryCtg::getProb() const {
  return prediction->getProb();
}


const vector<double>&  SummaryReg::getQPred() const {
  return prediction->getQPred();
}


const vector<double>& SummaryReg::getQEst() const {
  return prediction->getQEst();
}


vector<vector<vector<double>>> SummaryCtg::getMispredPermuted() const {
  return test->getMispredPermuted(permutationTest);
}


vector<vector<double>> SummaryCtg::getOOBErrorPermuted() const {
  return test->getOOBErrorPermuted(permutationTest);
}


vector<double> Predict::forestWeight(const Forest* forest,
				     const Sampler* sampler,
				     size_t nPredict,
				     const double finalIdx[]) {
  vector<vector<double>> obsWeight(nPredict);
  for (size_t idxPredict = 0; idxPredict != nPredict; idxPredict++) {
    obsWeight[idxPredict] = vector<double>(sampler->getNObs());
  }

  for (unsigned int tIdx = 0; tIdx < forest->getNTree(); tIdx++) {
    vector<vector<IdCount>> node2Idc = obsCounts(forest, sampler, tIdx);
    weighNode(forest, &finalIdx[tIdx], node2Idc, obsWeight);
  }

  return normalizeWeight(sampler, obsWeight);
}


vector<vector<IdCount>> Predict::obsCounts(const Forest* forest,
					   const Sampler* sampler,
					   unsigned int tIdx) {
  const Leaf& leaf = forest->getLeaf();
  const vector<DecNode>& decNode = forest->getNode(tIdx);
  const vector<IdCount> idCount = sampler->unpack(tIdx);
  const vector<vector<size_t>>& indices = leaf.getIndices(tIdx);

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
  IndexT noNode = forest->getNoNode(); // Excludes bagged observations.
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
  vector<double> weight(obsWeight.size() * nObs);
  size_t idxPredict = 0;
  for (const vector<double>& obsW : obsWeight) {
    double weightRecip = 1.0 / accumulate(obsW.begin(), obsW.end(), 0.0);
    transform(obsW.begin(), obsW.end(), &weight[idxPredict * nObs],
                   [&weightRecip](double element) { return element * weightRecip; });
    idxPredict++;
  }
  return weight;
}

