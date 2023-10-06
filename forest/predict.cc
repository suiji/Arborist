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


unsigned int Predict::nPermute = 0;
size_t Predict::nObs = 0;
unsigned int Predict::nTree = 0;


void Predict::init(unsigned int nPermute_) {
  nPermute = nPermute_;
}


void Predict::deInit() {
  nPermute = 0;
  nObs = 0;
  nTree = 0;
}


Predict::Predict(unique_ptr<RLEFrame> rleFrame_) :
  rleFrame(std::move(rleFrame_)) {
  if (rleFrame != nullptr) { // TEMPORARY
  rleFrame->reorderRow(); // For now, all frames pre-ranked.
  nObs = rleFrame->getNRow();
  }
}


PredictReg::PredictReg(unique_ptr<RLEFrame> rleFrame_) :
  Predict(std::move(rleFrame_)) {
}


unique_ptr<PredictReg> Predict::makeReg(unique_ptr<RLEFrame> rleFrame) {
  return make_unique<PredictReg>(std::move(rleFrame));
}


unique_ptr<PredictCtg> Predict::makeCtg(unique_ptr<RLEFrame> rleFrame) {
  return make_unique<PredictCtg>(std::move(rleFrame));
}


PredictCtg::PredictCtg(unique_ptr<RLEFrame> rleFrame_) :
  Predict(std::move(rleFrame_)) {
}


unique_ptr<SummaryReg> PredictReg::predictReg(const Sampler* sampler,
					      Forest* forest,
					      const vector<double>& yTest) {
  nTree = forest->getNTree();
  return make_unique<SummaryReg>(sampler, yTest, forest, rleFrame.get());
}


unique_ptr<SummaryCtg> PredictCtg::predictCtg(const Sampler* sampler,
					      Forest* forest,
					      const vector<unsigned int>& yTest) {
  nTree = forest->getNTree();
  return make_unique<SummaryCtg>(sampler, yTest, forest, rleFrame.get());
}



SummaryReg::SummaryReg(const Sampler* sampler,
		       const vector<double>& yTest,
		       Forest* forest,
		       RLEFrame* rleFrame) :
  prediction(forest->predictReg(sampler, rleFrame)),
  test(prediction->test(yTest)),
  permutationTest(permute(sampler, rleFrame, forest, yTest)) {	     
}


SummaryCtg::SummaryCtg(const Sampler* sampler,
		       const vector<unsigned int>& yTest,
		       Forest* forest,
		       RLEFrame* rleFrame) :
  nCtgTrain(sampler->getNCtg()),
  prediction(forest->predictCtg(sampler, rleFrame)),
  test(prediction->test(yTest)),
  permutationTest(permute(sampler, rleFrame, forest, yTest)) {	     
}


vector<vector<unique_ptr<TestReg>>> SummaryReg::permute(const Sampler* sampler,
							RLEFrame* rleFrame,
							Forest* forest,
							const vector<double>& yTest) {
  if (yTest.empty() || Predict::nPermute == 0)
    return vector<vector<unique_ptr<TestReg>>>(0);

  vector<vector<unique_ptr<TestReg>>> testPermute(rleFrame->getNPred());
  for (PredictorT predIdx = 0; predIdx < rleFrame->getNPred(); predIdx++) {
    vector<RLEVal<szType>> rleTemp = std::move(rleFrame->rlePred[predIdx]);
    for (unsigned int rep = 0; rep != Predict::nPermute; rep++) {
      rleFrame->rlePred[predIdx] = rleFrame->permute(predIdx, Sample::permute(rleFrame->getNRow()));
      unique_ptr<ForestPredictionReg> repReg = forest->predictReg(sampler, rleFrame);
      testPermute[predIdx].emplace_back(repReg->test(yTest));
    }
    rleFrame->rlePred[predIdx] = std::move(rleTemp);
  }

  return testPermute;
}


vector<vector<unique_ptr<TestCtg>>> SummaryCtg::permute(const Sampler* sampler,
							RLEFrame* rleFrame,
							Forest* forest,
							const vector<unsigned int>& yTest) {
  if (yTest.empty() || Predict::nPermute == 0)
    return vector<vector<unique_ptr<TestCtg>>>(0);

  vector<vector<unique_ptr<TestCtg>>> testPermute(rleFrame->getNPred());
  for (PredictorT predIdx = 0; predIdx < rleFrame->getNPred(); predIdx++) {
    vector<RLEVal<szType>> rleTemp = std::move(rleFrame->rlePred[predIdx]);
    for (unsigned int rep = 0; rep != Predict::nPermute; rep++) {
      rleFrame->rlePred[predIdx] = rleFrame->permute(predIdx, Sample::permute(rleFrame->getNRow()));
      unique_ptr<ForestPredictionCtg> repCtg = forest->predictCtg(sampler, rleFrame);
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

  /**
     @return handle to cached index vector.
   */
const vector<size_t>& SummaryCtg::getIndices() const {
  return prediction->idxFinal;
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
