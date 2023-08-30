// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forestscorer.cc

   @brief Dispatches forest scorer.

   @author Mark Seligman
*/

#include "forestscorer.h"
#include "scoredesc.h"
#include "response.h"
#include "predict.h"
#include "quant.h"


map<const string, function<ForestScore(ForestScorer*, Predict*, size_t)>> ForestScorer::scorerTable {
  {"mean", &ForestScorer::predictMean},
  {"plurality", &ForestScorer::predictPlurality},
  {"sum", &ForestScorer::predictSum},
  {"logistic", &ForestScorer::predictLogistic}
};


ForestScorer::ForestScorer(const ScoreDesc* scoreDesc,
			   const ResponseReg* response,
			   const Forest* forest,
			   const Leaf* leaf,
			   const PredictReg* predict,
			   vector<double> quantile) :
  nu(scoreDesc->nu),
  baseScore(scoreDesc->baseScore),
  nCtg(response->getNCtg()),
  defaultPrediction(response->getDefaultPrediction()),
  scorer(scorerTable[scoreDesc->scorer]),
  quant(make_unique<Quant>(forest, leaf, predict, response, std::move(quantile))) {
}


ForestScorer::ForestScorer(const ScoreDesc* scoreDesc,
			   const ResponseCtg* response,
			   size_t nObs,
			   bool doProb) :
  nu(scoreDesc->nu),
  baseScore(scoreDesc->baseScore),
  nCtg(response->getNCtg()),
  defaultPrediction(response->getDefaultPrediction()),
  scorer(scorerTable[scoreDesc->scorer]),
  census(vector<unsigned int>(nObs * nCtg)),
  ctgProb(make_unique<CtgProb>(nObs, response, doProb)) {
}


unsigned ForestScorer::scoreObs(class PredictReg* predict, size_t obsIdx, vector<double>& yTarg) {
  ForestScore score = scoreObs(predict, obsIdx);
  yTarg[obsIdx] = score.score.num;

  // Relies on yTarg[] having been set:
  quant->predictRow(predict, obsIdx);

  return score.nEst;
}
  

unsigned ForestScorer::scoreObs(class PredictCtg* predict, size_t obsIdx, vector<CtgT>& yTarg) {
  ForestScore score = scoreObs(predict, obsIdx);
  yTarg[obsIdx] = score.score.ctg;
  return score.nEst;
}
  

ForestScore ForestScorer::predictMean(Predict* predict, size_t obsIdx) const {
  double sumScore = 0.0;
  unsigned int nEst = 0;
  for (unsigned int tIdx = 0; tIdx < predict->getNTree(); tIdx++) {
    double score;
    if (predict->isNodeIdx(obsIdx, tIdx, score)) {
      nEst++;
      sumScore += score;
    }
  }

  return ForestScore(nEst, nEst > 0 ? sumScore / nEst : defaultPrediction);
}


ForestScore ForestScorer::predictSum(Predict* predict, size_t obsIdx) const {
  double sumScore = baseScore;
  unsigned int nEst = 0;
  for (unsigned int tIdx = 0; tIdx < predict->getNTree(); tIdx++) {
    double score;
    if (predict->isNodeIdx(obsIdx, tIdx, score)) {
      sumScore += nu * score;
      nEst++;
    }
  }
  return ForestScore(nEst, sumScore);
}


ForestScore ForestScorer::predictLogistic(Predict* predict, size_t obsIdx) {
  ForestScore logOdds = predictSum(predict, obsIdx);
  double p1 = 1.0 / (1.0 + exp(-logOdds.score.num));
  ctgProb->assignBinary(obsIdx, p1);
  CtgT ctg = p1 > 0.5 ? 1 : 0;
  census[obsIdx * nCtg + ctg] = 1;

  return ForestScore(logOdds.nEst, ctg);
}


ForestScore ForestScorer::predictPlurality(Predict* predict, size_t obsIdx) {
  unsigned int nEst = 0; // # participating trees.
  vector<double> ctgJitter(nCtg); // Accumulates jitter by category.
  unsigned int *censusRow = &census[obsIdx * nCtg];
  for (unsigned int tIdx = 0; tIdx < predict->getNTree(); tIdx++) {
    double score;
    if (predict->isNodeIdx(obsIdx, tIdx, score)) {
      nEst++;
      CtgT ctg = floor(score); // Truncates jittered score ut index.
      censusRow[ctg]++;
      ctgJitter[ctg] += score - ctg;
    }
  }
  if (nEst == 0) { // Default category unity, all others zero.
    censusRow[CtgT(defaultPrediction)] = 1;
  }

  ctgProb->predictRow(obsIdx, censusRow);

  return ForestScore(nEst, argMaxJitter(censusRow, ctgJitter));
}


CtgT ForestScorer::argMaxJitter(const unsigned int censusRow[],
				const vector<double>& ctgJitter) const {
  CtgT argMax = 0;
  IndexT countMax = 0;
  // Assumes at least one slot has nonzero count.
  for (CtgT ctg = 0; ctg < ctgJitter.size(); ctg++) {
    IndexT count = censusRow[ctg];
    if (count == 0)
      continue;
    else if (count > countMax) {
      countMax = count;
      argMax = ctg;
    }
    else if (count == countMax) {
      if (ctgJitter[ctg] > ctgJitter[argMax]) {
	argMax = ctg;
      }
    }
  }
  return argMax;
}



CtgProb::CtgProb(size_t nObs,
		 const ResponseCtg* response,
		 bool doProb) :
  nCtg(response->getNCtg()),
  probDefault(response->ctgProb()),
  probs(vector<double>(doProb ? nObs * nCtg : 0)) {
}


void CtgProb::predictRow(size_t obsIdx,  const unsigned int censusRow[]) {
  if (isEmpty())
    return;

  unsigned int nEst = accumulate(censusRow, censusRow + nCtg, 0ul);
  double* probRow = &probs[obsIdx * nCtg];
  if (nEst == 0) {
    applyDefault(probRow);
  }
  else {
    double scale = 1.0 / nEst;
    for (PredictorT ctg = 0; ctg < nCtg; ctg++)
      probRow[ctg] = censusRow[ctg] * scale;
  }
}

void CtgProb::assignBinary(size_t obsIdx, double p1) {
  double* probRow = &probs[obsIdx * 2];
  probRow[0] = 1.0 - p1;
  probRow[1] = p1;
}


void CtgProb::applyDefault(double probPredict[]) const {
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    probPredict[ctg] = probDefault[ctg];
  }
}


const vector<double>& ForestScorer::getProb() const {
  return ctgProb->getProb();
}


const vector<double>& ForestScorer::getQPred() const {
  return quant->getQPred();
}


const vector<double>& ForestScorer::getQEst() const {
  return quant->getQEst();
}
